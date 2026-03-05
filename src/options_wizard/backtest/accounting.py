"""
Accounting and transaction-cost models for backtest trades.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from options_wizard.data.trade import (
    AccountingConvention,
    Cashflow,
    EntryData,
    Equity,
    PositionType,
)
from options_wizard.data.contract import BaseUnderlying

if TYPE_CHECKING:
    from options_wizard.data.date import DateObj
    from options_wizard.backtest.trade import Trade


class BaseTCM(ABC):
    """Abstract definition for transaction-cost models."""

    def __call__(self, price: BaseUnderlying) -> float:
        return self.calculate_tcs(price)

    @abstractmethod
    def calculate_tcs(self, price: BaseUnderlying) -> float:
        """Return per-unit transaction cost in price units."""

    def configure(self, **kwargs) -> None:
        """Optional runtime configuration hook."""
        _ = kwargs


class NoTCM(BaseTCM):
    """No transaction costs."""

    def calculate_tcs(self, price: BaseUnderlying) -> float:
        _ = price
        return 0.0


class SpreadTCM(BaseTCM):
    """Half-spread cost model scaled by (1 - spread_capture)."""

    def __init__(self, **kwargs) -> None:
        self._spread_capture = float(kwargs.get("spread_capture", 1.0))

    def configure(self, **kwargs) -> None:
        if "spread_capture" in kwargs and kwargs["spread_capture"] is not None:
            self._spread_capture = float(kwargs["spread_capture"])

    def calculate_tcs(self, price: BaseUnderlying) -> float:
        spread = max(float(price.ask) - float(price.bid), 0.0)
        capture = min(max(self._spread_capture, 0.0), 1.0)
        return (spread / 2.0) * (1.0 - capture)


class BPSTCM(BaseTCM):
    """Transaction cost as spread in basis points of mid price."""

    def __init__(self, **kwargs) -> None:
        self._spread = float(kwargs.get("spread_bps", 0.0)) / 10_000.0

    def configure(self, **kwargs) -> None:
        if "spread_bps" in kwargs and kwargs["spread_bps"] is not None:
            self._spread = float(kwargs["spread_bps"]) / 10_000.0

    def calculate_tcs(self, price: BaseUnderlying) -> float:
        mid = (float(price.bid) + float(price.ask)) / 2.0
        return mid * self._spread


class BaseAccountingModel(ABC):
    def __init__(self, entry_data: EntryData, tcm: BaseTCM, parent_trade: Trade) -> None:
        self._tcm = tcm
        self._et = entry_data
        self._ps = entry_data.price_series
        self._pt = parent_trade

    @abstractmethod
    def open_trade(self, date: DateObj) -> Cashflow:
        raise NotImplementedError

    @abstractmethod
    def close_trade(self, date: DateObj) -> Cashflow:
        raise NotImplementedError

    @abstractmethod
    def live_equity(self, date: DateObj) -> Equity:
        raise NotImplementedError

    def _price_on(self, date: DateObj) -> BaseUnderlying:
        return self._ps.prices[date.to_iso()]

    @property
    def _pos_sign(self) -> int:
        return 1 if self._et.position_type == PositionType.LONG else -1

    @property
    def _entry_date(self) -> DateObj:
        return self._et.entry_date

    @property
    def _position_size(self) -> float:
        return self._et.position_size

    def _mid_price(self, price: BaseUnderlying) -> float:
        return (float(price.bid) + float(price.ask)) / 2.0

    def _signed_notional(self, price: BaseUnderlying) -> float:
        return self._mid_price(price) * self._pos_sign * self._position_size

    def _fee_per_unit(self, price: BaseUnderlying) -> float:
        return self._tcm(price)

    def _fee_total(self, price: BaseUnderlying) -> float:
        return self._fee_per_unit(price) * self._position_size

    def _execution_price(self, price: BaseUnderlying, is_buy: bool) -> float:
        fee = self._fee_per_unit(price)
        return self._mid_price(price) + fee if is_buy else self._mid_price(price) - fee

    def _cashflow_from_execution(self, price: BaseUnderlying, is_buy: bool) -> float:
        exec_price = self._execution_price(price, is_buy=is_buy)
        return -exec_price * self._position_size if is_buy else exec_price * self._position_size


class CashAccounting(BaseAccountingModel):
    ACCOUNTING_TYPE = AccountingConvention.CASH

    @override
    def open_trade(self, date: DateObj) -> Cashflow:
        price = self._price_on(date)
        is_buy = self._pos_sign > 0
        amount = self._cashflow_from_execution(price, is_buy=is_buy)
        return Cashflow(
            date=date,
            amount=amount,
            accounting_convention=self.ACCOUNTING_TYPE,
            parent_trade=self._pt,
        )

    @override
    def live_equity(self, date: DateObj) -> Equity:
        price = self._price_on(date)
        value = self._signed_notional(price)
        return Equity(
            date=date,
            value=value,
            accounting_convention=self.ACCOUNTING_TYPE,
            parent_trade=self._pt,
        )

    @override
    def close_trade(self, date: DateObj) -> Cashflow:
        price = self._price_on(date)
        is_buy = self._pos_sign < 0
        amount = self._cashflow_from_execution(price, is_buy=is_buy)
        return Cashflow(
            date=date,
            amount=amount,
            accounting_convention=self.ACCOUNTING_TYPE,
            parent_trade=self._pt,
        )


class MTMAccounting(BaseAccountingModel):
    ACCOUNTING_TYPE = AccountingConvention.MTM

    @override
    def open_trade(self, date: DateObj) -> Cashflow:
        price = self._price_on(date)
        amount = -self._fee_total(price)
        return Cashflow(
            date=date,
            amount=amount,
            accounting_convention=self.ACCOUNTING_TYPE,
            parent_trade=self._pt,
        )

    @override
    def live_equity(self, date: DateObj) -> Equity:
        live_price = self._price_on(date)
        entry_price = self._price_on(self._entry_date)
        value = self._signed_notional(live_price) - self._signed_notional(entry_price)
        return Equity(
            date=date,
            value=value,
            accounting_convention=self.ACCOUNTING_TYPE,
            parent_trade=self._pt,
        )

    @override
    def close_trade(self, date: DateObj) -> Cashflow:
        live_price = self._price_on(date)
        entry_price = self._price_on(self._entry_date)
        amount = (
            self._signed_notional(live_price)
            - self._signed_notional(entry_price)
            - self._fee_total(live_price)
        )
        return Cashflow(
            date=date,
            amount=amount,
            accounting_convention=self.ACCOUNTING_TYPE,
            parent_trade=self._pt,
        )
