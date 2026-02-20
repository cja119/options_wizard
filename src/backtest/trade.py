"""
Trade class definitions
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import logging

from data.trade import (
    TransactionCostModel,
    AccountingConvention,
    Equity,
    Cashflow,
    PositionType,
)

from data.date import DateObj

SPREAD_CAPTURE = 1.0

if TYPE_CHECKING:
    from data.date import DateObj
    from data.trade import EntryData, BaseUnderlying, BaseTradeFeatures


class Trade:
    def __init__(
        self,
        entry_data: EntryData,
        transaction_cost_model: TransactionCostModel = TransactionCostModel.NONE,
        accounting_type: AccountingConvention = AccountingConvention.CASH,
    ) -> None:
        self.entry_data = entry_data
        self.accounting_type = accounting_type
        self.transaction_cost_model = transaction_cost_model
        self._opened = False
        self._closed = False
        self._tick = entry_data.tick
        self._spread_capture = SPREAD_CAPTURE

    # ---- External Interface ---- #
    @property
    def entry_cost(self) -> float:
        price: BaseUnderlying = self.entry_data.price_series.prices[
            self.entry_data.entry_date.to_iso()
        ]

        return self._cash_position(price)

    def __call__(
        self, date: DateObj, spread_capture: float = SPREAD_CAPTURE
    ) -> Tuple[Equity | None, Cashflow | None]:

        if spread_capture != SPREAD_CAPTURE:
            self._spread_capture = spread_capture

        if date not in self.entry_data.price_series.prices:
            date = self._nearest_date_below(date)

        if not self._live_condition(date):
            return None, None

        cashflow = None

        if self._open_condition(date):
            cashflow: Cashflow = self._initialize_trade(date)
        elif self._close_condition(date):
            cashflow: Cashflow = self._close_trade()
            equity: Equity | None = None
            return equity, cashflow

        if self.accounting_type == AccountingConvention.MTM:
            equity: Equity = self._mtm_equity(date)

        elif self.accounting_type == AccountingConvention.CASH:
            equity: Equity = self._cash_equity(date)

        return equity, cashflow

    @property
    def features(self) -> "BaseTradeFeatures" | None:
        return self.entry_data.features

    def is_open_on(self, date: DateObj) -> bool:
        return (
            self._open_condition(date)
            or self._live_condition(date)
            or self._close_condition(date)
        )

    def close(self, date: DateObj) -> Tuple[Equity, Cashflow]:
        date = self._nearest_date_above(date)
        self.entry_data.exit_date = date
        if (not self._opened) and date == self.entry_data.entry_date:
            
            entry_cashflow = self._initialize_trade(date)
            exit_cashflow = self._close_trade()
            
            logging.warning(
                "Entering trade exiting on same-day "
                f"entry={self.entry_data.entry_date.to_iso()} | "
                f"exit={date.to_iso()} | "
                f"pos={self.entry_data.position_type.name} | "
                f"size={self.entry_data.position_size}",
                extra={"tick_name": self._tick}
            )

            net_amount = 0.0
            if entry_cashflow is not None:
                net_amount += entry_cashflow.amount
            if exit_cashflow is not None:
                net_amount += exit_cashflow.amount
            cashflow = Cashflow(
                date=date,
                amount=net_amount,
                accounting_convention=self.accounting_type,
                parent_trade=self,
            )
            return None, cashflow
        equity, cashflow = self(date)
        return equity, cashflow

    def is_entering_on(self, date: DateObj) -> bool:
        return self._open_condition(date)

    def days_open(self, current_date: DateObj) -> int:
        if not self._live_condition(current_date):
            return 0
        delta = current_date - self.entry_data.entry_date
        return delta + 1

    # ---- Internal Methods ---- #
    def _calculate_tcs(self, price: BaseUnderlying) -> float:
        if self.transaction_cost_model == TransactionCostModel.NONE:
            return 0.0
        if self.transaction_cost_model == TransactionCostModel.SPREAD:
            spread = (price.ask - price.bid) * abs(self.entry_data.position_size)
            return (spread / 2) * (1 - self._spread_capture)
        else:
            return 0.0

    def _mid_price(self, price: BaseUnderlying) -> float:
        return (price.bid + price.ask) / 2

    def _half_spread_adj(self, price: BaseUnderlying, is_buy: bool) -> float:
        if self.transaction_cost_model != TransactionCostModel.SPREAD:
            return 0.0
        half_spread = (price.ask - price.bid) / 2 * (1 - self._spread_capture)
        return half_spread if is_buy else -half_spread

    def _execution_price(self, price: BaseUnderlying, is_buy: bool) -> float:
        return self._mid_price(price) + self._half_spread_adj(price, is_buy)

    def _mtm_equity(self, date: DateObj) -> Equity:
        price: BaseUnderlying = self.entry_data.price_series.prices[date.to_iso()]
        equity = Equity(
            date=date,
            value=self._price_exposure(price) - self._entry_exposure,
            accounting_convention=AccountingConvention.MTM,
            parent_trade=self,
        )
        return equity

    @property
    def _pos_sign(self) -> int:
        return 1 if self.entry_data.position_type == PositionType.LONG else -1

    def _cash_equity(self, date: DateObj) -> Equity:
        price: BaseUnderlying = self.entry_data.price_series.prices[date.to_iso()]
        ps: float = self.entry_data.position_size
        value: float = self._mid_price(price) * self._pos_sign * ps

        equity = Equity(
            date=date,
            value=value,
            accounting_convention=AccountingConvention.CASH,
            parent_trade=self,
        )

        return equity

    def _open_condition(self, date: DateObj) -> bool:
        return (
            date == self.entry_data.entry_date and not self._opened and not self._closed
        )

    def _close_condition(self, date: DateObj) -> bool:
        return date == self.entry_data.exit_date and not self._closed and self._opened

    def _live_condition(self, date: DateObj) -> bool:
        return (
            self.entry_data.entry_date <= date <= self.entry_data.exit_date
            and not self._closed
        )

    def _close_trade(self) -> Cashflow:
        price: BaseUnderlying = self.entry_data.price_series.prices[
            self.entry_data.exit_date.to_iso()
        ]
        if self.accounting_type == AccountingConvention.CASH:
            ps: float = self.entry_data.position_size
            is_buy = self._pos_sign < 0  # closing a short is a buy
            exec_price = self._execution_price(price, is_buy=is_buy)
            cashflow_amount = -exec_price * ps if is_buy else exec_price * ps
        elif self.accounting_type == AccountingConvention.MTM:
            fees = self._calculate_tcs(price)
            cashflow_amount = self._price_exposure(price) - self._entry_exposure - fees

        cashflow = Cashflow(
            date=self.entry_data.exit_date,
            amount=cashflow_amount,
            accounting_convention=self.accounting_type,
            parent_trade=self,
        )

        self._closed = True
        self._opened = False

        return cashflow

    def __rmul__(self, other: float) -> "Trade":
        if self._opened:
            raise RuntimeError("Cannot resize an already-open trade")
        self.entry_data.position_size *= other
        return self

    def __imul__(self, other: float) -> "Trade":
        if self._opened:
            raise RuntimeError("Cannot resize an already-open trade")
        self.entry_data.position_size *= other
        return self

    def _cash_position(self, price: BaseUnderlying) -> float:
        ps: float = self.entry_data.position_size
        is_buy = self._pos_sign > 0
        exec_price = self._execution_price(price, is_buy=is_buy)
        return -exec_price * ps if is_buy else exec_price * ps

    def _price_exposure(self, price: BaseUnderlying) -> float:
        ps: float = self.entry_data.position_size
        return self._mid_price(price) * self._pos_sign * ps

    def _initialize_trade(self, date: 'DateObj') -> Cashflow:

        price: BaseUnderlying = self.entry_data.price_series.prices[
            self.entry_data.entry_date.to_iso()
        ]

        if self.accounting_type == AccountingConvention.CASH:
            cashflow_amount = self._cash_position(price)

        elif self.accounting_type == AccountingConvention.MTM:
            self._entry_exposure = self._price_exposure(price)
            cashflow_amount = -self._calculate_tcs(price)

        cashflow = Cashflow(
            date=self.entry_data.entry_date,
            amount=cashflow_amount,
            accounting_convention=self.accounting_type,
            parent_trade=self,
        )

        self._opened = True
        self._closed = False

        logging.debug(
            f"Entering trade on {date.to_iso()}",
            extra={"tick_name": self._tick}
        )

        return cashflow

    def _nearest_date_below(self, date: DateObj, cycl_break: bool = False) -> DateObj:
        if date not in self.entry_data.price_series.prices:
            candidates = [
                DateObj.from_iso(d)
                for d in self.entry_data.price_series.prices.keys()
                if DateObj.from_iso(d) <= date
            ]
            if candidates:
                date = max(candidates)
            elif not cycl_break:
                return self._nearest_date_above(date, cycl_break=True)
        return date

    def _nearest_date_above(self, date: DateObj, cycl_break: bool = False) -> DateObj:
        if date not in self.entry_data.price_series.prices:
            candidates = [
                DateObj.from_iso(d)
                for d in self.entry_data.price_series.prices.keys()
                if DateObj.from_iso(d) >= date
            ]
            if candidates:
                date = min(candidates)
            elif not cycl_break:
                return self._nearest_date_below(date, cycl_break=True)
        return date
