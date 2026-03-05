"""
Trade class definitions
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING
import structlog

from options_wizard.data.trade import (
    TransactionCostModel,
    AccountingConvention,
    Equity,
    Cashflow,
    PositionType,
)

from options_wizard.data.date import DateObj

SPREAD_CAPTURE = 1.0

if TYPE_CHECKING:
    from options_wizard.data.date import DateObj
    from options_wizard.data.trade import EntryData, BaseUnderlying, BaseTradeFeatures

logger = structlog.get_logger(__name__)


class Trade:
    def __init__(
        self,
        entry_data: EntryData,
        accounting_model,
    ) -> None:
        self._opened = False
        self._closed = False

        self.entry_data = entry_data
        self._tick = entry_data.tick
        self._model = accounting_model(
            entry_data=entry_data,
            parent_trade=self
            )

        
    # ---- External Interface ---- #
    @property
    def entry_cost(self) -> float:
        date = self.entry_data.entry_date.to_iso()
        return self._model.open_trade(date)

    def __call__(self, date: DateObj ) -> Tuple[Equity | None, Cashflow | None]:


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

        equity: Equity = self._model.live_equity(date)

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
            
            entry_cashflow = self._model.open_trade(date)
            exit_cashflow = self._model.close_trade(date)
            
            logger.warning(
                "Entering trade exiting on same day",
                tick=self._tick,
                entry_date=self.entry_data.entry_date.to_iso(),
                exit_date=date.to_iso(),
                position_type=self.entry_data.position_type.name,
                position_size=self.entry_data.position_size,
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

    def _mid_price(self, price: BaseUnderlying) -> float:
        return (price.bid + price.ask) / 2

    def _half_spread_adj(self, price: BaseUnderlying, is_buy: bool) -> float:
        if self.transaction_cost_model != TransactionCostModel.SPREAD:
            return 0.0
        half_spread = (price.ask - price.bid) / 2 * (1 - self._spread_capture)
        return half_spread if is_buy else -half_spread

    def _execution_price(self, price: BaseUnderlying, is_buy: bool) -> float:
        return self._mid_price(price) + self._half_spread_adj(price, is_buy)

    @property
    def _pos_sign(self) -> int:
        return 1 if self.entry_data.position_type == PositionType.LONG else -1

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

        date = self.entry_data.exit_date

        cashflow = self._model.close_trade(date)

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

    def _initialize_trade(self, date: 'DateObj') -> Cashflow:

        cashflow = self._model.open_trade(date)

        self._opened = True
        self._closed = False

        logger.debug(
            "Entering trade",
            tick=self._tick,
            date=date.to_iso(),
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
