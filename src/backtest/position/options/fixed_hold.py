from typing import List, Dict, Tuple
from collections import deque
from ..base import PositionBase, BackTestConfig

from data.date import DateObj
from data.trade import Cashflow
from backtest.trade import Trade
from data.trade import PositionType
from typing import override


class FixedHoldNotional(PositionBase):
    _carry = 0.0
    _notional_exposure = None
    _carry_window = 5

    def __init__(self, config: BackTestConfig) -> None:
        super().__init__(config)
        window = self._kwargs.get("carry_window", self._carry_window)
        try:
            window = int(window)
        except Exception:
            window = self._carry_window
        if window < 0:
            window = 0
        self._carry_window = window
        self._carry = deque(maxlen=window)

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:
        protected_notional = self._kwargs.get("protected_notional", 1_000_000)
        hold_period = self._kwargs.get("hold_period", 30)
        notional = protected_notional / (hold_period)

        # If nothing to enter today, roll the daily budget forward once
        if not entering_trades:
            if self._carry_window > 0:
                self._carry.append(notional)
            return {}

        # Deploy the accumulated budget across all trades entering today
        if self._carry_window > 0:
            notional += sum(self._carry)
            self._carry.clear()

        # Anchor sizing on short-leg exposure per ticker so each name gets the
        # same notional allocation on a given day, regardless of how many legs
        # it uses.
        short_exp_by_tick = {}
        for trade in entering_trades:
            if trade.entry_data.position_type != PositionType.SHORT:
                continue
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            if (
                entry_px is None
                or entry_px.delta is None
                or entry_px.underlying is None
            ):
                continue
            exp = abs(
                entry_px.delta
                * trade.entry_data.position_size
                * entry_px.underlying.ask
            )
            short_exp_by_tick[entry_px.tick] = (
                short_exp_by_tick.get(entry_px.tick, 0.0) + exp
            )

        if not short_exp_by_tick:
            # No meaningful anchor to size off today; roll budget forward.
            if self._carry_window > 0:
                self._carry.append(notional)
            return {trade: 0.0 for trade in entering_trades}

        per_tick_budget = notional / len(short_exp_by_tick)

        sizes = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            tick = entry_px.tick if entry_px is not None else None
            tick_exp = short_exp_by_tick.get(tick, 0.0)
            if tick_exp == 0.0:
                sizes[trade] = 0.0
                print("Trade ZERO size due to zero exposure anchor.")
                continue
            sizes[trade] = per_tick_budget / tick_exp

        return sizes

    @override
    def exit_trigger(
        self,
        live_trades: List[Trade],
        current_date: DateObj,
        scheduled_exits: List[Trade],
    ) -> Tuple[List[Trade], List[Cashflow]]:
        exit_cashflows = []
        trades_to_close = set()
        max_days_open = self._kwargs.get("hold_period", 30)
        for trade in live_trades:
            if trade.days_open(current_date) >= max_days_open:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)

        for trade in scheduled_exits:
            if trade not in trades_to_close:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)
        return trades_to_close, exit_cashflows
