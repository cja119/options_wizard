from ..base import PositionBase
from typing import List, Dict, Tuple

from data.date import DateObj
from data.trade import Cashflow
from backtest.trade import Trade
from data.trade import PositionType
from typing import override

PERC_DOWNSIDE = 0.025


class CappedDownsideTrade(PositionBase):
    _carry = 0.0
    _notional_exposure = None

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:

        te = self._snapshot.total_equity + self._snapshot.total_cash
        downside_cap = te * PERC_DOWNSIDE

        spread_cost = {}
        for trade in entering_trades:
            if spread_cost.get(trade.entry_data.tick) is None:
                spread_cost[trade.entry_data.tick] = 0.0

            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            if entry_px is None or entry_px.underlying is None:
                continue

            cost = trade.entry_cost * abs(trade.entry_data.position_size)
            spread_cost[trade.entry_data.tick] += cost

        sizes = {}
        total_cst = sum(spread_cost.values())
        if total_cst == 0.0:
            return {trade: 0.0 for trade in entering_trades}
        for trade in entering_trades:
            tick = trade.entry_data.tick
            if spread_cost[tick] == 0.0:
                sizes[trade] = 0.0
            sizes[trade] = downside_cap * (spread_cost[tick] / total_cst)

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
