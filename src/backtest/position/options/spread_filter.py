from ..base import PositionBase
from typing import List, Dict, Tuple

from data.date import DateObj
from data.trade import Cashflow
from backtest.trade import Trade
from data.trade import PositionType
from typing import override

TRADE_EXPOSURE_LIM = 0.025
DAILY_EXPOSURE_LIM = 0.1


class SpreadFilteredTrade(PositionBase):
    _carry = 0.0
    _notional_exposure = None

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:

        te = self._snapshot.total_equity + self._snapshot.total_cash
        daily_lim = te * DAILY_EXPOSURE_LIM
        trade_lim = te * TRADE_EXPOSURE_LIM

        short_size = {}
        spread_price = {}
        rec_high_low = {}
        for trade in entering_trades:
            if short_size.get(trade.entry_data.tick) is None:
                short_size[trade.entry_data.tick] = 0.0

            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)

            if spread_price.get(trade.entry_data.tick) is None:
                spread_price[trade.entry_data.tick] = 0.0

            _dir = -1 if trade.entry_data.position_type == PositionType.SHORT else 1
            spread_price[trade.entry_data.tick] += (
                (entry_px.bid + entry_px.ask / 2)
                * _dir
                * abs(trade.entry_data.position_size)
            )

            if (
                entry_px is None
                or entry_px.underlying is None
                or trade.entry_data.position_type != PositionType.SHORT
            ):
                continue

            short_size[trade.entry_data.tick] += abs(
                entry_px.bid * abs(trade.entry_data.position_size)
            )
            rec_high_low[trade.entry_data.tick] = entry_px.other.get(
                "recent_high_low", None
            )

        for tick, val in short_size.items():
            if val == 0.0:
                continue
            short_size[tick] = trade_lim / val

            if spread_price[tick] > 0.5 * rec_high_low[tick]:
                short_size[tick] = 0.0

        daily_sum = sum(short_size.values())
        if daily_sum > daily_lim:
            scale = daily_lim / daily_sum
            for tick in short_size.keys():
                short_size[tick] *= scale

        sizes = {}
        for trade in entering_trades:
            tick = trade.entry_data.tick
            sizes[trade] = short_size.get(tick, 0.0)

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
