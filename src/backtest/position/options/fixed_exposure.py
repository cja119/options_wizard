from ..base import PositionBase
from typing import List, Dict, Tuple, Optional

from data.date import DateObj
from data.trade import Cashflow
from backtest.trade import Trade
from data.trade import PositionType
from typing import override

TRADE_EXPOSURE_LIM = 0.025
DAILY_EXPOSURE_LIM = 0.1


class ShortExposureLimTrade(PositionBase):
    _carry = 0.0
    _notional_exposure = None
    _live_spreads = None

    def __init__(self, config) -> None:
        super().__init__(config)
        self._live_spreads = set()

    def _spread_key(self, trade: Trade) -> Optional[tuple]:
        features = trade.entry_data.features
        if features is None:
            return None
        other_contracts = getattr(features, "other_contracts", None)
        if not other_contracts:
            return None
        return (trade.entry_data.tick, tuple(other_contracts))

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:

        if not entering_trades:
            return {}

        grouped = {}
        allowed_trades = []
        for trade in entering_trades:
            key = self._spread_key(trade)
            if key is None:
                allowed_trades.append(trade)
                continue
            grouped.setdefault(key, []).append(trade)

        for key, trades in grouped.items():
            if key in self._live_spreads:
                continue
            allowed_trades.extend(trades)
            self._live_spreads.add(key)

        if not allowed_trades:
            return {trade: 0.0 for trade in entering_trades}

        te = self._snapshot.total_equity + self._snapshot.total_cash
        daily_lim = te * DAILY_EXPOSURE_LIM
        trade_lim = te * TRADE_EXPOSURE_LIM

        short_size = {}
        for trade in allowed_trades:
            if short_size.get(trade.entry_data.tick) is None:
                short_size[trade.entry_data.tick] = 0.0

            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            if entry_px is None or trade.entry_data.position_type != PositionType.SHORT:
                continue

            if (entry_px.ask - entry_px.bid) / (entry_px.bid + entry_px.ask) > 0.15:
                continue

            short_size[trade.entry_data.tick] += abs(
                entry_px.ask * abs(trade.entry_data.position_size)
            )

        for tick, val in short_size.items():
            if val == 0.0:
                continue
            short_size[tick] = trade_lim / val

        daily_sum = sum(short_size.values()) + 1e-10
        if daily_sum > daily_lim:
            scale = daily_lim / daily_sum
            for tick in short_size.keys():
                short_size[tick] *= scale

        sizes = {trade: 0.0 for trade in entering_trades}
        for trade in allowed_trades:
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

        if self._live_spreads:
            for trade in trades_to_close:
                key = self._spread_key(trade)
                if key is not None:
                    self._live_spreads.discard(key)
        return trades_to_close, exit_cashflows
