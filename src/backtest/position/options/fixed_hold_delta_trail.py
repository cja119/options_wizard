from typing import List, Dict, Tuple, Set, Any

from data.date import DateObj
from data.trade import Cashflow, PositionType
from backtest.trade import Trade
from typing import override

from .fixed_hold import FixedHoldNotional


def _to_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


class FixedHoldNotionalDeltaTrail(FixedHoldNotional):
    """
    FixedHoldNotional variant with a simple delta trailing stop per contract.

    - Track the best (most favorable) abs(delta) per trade.
      * Shorts: best is the minimum abs(delta); stop when abs(delta) >= best + gap.
      * Longs: best is the maximum abs(delta); stop when abs(delta) <= best - gap.
    - Gap configured via trailing_gap (default 0.05).
    - When any leg stops, exit all legs of the structure (both longs and shorts).
    """

    def __init__(self, config):
        super().__init__(config)
        self._trailing_gap = _to_float(self._kwargs.get("trailing_gap", 0.05), 0.05)
        self._best_abs_delta: Dict[Trade, float] = {}
        self._last_delta: Dict[Trade, float] = {}

    @override
    def exit_trigger(
        self,
        live_trades: List[Trade],
        current_date: DateObj,
        scheduled_exits: List[Trade] | Set[Trade],
    ) -> Tuple[List[Trade], List[Cashflow]]:
        exit_cashflows: List[Cashflow] = []
        trades_to_close: set[Trade] = set()
        max_days_open = self._kwargs.get("hold_period", 30)

        stop_long = False
        stop_short = False

        for trade in live_trades:
            # Hold-period exit
            if trade.days_open(current_date) >= max_days_open:
                trades_to_close.add(trade)
                continue

            px_date = trade._nearest_date_below(current_date)
            px = trade.entry_data.price_series.get(px_date)
            delta = getattr(px, "delta", None) if px is not None else None
            if delta is None:
                # fallback to nearest above if nothing below
                px_date = trade._nearest_date_above(current_date)
                px = trade.entry_data.price_series.get(px_date)
                delta = getattr(px, "delta", None) if px is not None else None
            if delta is None:
                continue

            abs_delta = abs(delta)
            self._last_delta[trade] = abs_delta

            prev_best = self._best_abs_delta.get(trade, abs_delta)

            if trade.entry_data.position_type == PositionType.SHORT:
                best = min(prev_best, abs_delta)
                self._best_abs_delta[trade] = best
                if abs_delta - best >= self._trailing_gap:
                    stop_short = True
                    trades_to_close.add(trade)
            else:
                best = max(prev_best, abs_delta)
                self._best_abs_delta[trade] = best
                if best - abs_delta >= self._trailing_gap:
                    stop_long = True
                    trades_to_close.add(trade)

        # If any leg stopped out, exit all remaining legs to keep structure aligned
        if stop_long or stop_short:
            for trade in live_trades:
                trades_to_close.add(trade)

        # Include scheduled exits
        trades_to_close.update(scheduled_exits)

        # Close trades and collect cashflows
        for trade in trades_to_close:
            _, cashflow = trade.close(current_date)
            exit_cashflows.append(cashflow)

        # Cleanup best/last tracking for closed trades
        for trade in list(self._best_abs_delta.keys()):
            if trade in trades_to_close:
                self._best_abs_delta.pop(trade, None)
        if hasattr(self, "_last_delta"):
            for trade in list(self._last_delta.keys()):
                if trade in trades_to_close:
                    self._last_delta.pop(trade, None)

        return trades_to_close, exit_cashflows
