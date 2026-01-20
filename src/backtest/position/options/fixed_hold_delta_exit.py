from typing import List, Dict, Tuple, Set, Any

from data.date import DateObj
from data.trade import Cashflow, PositionType
from backtest.trade import Trade
from typing import override

from .fixed_hold import FixedHoldNotional


def _to_float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


class FixedHoldNotionalDeltaExit(FixedHoldNotional):
    """
    FixedHoldNotional variant that exits early on delta thresholds.

    - Shorts: close if abs(delta) < short_abs_delta_lower or > short_abs_delta_upper (defaults 0.05 / None)
    - Longs: close if abs(delta) < long_abs_delta_lower or > long_abs_delta_upper (defaults None / 0.5)
    """

    def __init__(self, config):
        super().__init__(config)
        self._short_abs_delta_lower = _to_float_or_none(
            self._kwargs.get("short_abs_delta_lower", 0.05)
        )
        self._short_abs_delta_upper = _to_float_or_none(
            self._kwargs.get("short_abs_delta_upper", None)
        )
        self._long_abs_delta_lower = _to_float_or_none(
            self._kwargs.get("long_abs_delta_lower", None)
        )
        self._long_abs_delta_upper = _to_float_or_none(
            self._kwargs.get("long_abs_delta_upper", 0.5)
        )

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

        for trade in live_trades:
            # Hold-period exit
            if trade.days_open(current_date) >= max_days_open:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)
                continue

            # Delta-based exits
            px = trade.entry_data.price_series.get(current_date)
            delta = getattr(px, "delta", None) if px is not None else None
            if delta is None:
                continue
            abs_delta = abs(delta)

            if trade.entry_data.position_type == PositionType.SHORT:
                if (
                    self._short_abs_delta_lower is not None
                    and abs_delta < self._short_abs_delta_lower
                ) or (
                    self._short_abs_delta_upper is not None
                    and abs_delta > self._short_abs_delta_upper
                ):
                    _, cashflow = trade.close(current_date)
                    exit_cashflows.append(cashflow)
                    trades_to_close.add(trade)
            else:
                # LONG
                if (
                    self._long_abs_delta_lower is not None
                    and abs_delta < self._long_abs_delta_lower
                ) or (
                    self._long_abs_delta_upper is not None
                    and abs_delta > self._long_abs_delta_upper
                ):
                    _, cashflow = trade.close(current_date)
                    exit_cashflows.append(cashflow)
                    trades_to_close.add(trade)

        for trade in scheduled_exits:
            if trade not in trades_to_close:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)

        return trades_to_close, exit_cashflows
