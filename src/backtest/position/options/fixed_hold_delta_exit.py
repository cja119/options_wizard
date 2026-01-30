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


class StockFixedHoldNotionalDeltaExit(FixedHoldNotionalDeltaExit):
    """
    Equal-notional sizing across multiple tickers with delta-based exits.

    Sizes entering trades so each ticker receives the same notional budget
    based on gross (absolute) delta exposure of all entering legs.
    """

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:
        protected_notional = self._kwargs.get("protected_notional", 1_000_000)
        hold_period = self._kwargs.get("hold_period", 30)
        notional = protected_notional / hold_period

        if not entering_trades:
            if getattr(self, "_carry_window", 0) > 0:
                self._carry.append(notional)
            return {}

        if getattr(self, "_carry_window", 0) > 0:
            notional += sum(self._carry)
            self._carry.clear()

        tick_exposure: Dict[str, float] = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            if (
                entry_px is None
                or entry_px.delta is None
                or entry_px.underlying is None
            ):
                continue
            tick = entry_px.tick
            exp = abs(
                entry_px.delta
                * trade.entry_data.position_size
                * entry_px.underlying.ask
            )
            tick_exposure[tick] = tick_exposure.get(tick, 0.0) + exp

        if not tick_exposure:
            if getattr(self, "_carry_window", 0) > 0:
                self._carry.append(notional)
            return {trade: 0.0 for trade in entering_trades}

        per_tick_budget = notional / len(tick_exposure)

        sizes: Dict[Trade, float] = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            tick = entry_px.tick if entry_px is not None else None
            tick_exp = tick_exposure.get(tick or "", 0.0)
            if tick_exp == 0.0:
                sizes[trade] = 0.0
                continue
            sizes[trade] = per_tick_budget / tick_exp

        return sizes


class FixedHoldNotionalDeltaExitShortLimitRoll(FixedHoldNotionalDeltaExit):
    """
    FixedHoldNotionalDeltaExit variant that closes the short + its paired leg(s)
    (same ticker, same entry date) if the short hits a specified absolute-delta
    limit, and rolls that short's entry notional into the next entry day.
    """

    def __init__(self, config):
        super().__init__(config)
        limit_val = self._kwargs.get("short_abs_delta_limit", None)
        if limit_val is None:
            limit_val = self._short_abs_delta_upper
        self._short_abs_delta_limit = _to_float_or_none(limit_val)
        self._roll_notional = 0.0

    def _entry_notional(self, trade: Trade) -> float:
        entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
        if (
            entry_px is None
            or entry_px.delta is None
            or entry_px.underlying is None
        ):
            return 0.0
        return abs(
            entry_px.delta
            * trade.entry_data.position_size
            * entry_px.underlying.ask
        )

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:
        protected_notional = self._kwargs.get("protected_notional", 1_000_000)
        hold_period = self._kwargs.get("hold_period", 30)
        notional = protected_notional / hold_period

        if not entering_trades:
            if self._carry_window > 0:
                self._carry.append(notional)
            return {}

        if self._carry_window > 0:
            notional += sum(self._carry)
            self._carry.clear()

        if self._roll_notional > 0.0:
            notional += self._roll_notional
            self._roll_notional = 0.0

        short_exp_by_tick: Dict[str, float] = {}
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
            if self._carry_window > 0:
                self._carry.append(notional)
            return {trade: 0.0 for trade in entering_trades}

        per_tick_budget = notional / len(short_exp_by_tick)

        sizes: Dict[Trade, float] = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            tick = entry_px.tick if entry_px is not None else None
            tick_exp = short_exp_by_tick.get(tick or "", 0.0)
            if tick_exp == 0.0:
                sizes[trade] = 0.0
                continue
            sizes[trade] = per_tick_budget / tick_exp

        return sizes

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

        # Hold-period exits (natural roll)
        for trade in live_trades:
            if trade.days_open(current_date) >= max_days_open:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)

        remaining = [t for t in live_trades if t not in trades_to_close]

        # Short upper-limit trigger: close the pair (same tick + entry date)
        limit_pairs: set[tuple[str, str]] = set()
        if self._short_abs_delta_limit is not None:
            for trade in remaining:
                if trade.entry_data.position_type != PositionType.SHORT:
                    continue
                px = trade.entry_data.price_series.get(current_date)
                delta = getattr(px, "delta", None) if px is not None else None
                if delta is None:
                    continue
                if abs(delta) >= self._short_abs_delta_limit:
                    key = (
                        trade.entry_data.tick,
                        trade.entry_data.entry_date.to_iso(),
                    )
                    limit_pairs.add(key)

        if limit_pairs:
            roll_notional = 0.0
            for trade in remaining:
                key = (
                    trade.entry_data.tick,
                    trade.entry_data.entry_date.to_iso(),
                )
                if key in limit_pairs:
                    _, cashflow = trade.close(current_date)
                    exit_cashflows.append(cashflow)
                    trades_to_close.add(trade)
                    if trade.entry_data.position_type == PositionType.SHORT:
                        roll_notional += self._entry_notional(trade)
            if roll_notional > 0.0:
                self._roll_notional += roll_notional

        remaining = [t for t in remaining if t not in trades_to_close]

        # Standard delta-based exits for remaining trades
        for trade in remaining:
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


class StockFixedHoldNotionalDeltaExitShortLimitRoll(
    FixedHoldNotionalDeltaExitShortLimitRoll
):
    """
    StockFixedHoldNotionalDeltaExit variant with short-limit roll logic.

    Uses equal-notional sizing across tickers and rolls short entry notional
    into the next entry day when the short hits the limit.
    """

    @override
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:
        protected_notional = self._kwargs.get("protected_notional", 1_000_000)
        hold_period = self._kwargs.get("hold_period", 30)
        notional = protected_notional / hold_period

        if not entering_trades:
            return {}

        if self._roll_notional > 0.0:
            notional += self._roll_notional
            self._roll_notional = 0.0

        tick_exposure: Dict[str, float] = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            if (
                entry_px is None
                or entry_px.delta is None
                or entry_px.underlying is None
            ):
                continue
            tick = entry_px.tick
            exp = abs(
                entry_px.delta
                * trade.entry_data.position_size
                * entry_px.underlying.ask
            )
            tick_exposure[tick] = tick_exposure.get(tick, 0.0) + exp

        if not tick_exposure:
            return {trade: 0.0 for trade in entering_trades}

        per_tick_budget = notional / len(tick_exposure)

        sizes: Dict[Trade, float] = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            tick = entry_px.tick if entry_px is not None else None
            tick_exp = tick_exposure.get(tick or "", 0.0)
            if tick_exp == 0.0:
                sizes[trade] = 0.0
                continue
            sizes[trade] = per_tick_budget / tick_exp

        return sizes
