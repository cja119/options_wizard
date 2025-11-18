"""
Refactored backtesting pipeline with a pluggable ``main`` entry point.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Callable, Mapping

import numpy as np
import pandas as pd
import exchange_calendars as ec


ContractKey = tuple[pd.Timestamp, int, float]
PositionKey = tuple[pd.Timestamp, int, float, pd.Timestamp]


def _naive_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Return a timezone-naive pandas Timestamp."""
    ts = pd.Timestamp(ts)
    try:
        return ts.tz_localize(None)
    except TypeError:
        # Already tz-naive
        return ts


@dataclass
class TradedContract:
    expiry_date_idx: pd.Timestamp
    call_put_idx: int
    strike_idx: float
    base_position: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    scaled_position: float
    last_price: float
    delta: float
    theta: float
    vega: float
    has_price_today: bool = True

    @property
    def key(self) -> ContractKey:
        return (self.expiry_date_idx, self.call_put_idx, self.strike_idx)

    @property
    def position_key(self) -> PositionKey:
        """Unique identifier for a trade, including its entry date."""
        return (
            self.expiry_date_idx,
            self.call_put_idx,
            self.strike_idx,
            _naive_timestamp(self.entry_date),
        )

    @property
    def market_value(self) -> float:
        return self.scaled_position * self.last_price


@dataclass
class Spread:
    short_legs: list[TradedContract]
    long_legs: list[TradedContract]

    @property
    def all_legs(self) -> list[TradedContract]:
        return self.short_legs + self.long_legs

    def exit_date(self) -> pd.Timestamp:
        return self.all_legs[0].exit_date if self.all_legs else pd.NaT

    def has_missing_prices(self) -> bool:
        return any(not leg.has_price_today for leg in self.all_legs)

    @property
    def market_value(self) -> float:
        return sum(leg.market_value for leg in self.all_legs)

    def update_prices(self, price_lookup: dict[ContractKey, dict[str, float]]) -> None:
        for leg in self.all_legs:
            values = price_lookup.get(leg.key)
            if values is None:
                leg.has_price_today = False
                continue
            leg.last_price = float(values["mid_price"])
            leg.delta = float(values["delta"])
            leg.theta = float(values["theta"])
            leg.vega = float(values["vega"])
            leg.has_price_today = True


@dataclass
class PortfolioState:
    cash: float
    active_spreads: list[Spread] = field(default_factory=list)

    @property
    def exposure_value(self) -> float:
        return sum(spread.market_value for spread in self.active_spreads)

    @property
    def equity(self) -> float:
        return self.cash + self.exposure_value


@dataclass
class SubStrategy:
    date: pd.Timestamp
    active_contracts: list[TradedContract]
    current_cash: float

    @property
    def positions_value(self) -> float:
        return sum(c.market_value for c in self.active_contracts)

    @property
    def total(self) -> float:
        return self.current_cash + self.positions_value

    @property
    def delta(self) -> float:
        denom = self.total
        if denom == 0:
            return 0.0
        return sum(c.scaled_position * c.delta * c.last_price for c in self.active_contracts) / denom

    @property
    def theta(self) -> float:
        denom = self.total
        if denom == 0:
            return 0.0
        return sum(c.scaled_position * c.theta * c.last_price for c in self.active_contracts) / denom

    @property
    def vega(self) -> float:
        denom = self.total
        if denom == 0:
            return 0.0
        return sum(c.scaled_position * c.vega * c.last_price for c in self.active_contracts) / denom

    @property
    def shorts_mag(self) -> float:
        return sum(abs(c.scaled_position) * c.last_price for c in self.active_contracts if c.scaled_position < 0)

    @property
    def longs_mag(self) -> float:
        return sum(abs(c.scaled_position) * c.last_price for c in self.active_contracts if c.scaled_position > 0)


class SizingFn(Protocol):
    def __call__(self, portfolio: PortfolioState, entry_group: pd.DataFrame) -> float:
        """Return the contract multiplier to apply for ``entry_group``."""


class ExitFn(Protocol):
    def __call__(self, spread: Spread, current_date: pd.Timestamp) -> bool:
        """Return True if the spread should be closed on ``current_date``."""


class CapitalAllocationFn(Protocol):
    def __call__(self, date: pd.Timestamp, short_notional: dict[str, float]) -> dict[str, float]:
        """Return weights to apply across ticks for the specified date."""


@dataclass
class MultiEquitySnapshot:
    date: pd.Timestamp
    combined: SubStrategy
    per_tick: dict[str, SubStrategy]


def make_equity_fraction_sizer(
    capital_fraction: float,
    entry_cost_size: str = "short",
) -> SizingFn:
    """Match the sizing behaviour from ``multi_day_entry``."""

    def _cost(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        return float((df["mid_price"] * df["position"].abs()).sum())

    def _sizer(portfolio: PortfolioState, entry_group: pd.DataFrame) -> float:
        shorts_df = entry_group[entry_group["position"] < 0]
        longs_df = entry_group[entry_group["position"] > 0]

        if entry_cost_size == "short":
            denom = _cost(shorts_df)
        elif entry_cost_size == "long":
            denom = _cost(longs_df)
        elif entry_cost_size == "diff":
            denom = _cost(longs_df) - _cost(shorts_df)
        else:
            raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")

        if denom == 0 or np.isnan(denom):
            return 0.0

        equity = portfolio.equity
        if equity <= 0:
            return 0.0

        return capital_fraction * equity / denom

    return _sizer


def default_exit_condition(spread: Spread, current_date: pd.Timestamp) -> bool:
    """Exit when the hold period passes or a price is missing."""
    naive_date = (
        current_date.tz_localize(None) if getattr(current_date, "tzinfo", None) else current_date
    )
    exit_date = spread.exit_date()
    if pd.isna(exit_date):
        return True
    exit_date = exit_date.tz_localize(None) if getattr(exit_date, "tzinfo", None) else exit_date
    return naive_date >= exit_date or spread.has_missing_prices()


def _default_capital_allocation(date: pd.Timestamp, short_notional: dict[str, float]) -> dict[str, float]:
    """Equal allocation across all supplied ticks, regardless of exposure."""
    if not short_notional:
        return {}
    keys = list(short_notional.keys())
    weight = 1.0 / len(keys)
    return {tick: weight for tick in keys}


def _build_spread(
    entry_group: pd.DataFrame,
    size_multiplier: float,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> Spread:
    short_legs: list[TradedContract] = []
    long_legs: list[TradedContract] = []

    for _, row in entry_group.iterrows():
        leg = TradedContract(
            expiry_date_idx=row["expiry_date_idx"],
            call_put_idx=row["call_put_idx"],
            strike_idx=row["strike_idx"],
            base_position=float(row["position"]),
            entry_date=entry_date,
            exit_date=exit_date,
            scaled_position=size_multiplier * row["position"],
            last_price=float(row["mid_price"]),
            delta=float(row["delta"]),
            theta=float(row["theta"]),
            vega=float(row["vega"]),
        )
        if row["position"] < 0:
            short_legs.append(leg)
        else:
            long_legs.append(leg)

    return Spread(short_legs=short_legs, long_legs=long_legs)


def _prep_trade_frames(
    trade_entries: pd.DataFrame,
    option_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise dataframe indices and align contracts."""
    if trade_entries.empty or option_data.empty:
        return trade_entries.copy(), option_data.copy()

    keys = ["expiry_date_idx", "call_put_idx", "strike_idx"]
    trade_entries = trade_entries.reset_index()
    valid_contracts = trade_entries[keys].drop_duplicates()
    filtered_data = option_data.merge(valid_contracts, on=keys, how="inner")

    def _normalise_dates(df: pd.DataFrame, column: str) -> pd.Series:
        norm = pd.to_datetime(df[column], errors="coerce")
        try:
            norm = norm.dt.tz_localize(None)
        except TypeError:
            pass
        return norm

    trade_entries = trade_entries.copy()
    filtered_data = filtered_data.copy()
    trade_entries["norm_trade_date"] = _normalise_dates(trade_entries, "trade_date")
    filtered_data["norm_trade_date"] = _normalise_dates(filtered_data, "trade_date")

    trade_entries = trade_entries.sort_values("trade_date_idx")
    return trade_entries, filtered_data


def _price_lookup_for_date(
    filtered_data: pd.DataFrame,
    current_date_naive: pd.Timestamp,
    keys: list[str],
) -> dict[ContractKey, dict[str, float]]:
    todays_prices = filtered_data.loc[
        filtered_data["norm_trade_date"] == current_date_naive,
        keys + ["mid_price", "delta", "theta", "vega"],
    ]
    if todays_prices.empty:
        return {}

    price_frame = todays_prices.set_index(keys)
    lookup: dict[ContractKey, dict[str, float]] = {}
    for idx, row in price_frame.iterrows():
        lookup[idx] = {
            "mid_price": float(row["mid_price"]),
            "delta": float(row["delta"]),
            "theta": float(row["theta"]),
            "vega": float(row["vega"]),
        }
    return lookup


def main(
    trade_entries: pd.DataFrame,
    option_data: pd.DataFrame,
    *,
    hold_period: int = 30,
    initial_capital: float = 100_000.0,
    capital_per_trade: float = 0.05,
    entry_cost_size: str = "short",
    sizing_fn: SizingFn | None = None,
    exit_fn: ExitFn | None = None,
    calendar: str = "XNAS",
) -> pd.DataFrame:
    """Run the refactored daily backtest loop."""
    if trade_entries.empty:
        return pd.DataFrame(columns=["equity", "equity_obj", "daily_log_return"])

    hold_td = pd.Timedelta(days=hold_period)
    keys = ["expiry_date_idx", "call_put_idx", "strike_idx"]
    trades_df, filtered_data = _prep_trade_frames(trade_entries, option_data)

    if trades_df.empty or filtered_data.empty:
        return pd.DataFrame(columns=["equity", "equity_obj", "daily_log_return"])

    if sizing_fn is None:
        sizing_fn = make_equity_fraction_sizer(capital_per_trade, entry_cost_size)
    if exit_fn is None:
        exit_fn = default_exit_condition

    cal = ec.get_calendar(calendar)
    sessions = cal.sessions_in_range(
        trades_df["norm_trade_date"].min(),
        filtered_data["norm_trade_date"].max(),
    )

    portfolio = PortfolioState(cash=initial_capital)
    equity_curve: list[dict[str, float | pd.Timestamp | SubStrategy]] = []
    prev_equity = initial_capital

    for current_date in sessions:
        current_date_naive = (
            current_date.tz_localize(None) if getattr(current_date, "tzinfo", None) else current_date
        )

        todays_entries = trades_df[trades_df["norm_trade_date"] == current_date_naive]
        if not todays_entries.empty:
            for _, entry_group in todays_entries.groupby("trade_date_idx"):
                size_multiplier = sizing_fn(portfolio, entry_group)
                if size_multiplier <= 0:
                    continue

                entry_date = entry_group["trade_date"].iloc[0]
                exit_date = entry_date + hold_td
                spread = _build_spread(entry_group, size_multiplier, entry_date, exit_date)
                net_cost = spread.market_value
                if np.isnan(net_cost) or net_cost == 0.0:
                    continue

                portfolio.cash -= net_cost
                portfolio.active_spreads.append(spread)

        price_lookup = _price_lookup_for_date(filtered_data, current_date_naive, keys)
        for spread in portfolio.active_spreads:
            spread.update_prices(price_lookup)

        snapshot = SubStrategy(
            date=current_date_naive,
            active_contracts=[copy.copy(c) for sp in portfolio.active_spreads for c in sp.all_legs],
            current_cash=portfolio.cash,
        )
        current_equity = snapshot.total
        equity_curve.append(
            {
                "date": current_date_naive,
                "equity": current_equity,
                "equity_obj": snapshot,
                "daily_log_return": np.log(current_equity / prev_equity) if prev_equity > 0 else np.nan,
            }
        )
        prev_equity = current_equity

        remaining_spreads: list[Spread] = []
        for spread in portfolio.active_spreads:
            if exit_fn(spread, current_date_naive):
                portfolio.cash += spread.market_value
            else:
                remaining_spreads.append(spread)
        portfolio.active_spreads = remaining_spreads

    return pd.DataFrame(equity_curve).set_index("date").sort_index()


def combine_multi_equity(
    per_stock_equity: Mapping[str, pd.DataFrame],
    *,
    initial_capital: float = 100_000.0,
    capital_per_trade: float = 0.05,
    capital_allocation: CapitalAllocationFn | None = None,
    entry_cost_size: str = "short",
    top_per_date: Mapping[pd.Timestamp, list[str]] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Combine per-stock equity curves into a single multi-equity portfolio.

    Parameters
    ----------
    per_stock_equity:
        Mapping from ticker -> DataFrame produced by ``main``.
    capital_per_trade:
        Fraction of equity deployed per ticker each day (before spreading across ticks).
    capital_allocation:
        Callable that returns weights per ticker for a given date. Defaults to equal exposure.
    entry_cost_size:
        Which leg notionals (``short``/``long``/``diff``) to use when sizing spreads.
    top_per_date:
        Optional mapping/Series of index constituents by date. When provided, only those ticks
        present in the latest available constituent list (as of each date) are traded.
    """
    if not per_stock_equity:
        return pd.DataFrame(columns=["equity", "equity_obj", "daily_log_return"])

    allocation_fn = capital_allocation or _default_capital_allocation

    if top_per_date is not None:
        if isinstance(top_per_date, pd.Series):
            top_series = top_per_date.copy()
        else:
            top_series = pd.Series(top_per_date)
        top_series.index = pd.to_datetime(top_series.index)
        top_series = top_series.sort_index()
    else:
        top_series = None

    def _constituents_for(date: pd.Timestamp) -> set[str] | None:
        if top_series is None or top_series.empty:
            return None
        eligible = top_series.loc[top_series.index <= date]
        if eligible.empty:
            return None
        latest = eligible.iloc[-1]
        if isinstance(latest, (list, tuple, set)):
            return set(latest)
        return {latest}
    all_dates: list[pd.Timestamp] = sorted(
        {date for df in per_stock_equity.values() if not df.empty for date in df.index}
    )

    if not all_dates:
        return pd.DataFrame(columns=["equity", "equity_obj", "daily_log_return"])

    def _notional_for_contracts(contracts: list[TradedContract]) -> float:
        long_cost = sum(abs(c.base_position) * c.last_price for c in contracts if c.base_position > 0)
        short_cost = sum(abs(c.base_position) * c.last_price for c in contracts if c.base_position < 0)
        if entry_cost_size == "long":
            return long_cost
        if entry_cost_size == "diff":
            return long_cost - short_cost
        if entry_cost_size == "short":
            return short_cost
        raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")

    def _sizing_denominator(contracts: list[TradedContract]) -> float:
        long_cost = sum(abs(c.base_position) * c.last_price for c in contracts if c.base_position > 0)
        short_cost = sum(abs(c.base_position) * c.last_price for c in contracts if c.base_position < 0)
        if entry_cost_size == "long":
            return long_cost
        if entry_cost_size == "diff":
            return long_cost - short_cost
        if entry_cost_size == "short":
            return short_cost
        raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")

    holdings: dict[str, dict[PositionKey, TradedContract]] = {tick: {} for tick in per_stock_equity.keys()}
    combined_rows: list[dict[str, object]] = []
    portfolio_cash = initial_capital
    prev_equity = initial_capital

    for date in all_dates:
        allowed_ticks = _constituents_for(date)
        if allowed_ticks is None:
            active_ticks = list(per_stock_equity.keys())
        else:
            active_ticks = [tick for tick in per_stock_equity.keys() if tick in allowed_ticks]
        if not active_ticks:
            active_ticks = list(per_stock_equity.keys())
        new_entries: dict[str, list[tuple[PositionKey, TradedContract]]] = defaultdict(list)

        for tick, df in per_stock_equity.items():
            state = holdings.setdefault(tick, {})
            if df.empty or date not in df.index:
                for contract in state.values():
                    contract.has_price_today = False
                continue

            row = df.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            snapshot = row.get("equity_obj")
            if not isinstance(snapshot, SubStrategy):
                for contract in state.values():
                    contract.has_price_today = False
                continue

            snapshot_contracts = {
                contract.position_key: contract for contract in snapshot.active_contracts
            }

            for key in list(state.keys()):
                if key not in snapshot_contracts:
                    portfolio_cash += state[key].market_value
                    del state[key]

            for key, source_contract in snapshot_contracts.items():
                if key in state:
                    target = state[key]
                    target.last_price = float(source_contract.last_price)
                    target.delta = float(source_contract.delta)
                    target.theta = float(source_contract.theta)
                    target.vega = float(source_contract.vega)
                    target.has_price_today = bool(source_contract.has_price_today)

            if allowed_ticks is not None and tick not in allowed_ticks:
                continue

            for key, contract in snapshot_contracts.items():
                if key not in state:
                    new_entries[tick].append((key, contract))

        positions_value_before = sum(
            contract.market_value for tick_state in holdings.values() for contract in tick_state.values()
        )
        equity_before = portfolio_cash + positions_value_before

        entering_ticks = [tick for tick, contracts in new_entries.items() if contracts]
        if entering_ticks and active_ticks and equity_before > 0:
            allocation_inputs = {tick: 0.0 for tick in active_ticks}
            for tick in entering_ticks:
                allocation_inputs[tick] = _notional_for_contracts([contract for _, contract in new_entries[tick]])
            weights = allocation_fn(date, allocation_inputs)
            weights = {tick: weights.get(tick, 0.0) for tick in active_ticks}
            positive_sum = sum(weight for weight in weights.values() if weight > 0)
            if positive_sum <= 0:
                equal_weight = 1.0 / len(active_ticks)
                weights = {tick: equal_weight for tick in active_ticks}
            else:
                weights = {
                    tick: (weight / positive_sum) if weight > 0 else 0.0 for tick, weight in weights.items()
                }

            for tick in entering_ticks:
                weight = weights.get(tick, 0.0)
                if weight <= 0:
                    continue
                capital_fraction = capital_per_trade * weight
                if capital_fraction <= 0:
                    continue
                denom = _sizing_denominator([contract for _, contract in new_entries[tick]])
                if denom <= 0 or np.isnan(denom):
                    continue
                size_multiplier = (capital_fraction * equity_before) / denom
                if size_multiplier <= 0 or np.isnan(size_multiplier):
                    continue
                state = holdings.setdefault(tick, {})
                for key, contract in new_entries[tick]:
                    scaled_contract = copy.copy(contract)
                    base = contract.base_position if contract.base_position != 0 else contract.scaled_position
                    scaled_contract.scaled_position = base * size_multiplier
                    state[key] = scaled_contract
                    portfolio_cash -= scaled_contract.market_value

        combined_contracts: list[TradedContract] = []
        per_tick_scaled: dict[str, SubStrategy] = {}
        for tick, tick_state in holdings.items():
            if not tick_state:
                continue
            tick_contracts = [copy.copy(contract) for contract in tick_state.values()]
            per_tick_scaled[tick] = SubStrategy(
                date=date,
                active_contracts=tick_contracts,
                current_cash=0.0,
            )
            combined_contracts.extend(tick_contracts)

        positions_value = sum(contract.market_value for contract in combined_contracts)
        current_equity = portfolio_cash + positions_value
        combined_snapshot = SubStrategy(
            date=date,
            active_contracts=combined_contracts,
            current_cash=portfolio_cash,
        )
        multi_snapshot = MultiEquitySnapshot(
            date=date,
            combined=combined_snapshot,
            per_tick=per_tick_scaled,
        )
        daily_log_return = np.log(current_equity / prev_equity) if prev_equity > 0 and current_equity > 0 else np.nan
        combined_rows.append(
            {
                "date": date,
                "equity": current_equity,
                "equity_obj": combined_snapshot,
                "multi_snapshot": multi_snapshot,
                "daily_log_return": daily_log_return,
            }
        )
        prev_equity = current_equity

    return pd.DataFrame(combined_rows).set_index("date").sort_index()
