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
PositionKey = tuple[pd.Timestamp, int, float, pd.Timestamp, int | str | None]


def _naive_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Return a timezone-naive pandas Timestamp."""
    ts = pd.Timestamp(ts)
    try:
        return ts.tz_localize(None)
    except TypeError:
        # Already tz-naive
        return ts


def _normalise_entry_id(entry_id) -> int | str | None:
    """Convert raw trade identifiers to stable hashable values."""
    if entry_id is None:
        return None
    try:
        if pd.isna(entry_id):
            return None
    except TypeError:
        # Non-numeric types like str will raise here; keep as-is.
        pass
    if isinstance(entry_id, float):
        if np.isnan(entry_id):
            return None
        if entry_id.is_integer():
            return int(entry_id)
    if isinstance(entry_id, (np.integer, int)):
        return int(entry_id)
    return entry_id


@dataclass
class TradedContract:
    expiry_date_idx: pd.Timestamp
    call_put_idx: int
    strike_idx: float
    base_position: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    scaled_position: float
    recent_price: float
    last_price: float
    delta: float
    theta: float
    vega: float
    underlying_close: float | None = None
    has_price_today: bool = True
    trade_date_idx: int | str | None = None

    @property
    def current_pnl(self) -> float:
        return self.scaled_position * (self.recent_price - self.last_price) if self.has_price_today else 0.0
    
    @property
    def key(self) -> ContractKey:
        return (self.expiry_date_idx, self.call_put_idx, self.strike_idx)

    @property
    def position_key(self) -> PositionKey:
        """Unique identifier for a trade, including its entry date."""
        entry_id = _normalise_entry_id(self.trade_date_idx)
        return (
            self.expiry_date_idx,
            self.call_put_idx,
            self.strike_idx,
            _naive_timestamp(self.entry_date),
            entry_id,
        )

    @property
    def market_value(self) -> float:
        return self.scaled_position * self.recent_price


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
            leg.last_price = copy.deepcopy(leg.recent_price)
            leg.recent_price = float(values["mid_price"])
            leg.delta = float(values["delta"])
            leg.theta = float(values["theta"])
            leg.vega = float(values["vega"])
            underlying_px = values.get("underlying_close", leg.underlying_close)
            if underlying_px is None or pd.isna(underlying_px):
                leg.underlying_close = np.nan
            else:
                try:
                    leg.underlying_close = float(underlying_px)
                except (TypeError, ValueError):
                    leg.underlying_close = np.nan
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
        return sum(c.scaled_position * c.delta * c.recent_price for c in self.active_contracts) / denom

    @property
    def theta(self) -> float:
        denom = self.total
        if denom == 0:
            return 0.0
        return sum(c.scaled_position * c.theta * c.recent_price for c in self.active_contracts) / denom

    @property
    def vega(self) -> float:
        denom = self.total
        if denom == 0:
            return 0.0
        return sum(c.scaled_position * c.vega * c.recent_price for c in self.active_contracts) / denom

    @property
    def shorts_mag(self) -> float:
        return sum(abs(c.scaled_position) * c.recent_price for c in self.active_contracts if c.scaled_position < 0)

    @property
    def longs_mag(self) -> float:
        return sum(abs(c.scaled_position) * c.recent_price for c in self.active_contracts if c.scaled_position > 0)


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


def _underlying_allocation_value(
    contracts: list[TradedContract],
    entry_cost_size: str,
    *,
    use_base: bool = False,
) -> float:
    """Compute underlying notional for a set of contracts using the requested basis."""
    long_value = 0.0
    short_value = 0.0

    for contract in contracts:
        reference = contract.base_position if use_base and contract.base_position != 0 else contract.scaled_position
        try:
            position_value = float(reference)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(position_value) or position_value == 0:
            continue

        price_value = contract.underlying_close
        try:
            underlying_px = float(price_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(underlying_px):
            continue

        exposure = abs(position_value) * underlying_px
        if position_value > 0:
            long_value += exposure
        else:
            short_value += exposure

    if entry_cost_size == "long":
        return long_value
    if entry_cost_size == "diff":
        return long_value - short_value
    if entry_cost_size == "short":
        return short_value
    raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")


def _entry_group_underlying_value(entry_group: pd.DataFrame, entry_cost_size: str) -> float:
    """Underlying notional for an entry group using base positions."""
    if entry_group.empty:
        return 0.0
    if "underlying_close" not in entry_group.columns:
        return np.nan

    long_value = 0.0
    short_value = 0.0

    for _, row in entry_group.iterrows():
        try:
            position = float(row.get("position", np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(position) or position == 0:
            continue

        price_value = row.get("underlying_close", np.nan)
        try:
            underlying_px = float(price_value)
        except (TypeError, ValueError):
            underlying_px = np.nan
        if not np.isfinite(underlying_px):
            continue

        exposure = abs(position) * underlying_px
        if position > 0:
            long_value += exposure
        else:
            short_value += exposure

    if entry_cost_size == "long":
        return long_value
    if entry_cost_size == "diff":
        return long_value - short_value
    if entry_cost_size == "short":
        return short_value
    raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")


def _underlying_equity_value(contracts: list[TradedContract]) -> float:
    """Signed underlying MTM based on current scaled positions."""
    total = 0.0
    for contract in contracts:
        try:
            position_value = float(contract.scaled_position)
            underlying_px = float(contract.underlying_close)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(position_value) and np.isfinite(underlying_px)):
            continue
        total += position_value * underlying_px
    return total


def make_fixed_notional_sizer(
    target_notional: float,
    entry_cost_size: str = "short",
    entry_notional: bool = False,
) -> SizingFn:
    """Size trades to target a fixed underlying notional each day."""

    def _current_notional(portfolio: PortfolioState) -> float:
        contracts = [contract for spread in portfolio.active_spreads for contract in spread.all_legs]
        return _underlying_allocation_value(contracts, entry_cost_size)

    def _sizer(portfolio: PortfolioState, entry_group: pd.DataFrame) -> float:
        try:
            target = float(target_notional)
        except (TypeError, ValueError):
            return 0.0
        if target <= 0 or np.isnan(target):
            return 0.0

        denom = _entry_group_underlying_value(entry_group, entry_cost_size)
        if denom <= 0 or np.isnan(denom):
            return 0.0

        if entry_notional:
            notional_room = target
        else:
            current_notional = _current_notional(portfolio)
            if not np.isfinite(current_notional):
                return 0.0
            notional_room = max(target - current_notional, 0.0)

        if notional_room <= 0:
            return 0.0

        return notional_room / denom

    return _sizer


def make_variable_notional_sizer(
    target_supplier: Callable[[], float | None],
    entry_cost_size: str = "short",
    entry_notional: bool = False,
) -> SizingFn:
    """Size trades against a time-varying target notional supplied externally."""

    def _current_notional(portfolio: PortfolioState) -> float:
        contracts = [contract for spread in portfolio.active_spreads for contract in spread.all_legs]
        return _underlying_allocation_value(contracts, entry_cost_size)

    def _sizer(portfolio: PortfolioState, entry_group: pd.DataFrame) -> float:
        target = target_supplier()
        if target is None or np.isnan(target):
            return 0.0
        if target <= 0:
            return 0.0

        denom = _entry_group_underlying_value(entry_group, entry_cost_size)
        if denom <= 0 or np.isnan(denom):
            return 0.0

        if entry_notional:
            notional_room = target
        else:
            current_notional = _current_notional(portfolio)
            if not np.isfinite(current_notional):
                return 0.0
            notional_room = max(target - current_notional, 0.0)

        if notional_room <= 0:
            return 0.0

        return notional_room / denom

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


def _load_entry_notional_series(
    ticker: str,
    dates: list[pd.Timestamp],
    initial_capital: float,
) -> pd.Series | None:
    """Fetch and scale a price series to start at ``initial_capital``."""
    if not ticker or not dates:
        return None
    try:
        import yfinance as yf
    except Exception:
        return None

    start = min(dates)
    end = max(dates) + pd.Timedelta(days=1)
    try:
        hist = yf.download(ticker, start=start, end=end, progress=False)
    except Exception:
        return None
    if hist is None or hist.empty:
        return None

    price_col = None
    for candidate in ["Adj Close", "Close"]:
        if candidate in hist.columns:
            price_col = candidate
            break
    if price_col is None:
        return None

    series = hist[price_col].copy().dropna()
    if series.empty:
        return None
    series.index = pd.to_datetime(series.index)
    try:
        series.index = series.index.tz_localize(None)
    except TypeError:
        pass
    series = series.sort_index()

    try:
        first_val = float(series.iloc[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(first_val) or first_val <= 0:
        return None

    scale = initial_capital / first_val
    scaled = series * scale

    target_index = pd.to_datetime(dates)
    try:
        target_index = target_index.tz_localize(None)
    except TypeError:
        pass
    return scaled.reindex(target_index).ffill()


def _build_underlying_entry_series(
    option_data: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    initial_capital: float,
) -> pd.Series | None:
    """Create a scaled underlying series from the option data itself."""
    if option_data is None or option_data.empty:
        return None
    if "underlying_close" not in option_data.columns:
        return None

    df = option_data.copy()
    df["underlying_close"] = pd.to_numeric(df["underlying_close"], errors="coerce")
    df = df.dropna(subset=["underlying_close"])
    if df.empty:
        return None

    base_series = df.groupby(level=0)["underlying_close"].first().reindex(sessions)
    base_series = base_series.ffill()
    finite_base = base_series.dropna()
    if finite_base.empty:
        return None
    first_val = finite_base.iloc[0]
    if not np.isfinite(first_val) or first_val <= 0:
        return None

    scale = initial_capital / first_val
    return base_series * scale


def _build_spread(
    entry_group: pd.DataFrame,
    size_multiplier: float,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_id: int | None = None,
) -> Spread:
    short_legs: list[TradedContract] = []
    long_legs: list[TradedContract] = []

    for _, row in entry_group.iterrows():
        underlying_px = row.get("underlying_close")
        if underlying_px is None or pd.isna(underlying_px):
            underlying_px_val = np.nan
        else:
            try:
                underlying_px_val = float(underlying_px)
            except (TypeError, ValueError):
                underlying_px_val = np.nan
        leg = TradedContract(
            expiry_date_idx=row["expiry_date_idx"],
            call_put_idx=row["call_put_idx"],
            strike_idx=row["strike_idx"],
            base_position=float(row["position"]),
            entry_date=entry_date,
            exit_date=exit_date,
            scaled_position=size_multiplier * row["position"],
            last_price=float(row["mid_price"]),
            recent_price=float(row["mid_price"]),
            delta=float(row["delta"]),
            theta=float(row["theta"]),
            vega=float(row["vega"]),
            underlying_close=underlying_px_val,
            trade_date_idx=_normalise_entry_id(entry_id),
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
    filtered_data = filtered_data.sort_values("norm_trade_date").set_index("norm_trade_date")
    return trade_entries, filtered_data


def _price_lookup_for_date(
    filtered_data: pd.DataFrame,
    current_date_naive: pd.Timestamp,
    keys: list[str],
) -> dict[ContractKey, dict[str, float]]:
    base_price_cols = ["mid_price", "delta", "theta", "vega"]
    optional_cols = [col for col in ["underlying_close"] if col in filtered_data.columns]
    lookup_cols = keys + base_price_cols + optional_cols
    try:
        todays_prices = filtered_data.loc[
            [current_date_naive],
            lookup_cols,
        ]
    except KeyError:
        return {}

    price_frame = todays_prices.set_index(keys)
    has_underlying_close = "underlying_close" in price_frame.columns
    lookup: dict[ContractKey, dict[str, float]] = {}
    for idx, row in price_frame.iterrows():
        lookup[idx] = {
            "mid_price": float(row["mid_price"]),
            "delta": float(row["delta"]),
            "theta": float(row["theta"]),
            "vega": float(row["vega"]),
        }
        if has_underlying_close:
            value = row["underlying_close"]
            if value is None or pd.isna(value):
                lookup[idx]["underlying_close"] = np.nan
            else:
                try:
                    lookup[idx]["underlying_close"] = float(value)
                except (TypeError, ValueError):
                    lookup[idx]["underlying_close"] = np.nan
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
    fixed_notional_exposure: float | None = None,
    entry_notional: bool | str = False,
) -> pd.DataFrame:
    """
    Run the refactored daily backtest loop.

    When ``fixed_notional_exposure`` is supplied (and no explicit ``sizing_fn`` is
    provided), entries are scaled to target that underlying notional each day. Use
    ``entry_notional=True`` to target the notional for new entries without offsetting
    currently held exposure. Set ``entry_notional='underlying'`` to scale the target
    notional by the underlying's own price history from the supplied option data
    (using ``underlying_close``).
    """
    if trade_entries.empty:
        return pd.DataFrame(
            columns=["equity", "equity_obj", "daily_log_return", "underlying_equity", "equity_with_underlying"]
        )

    hold_td = pd.Timedelta(days=hold_period)
    keys = ["expiry_date_idx", "call_put_idx", "strike_idx"]
    trades_df, filtered_data = _prep_trade_frames(trade_entries, option_data)

    if trades_df.empty or filtered_data.empty:
        return pd.DataFrame(
            columns=["equity", "equity_obj", "daily_log_return", "underlying_equity", "equity_with_underlying"]
        )

    if fixed_notional_exposure is not None:
        try:
            target_notional = float(fixed_notional_exposure)
        except (TypeError, ValueError):
            target_notional = None
    else:
        target_notional = None
    if target_notional is not None and target_notional <= 0:
        target_notional = None

    entry_notional_input = entry_notional
    entry_notional_flag = bool(entry_notional_input)
    use_underlying_notional = isinstance(entry_notional_input, str) and entry_notional_input.lower() == "underlying"

    cal = ec.get_calendar(calendar)
    sessions = cal.sessions_in_range(
        trades_df["norm_trade_date"].min(),
        filtered_data.index.max(),
    )

    target_series = None
    if use_underlying_notional and target_notional is not None:
        target_base = _build_underlying_entry_series(filtered_data, sessions, initial_capital)
        if target_base is not None:
            target_series = target_notional * target_base

    target_context: dict[str, pd.Timestamp | None] = {"date": None}

    def _target_supplier() -> float | None:
        date = target_context["date"]
        if target_series is None or date is None:
            return None
        try:
            value = target_series.loc[date]
        except KeyError:
            value = np.nan
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        return value if np.isfinite(value) else None

    use_fixed_notional_sizer = sizing_fn is None and (target_notional is not None or target_series is not None)

    if sizing_fn is None:
        if use_fixed_notional_sizer:
            if target_series is not None:
                sizing_fn = make_variable_notional_sizer(
                    _target_supplier,
                    entry_cost_size,
                    entry_notional_flag,
                )
            else:
                sizing_fn = make_fixed_notional_sizer(
                    target_notional,
                    entry_cost_size,
                    entry_notional_flag,
                )
        else:
            sizing_fn = make_equity_fraction_sizer(capital_per_trade, entry_cost_size)
    if exit_fn is None:
        exit_fn = default_exit_condition

    portfolio = PortfolioState(cash=initial_capital)
    equity_curve: list[dict[str, float | pd.Timestamp | SubStrategy]] = []
    prev_equity = initial_capital
    underlying_series = _build_underlying_entry_series(filtered_data, sessions, initial_capital)
    last_underlying_equity = initial_capital if underlying_series is not None else np.nan

    for current_date in sessions:
        current_date_naive = (
            current_date.tz_localize(None) if getattr(current_date, "tzinfo", None) else current_date
        )
        target_context["date"] = current_date_naive

        todays_entries = trades_df[trades_df["norm_trade_date"] == current_date_naive]
        price_lookup = _price_lookup_for_date(filtered_data, current_date_naive, keys)

        if use_fixed_notional_sizer:
            for spread in portfolio.active_spreads:
                spread.update_prices(price_lookup)

        if not todays_entries.empty:
            for trade_date_idx, entry_group in todays_entries.groupby("trade_date_idx"):
                size_multiplier = sizing_fn(portfolio, entry_group)
                if size_multiplier <= 0:
                    continue

                entry_date = entry_group["trade_date"].iloc[0]
                exit_date = entry_date + hold_td
                entry_identifier = _normalise_entry_id(trade_date_idx)
                spread = _build_spread(
                    entry_group,
                    size_multiplier,
                    entry_date,
                    exit_date,
                    entry_id=entry_identifier,
                )
                net_cost = spread.market_value
                if np.isnan(net_cost):
                    continue

                portfolio.cash -= net_cost
                portfolio.active_spreads.append(spread)

        for spread in portfolio.active_spreads:
            spread.update_prices(price_lookup)

        snapshot = SubStrategy(
            date=current_date_naive,
            active_contracts=[copy.copy(c) for sp in portfolio.active_spreads for c in sp.all_legs],
            current_cash=portfolio.cash,
        )
        current_equity = snapshot.total
        if underlying_series is not None:
            try:
                underlying_equity = float(underlying_series.loc[current_date_naive])
            except KeyError:
                underlying_equity = np.nan
            if np.isfinite(underlying_equity):
                last_underlying_equity = underlying_equity
            else:
                underlying_equity = last_underlying_equity
        else:
            underlying_equity = np.nan
        equity_with_underlying = current_equity + underlying_equity if np.isfinite(underlying_equity) else np.nan
        equity_curve.append(
            {
                "date": current_date_naive,
                "equity": current_equity,
                "equity_obj": snapshot,
                "daily_log_return": np.log(current_equity / prev_equity) if prev_equity > 0 else np.nan,
                "underlying_equity": underlying_equity,
                "equity_with_underlying": equity_with_underlying,
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
    fixed_notional_exposure: float | None = None,
    entry_notional: bool | str = False,
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
    fixed_notional_exposure:
        When provided, overrides percentage-based sizing and instead targets this absolute
        notional exposure (based on ``allocation_basis``/``entry_cost_size``) across the
        combined portfolio.
    entry_notional:
        When True alongside ``fixed_notional_exposure``, target the specified notional for
        the new entries on each day (without offsetting currently held exposure). When
        provided as a ticker symbol, pull that ticker's history via yfinance, scale the
        series to start at ``initial_capital``, and target
        ``fixed_notional_exposure * value_on_date`` for each session. When set to
        ``"underlying"``, construct the series from the available ``underlying_close``
        values in the option data. ``underlying_equity`` uses this series when available;
        otherwise it falls back to the portfolio's implied underlying prices. Outputs
        include ``underlying_equity`` and ``equity_with_underlying`` (options equity +
        underlying_equity).
    """
    if not per_stock_equity:
        return pd.DataFrame(
            columns=[
                "equity",
                "equity_obj",
                "daily_log_return",
                "allocation_underlying_value",
                "stocks_entered",
                "underlying_equity",
                "equity_with_underlying",
            ]
        )

    allocation_fn = capital_allocation or _default_capital_allocation
    if fixed_notional_exposure is not None:
        try:
            target_notional = float(fixed_notional_exposure)
        except (TypeError, ValueError):
            target_notional = None
    else:
        target_notional = None
    if target_notional is not None and target_notional <= 0:
        target_notional = None
    entry_notional_input = entry_notional
    entry_notional_flag = bool(entry_notional_input)
    entry_notional_ticker = entry_notional_input.strip() if isinstance(entry_notional_input, str) else None
    use_underlying_entry_notional = (
        isinstance(entry_notional_ticker, str) and entry_notional_ticker.lower() == "underlying"
    )
    if use_underlying_entry_notional:
        entry_notional_ticker = None

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
        return pd.DataFrame(
            columns=[
                "equity",
                "equity_obj",
                "daily_log_return",
                "allocation_underlying_value",
                "stocks_entered",
                "underlying_equity",
                "equity_with_underlying",
            ]
        )

    entry_notional_series: pd.Series | None = None

    if use_underlying_entry_notional:
        def _underlying_series_from_snapshots() -> pd.Series | None:
            values: dict[pd.Timestamp, float] = {}
            for date in all_dates:
                price_candidates: list[float] = []
                for tick, df in per_stock_equity.items():
                    if df.empty or date not in df.index:
                        continue
                    row = df.loc[date]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    snapshot = row.get("equity_obj")
                    if not isinstance(snapshot, SubStrategy):
                        continue
                    for contract in snapshot.active_contracts:
                        try:
                            px = float(contract.underlying_close)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(px):
                            price_candidates.append(px)
                    if price_candidates:
                        break
                if price_candidates:
                    values[date] = price_candidates[0]
            if not values:
                return None
            series = pd.Series(values).sort_index()
            try:
                series.index = series.index.tz_localize(None)
            except TypeError:
                pass
            first_val = series.dropna().iloc[0] if not series.dropna().empty else np.nan
            if not np.isfinite(first_val) or first_val <= 0:
                return None
            scale = initial_capital / first_val
            return (series * scale).reindex(all_dates).ffill()

        entry_notional_series = _underlying_series_from_snapshots()
    elif entry_notional_ticker:
        entry_notional_series = _load_entry_notional_series(entry_notional_ticker, all_dates, initial_capital)

    def _notional_for_contracts(contracts: list[TradedContract]) -> float:
        long_cost = sum(abs(c.base_position) * c.recent_price for c in contracts if c.base_position > 0)
        short_cost = sum(abs(c.base_position) * c.recent_price for c in contracts if c.base_position < 0)
        if entry_cost_size == "long":
            return long_cost
        if entry_cost_size == "diff":
            return long_cost - short_cost
        if entry_cost_size == "short":
            return short_cost
        raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")

    def _sizing_denominator(contracts: list[TradedContract]) -> float:
        long_cost = sum(abs(c.base_position) * c.recent_price for c in contracts if c.base_position > 0)
        short_cost = sum(abs(c.base_position) * c.recent_price for c in contracts if c.base_position < 0)
        if entry_cost_size == "long":
            return long_cost
        if entry_cost_size == "diff":
            return long_cost - short_cost
        if entry_cost_size == "short":
            return short_cost
        raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")

    def _underlying_allocation_value(contracts: list[TradedContract], *, use_base: bool = False) -> float:
        long_value = 0.0
        short_value = 0.0
        for contract in contracts:
            if use_base:
                reference = contract.base_position if contract.base_position != 0 else contract.scaled_position
            else:
                reference = contract.scaled_position
            try:
                position_value = float(reference)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(position_value) or position_value == 0:
                continue
            price_value = contract.underlying_close
            try:
                underlying_px = float(price_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(underlying_px):
                continue
            exposure = abs(position_value) * float(underlying_px)
            if position_value > 0:
                long_value += exposure
            else:
                short_value += exposure

        if entry_cost_size == "long":
            return long_value
        if entry_cost_size == "diff":
            return long_value - short_value
        if entry_cost_size == "short":
            return short_value
        raise ValueError(f"Unknown entry_cost_size '{entry_cost_size}'")

    holdings: dict[str, dict[PositionKey, TradedContract]] = {tick: {} for tick in per_stock_equity.keys()}
    combined_rows: list[dict[str, object]] = []
    portfolio_cash = initial_capital
    prev_equity = initial_capital
    portfolio_underlying_px0: float | None = None
    last_underlying_equity = np.nan

    for date in all_dates:
        allowed_ticks = _constituents_for(date)
        if allowed_ticks is None:
            active_ticks = list(per_stock_equity.keys())
        else:
            active_ticks = [tick for tick in per_stock_equity.keys() if tick in allowed_ticks]
        if not active_ticks:
            active_ticks = list(per_stock_equity.keys())
        new_entries: dict[str, list[tuple[PositionKey, TradedContract]]] = defaultdict(list)
        entered_ticks_today: set[str] = set()

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
                    target.recent_price = float(source_contract.recent_price)
                    target.delta = float(source_contract.delta)
                    target.theta = float(source_contract.theta)
                    target.vega = float(source_contract.vega)
                    target.has_price_today = bool(source_contract.has_price_today)

            if allowed_ticks is not None and tick not in allowed_ticks:
                continue

            for key, contract in snapshot_contracts.items():
                if key not in state:
                    new_entries[tick].append((key, contract))

        current_contracts_before = [contract for tick_state in holdings.values() for contract in tick_state.values()]
        per_contract_values_before = [
            contract.market_value for tick_state in holdings.values() for contract in tick_state.values()
        ]
        positions_value_before = sum(per_contract_values_before)

        equity_before = portfolio_cash + positions_value_before
        current_underlying_notional = _underlying_allocation_value(current_contracts_before)
        notional_room = None
        daily_target_notional = target_notional
        if entry_notional_series is not None and target_notional is not None:
            try:
                base_value = float(entry_notional_series.loc[date])
            except (KeyError, TypeError, ValueError):
                base_value = np.nan
            if np.isfinite(base_value):
                daily_target_notional = target_notional * base_value
            else:
                daily_target_notional = None
        if daily_target_notional is not None:
            if entry_notional_flag:
                notional_room = daily_target_notional
            else:
                notional_room = max(daily_target_notional - current_underlying_notional, 0.0)

        entering_ticks = [tick for tick, contracts in new_entries.items() if contracts]
        if entering_ticks:
            allocation_inputs = {
                tick: _notional_for_contracts([contract for _, contract in new_entries[tick]])
                for tick in entering_ticks
            }
            weights = allocation_fn(date, allocation_inputs)
            weights = {tick: weights.get(tick, 0.0) for tick in entering_ticks}
            positive_sum = sum(weight for weight in weights.values() if weight > 0)
            if positive_sum <= 0:
                equal_weight = 1.0 / len(entering_ticks)
                weights = {tick: equal_weight for tick in entering_ticks}
            else:
                weights = {
                    tick: (weight / positive_sum) if weight > 0 else 0.0 for tick, weight in weights.items()
                }

            if target_notional is None:
                if equity_before > 0:
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
                        executed = False
                        for key, contract in new_entries[tick]:
                            scaled_contract = copy.copy(contract)
                            base = (
                                contract.base_position
                                if contract.base_position != 0
                                else contract.scaled_position
                            )
                            scaled_contract.scaled_position = base * size_multiplier
                            state[key] = scaled_contract
                            portfolio_cash -= scaled_contract.market_value
                            executed = True
                        if executed:
                            entered_ticks_today.add(tick)
            else:
                if notional_room is not None and notional_room > 0:
                    for tick in entering_ticks:
                        weight = weights.get(tick, 0.0)
                        if weight <= 0:
                            continue
                        per_tick_notional = notional_room * weight
                        if per_tick_notional <= 0:
                            continue
                        denom = _underlying_allocation_value(
                            [contract for _, contract in new_entries[tick]],
                            use_base=True,
                        )
                        if denom <= 0 or np.isnan(denom):
                            continue
                        size_multiplier = per_tick_notional / denom
                        if size_multiplier <= 0 or np.isnan(size_multiplier):
                            continue
                        state = holdings.setdefault(tick, {})
                        executed = False
                        for key, contract in new_entries[tick]:
                            scaled_contract = copy.copy(contract)
                            base = (
                                contract.base_position
                                if contract.base_position != 0
                                else contract.scaled_position
                            )
                            scaled_contract.scaled_position = base * size_multiplier
                            state[key] = scaled_contract
                            portfolio_cash -= scaled_contract.market_value
                            executed = True
                        if executed:
                            entered_ticks_today.add(tick)

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
        allocation_underlying_value = _underlying_allocation_value(combined_contracts)
        current_equity = portfolio_cash + positions_value

        # Track a benchmark “buy and hold” in the entry_notional series (e.g., ^NDX) when supplied.
        if entry_notional_series is not None:
            try:
                underlying_equity = float(entry_notional_series.loc[date])
            except (KeyError, TypeError, ValueError):
                underlying_equity = np.nan
        else:
            underlying_equity = np.nan

        if not np.isfinite(underlying_equity):
            price_candidates: list[float] = []
            for contract in combined_contracts:
                try:
                    px = float(contract.underlying_close)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(px):
                    price_candidates.append(px)
            if price_candidates:
                avg_px = float(np.mean(price_candidates))
                if portfolio_underlying_px0 is None and np.isfinite(avg_px) and avg_px > 0:
                    portfolio_underlying_px0 = avg_px
                if portfolio_underlying_px0 is not None and np.isfinite(avg_px) and avg_px > 0:
                    underlying_equity = initial_capital * (avg_px / portfolio_underlying_px0)
                else:
                    underlying_equity = last_underlying_equity
            else:
                underlying_equity = last_underlying_equity

        if np.isfinite(underlying_equity):
            last_underlying_equity = underlying_equity

        equity_with_underlying = current_equity + underlying_equity if np.isfinite(underlying_equity) else np.nan
        stocks_entered_count = len(entered_ticks_today)
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
                "allocation_underlying_value": allocation_underlying_value,
                "stocks_entered": stocks_entered_count,
                "daily_log_return": daily_log_return,
                "underlying_equity": underlying_equity,
                "equity_with_underlying": equity_with_underlying,
            }
        )
        prev_equity = current_equity

    return pd.DataFrame(combined_rows).set_index("date").sort_index()
