"""
Data featuresfunctions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import pandas as pd
import numpy as np
import inspect

from . import backtest_engine as backtest_engine

if TYPE_CHECKING:
    from .manager import DataManager

# Re-export dataclasses for backward compatibility
TradedContract = backtest_engine.TradedContract
Spread = backtest_engine.Spread
SubStrategy = backtest_engine.SubStrategy

def all_stocks(method: Callable) -> Callable:
    """Decorator to flag methods that should run once using all stocks."""
    target = method.__func__ if isinstance(method, staticmethod) else method
    setattr(target, "_all_stocks", True)
    return method


def _propagate_columns_to_output(
    output_df: pd.DataFrame,
    source_df: pd.DataFrame | None,
    columns: list[str] | tuple[str, ...] | str,
    fallback_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach per-day values from ``source_df`` onto a backtest output frame.
    Columns are averaged by trade_date to match the equity index.
    """
    if output_df is None or output_df.empty:
        return output_df

    if isinstance(columns, str):
        columns = [columns]
    columns = [c for c in columns if c]
    if not columns:
        return output_df

    def _norm_dates(values: pd.Series | pd.Index) -> pd.Series | pd.DatetimeIndex:
        norm = pd.to_datetime(values, errors="coerce")
        try:
            norm = norm.tz_localize(None)
        except TypeError:
            try:
                norm = norm.tz_convert(None)
            except Exception:
                pass
        return norm

    def _find_date_series(df: pd.DataFrame) -> pd.Series:
        idx = df.index
        idx_names = getattr(idx, "names", None) or []
        if "trade_date_idx" in df.columns:
            return _norm_dates(df["trade_date_idx"])
        if "trade_date_idx" in idx_names:
            return _norm_dates(idx.get_level_values("trade_date_idx"))
        if "trade_date" in df.columns:
            return _norm_dates(df["trade_date"])
        if "trade_date" in idx_names:
            return _norm_dates(idx.get_level_values("trade_date"))
        if isinstance(idx, pd.MultiIndex):
            try:
                return _norm_dates(idx.get_level_values(0))
            except Exception:
                try:
                    return _norm_dates(idx.get_level_values(-1))
                except Exception:
                    pass
        return _norm_dates(idx)

    base_df = source_df if source_df is not None else fallback_df
    if base_df is None or base_df.empty:
        return output_df

    available_cols = [c for c in columns if c in base_df.columns]
    if not available_cols and fallback_df is not None and fallback_df is not base_df:
        base_df = fallback_df
        available_cols = [c for c in columns if c in base_df.columns]
    if not available_cols:
        return output_df

    date_key = pd.to_datetime(_find_date_series(base_df)).normalize()
    if date_key.isna().all():
        return output_df

    try:
        aggregated = (
            pd.DataFrame(
                {
                    col: pd.to_numeric(base_df[col], errors="coerce")
                    for col in available_cols
                }
            )
            .groupby(date_key)
            .mean()
        )
    except Exception:
        # On any unexpected shape/memory issues, skip propagation silently.
        return output_df

    if aggregated.empty:
        return output_df

    try:
        target_dates = pd.to_datetime(_norm_dates(output_df.index)).normalize()
        output_df = output_df.copy()
        for col in aggregated.columns:
            output_df[col] = target_dates.map(aggregated[col])
    except Exception:
        return output_df
    return output_df

# ---- Equity calc: cash + MTM of all active contracts ----
def calculate_equity(data, current_cash, traded_contracts):
    # `data` not needed here; we use last_price already stored on each contract
    positions_value = sum(
        c.scaled_position * c.last_price
        for c in traded_contracts
    )
    return current_cash + positions_value


class BacktestWrapper:
    def __init__(self, ticks: list[str] | str, features: Backtest):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.features: Backtest = features
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run features methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Backtest'

        for arg in args:
            method = getattr(self.features, arg, None)
            if not callable(method):
                raise AttributeError(f"'{arg}' is not a valid feature")
            methods.append(method)

        manager = self.features.manager
        if manager.load_lazy:
            for method in methods:
                manager.add_method(self.ticks, method, kwargs)
        else:
            for method in methods:
                manager.run_method(self.ticks, method, kwargs)

class Backtest:
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> BacktestWrapper:
        return BacktestWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        BacktestWrapper(self.manager.universe.ticks, self)(*args, **kwargs)
        return None
    
    def add(self, name: str, func: Callable) -> None:
        """Dynamically add a new feature method."""
        ref_sig = inspect.signature(self.iv_rv_ratio)
        func_sig = inspect.signature(func)

        if ref_sig != func_sig:
            raise TypeError(
                f"Function '{func.__name__}' signature {func_sig} does not match expected {ref_sig}"
            )
        setattr(self, name, func.__get__(self))
        return None

    @staticmethod
    def fixed_hold_trade(trade_entries, data, **kwargs) -> pd.DataFrame:

        # ---- Params ----
        hold_period_days = kwargs.get('hold_period', 30)
        hold_period = pd.Timedelta(days=hold_period_days)

        keys = ['expiry_date_idx', 'call_put_idx', 'strike_idx']
        entry_cost_param = kwargs.get('entry_cost_size', 'short')  # 'short', 'long', 'diff'
        initial_capital = kwargs.get('initial_capital', 100000.0)
        capital_per_trade = kwargs.get('capital_per_trade', 0.05)
        all_returns = kwargs.get('all_returns', False)

        # Ensure sorted by trade_date_idx
        trade_entries = trade_entries.sort_values('trade_date_idx')

        def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
            ts = pd.to_datetime(ts)
            return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts

        sequential_entries: list[tuple[int, pd.Timestamp]] = []
        selected_groups: list[pd.DataFrame] = []
        next_entry_allowed: pd.Timestamp | None = None

        for trade_date_idx, entry_group in trade_entries.groupby('trade_date_idx'):
            entry_date = _to_naive(entry_group['trade_date'].iloc[0])
            if next_entry_allowed is not None and entry_date < next_entry_allowed:
                continue
            sequential_entries.append((trade_date_idx, entry_date))
            selected_groups.append(entry_group)
            next_entry_allowed = entry_date + hold_period

        if not selected_groups:
            return pd.DataFrame(columns=['log_return'])

        filtered_entries = pd.concat(selected_groups).sort_values('trade_date_idx')

        equity_df = backtest_engine.main(
            filtered_entries,
            data,
            hold_period=hold_period_days,
            initial_capital=float(initial_capital),
            capital_per_trade=float(capital_per_trade),
            entry_cost_size=entry_cost_param,
            sizing_fn=kwargs.get('sizing_fn'),
            exit_fn=kwargs.get('exit_fn'),
            calendar=kwargs.get('calendar', 'XNAS'),
        )

        if equity_df.empty:
            return pd.DataFrame(columns=['log_return'])

        log_returns = equity_df['daily_log_return']

        if all_returns:
            return log_returns.to_frame(name='log_return')

        trade_returns: dict[int, float] = {}
        for trade_date_idx, entry_date in sequential_entries:
            exit_date = entry_date + hold_period
            mask = (log_returns.index >= entry_date) & (log_returns.index <= exit_date)
            if not mask.any():
                continue
            trade_returns[trade_date_idx] = log_returns.loc[mask].sum()

        result = (
            pd.Series(trade_returns, name='log_return')
            .to_frame()
            .sort_index()
        )
        propagate_cols = kwargs.get("propagate_columns") or kwargs.get("propagate_cols")
        if propagate_cols:
            try:
                source_df = trade_entries if isinstance(trade_entries, pd.DataFrame) else pd.DataFrame()
                result = _propagate_columns_to_output(result, source_df, propagate_cols, fallback_df=data)
            except Exception:
                pass
        return result

    @staticmethod
    def multi_day_entry(trade_entries, data, **kwargs) -> pd.DataFrame:
        propagate_cols = kwargs.pop("propagate_columns", None) or kwargs.pop("propagate_cols", None)
        allocation_basis = kwargs.get('allocation_basis')
        if allocation_basis is None:
            allocation_basis = kwargs.get('entry_cost_size', 'short')
        fixed_notional_exposure = kwargs.get('fixed_notional_exposure')
        entry_notional = bool(kwargs.get('entry_notional', False))
        equity_df = backtest_engine.main(
            trade_entries,
            data,
            hold_period=kwargs.get('hold_period', 30),
            initial_capital=float(kwargs.get('initial_capital', 100000.0)),
            capital_per_trade=float(kwargs.get('capital_per_trade', 0.05)),
            entry_cost_size=allocation_basis,
            sizing_fn=kwargs.get('sizing_fn'),
            exit_fn=kwargs.get('exit_fn'),
            calendar=kwargs.get('calendar', 'XNAS'),
            fixed_notional_exposure=None if fixed_notional_exposure is None else float(fixed_notional_exposure),
            entry_notional=entry_notional,
        )
        if propagate_cols:
            try:
                source_df = trade_entries if isinstance(trade_entries, pd.DataFrame) else pd.DataFrame()
                equity_df = _propagate_columns_to_output(equity_df, source_df, propagate_cols, fallback_df=data)
            except Exception:
                pass
        return equity_df

    @staticmethod
    def compress_trades_day(equity_df: pd.DataFrame, data=None, **kwargs) -> pd.DataFrame:
        """
        Compress a ``backtest_engine.main`` equity frame into per-entry rows.
        ``data`` is accepted for pipeline compatibility but ignored.

        Parameters
        ----------
        equity_df:
            Output DataFrame from ``backtest_engine.main`` (e.g. ``multi_day_entry``) that
            contains ``equity_obj`` snapshots for each session.
        initial_capital / cash_init:
            Baseline capital used when translating entry/exit quotes into log returns via
            ``log((cash + exit_value) / (cash + entry_value))``.
        include_open:
            When True, still-open positions at the tail of the DataFrame are emitted using
            their latest marks. Defaults to False (drops incomplete trades).
        """
        # Optional: carry selected features through to the compressed trades.
        feature_cols = kwargs.get("feature_cols") or kwargs.get("feature_names") or kwargs.get("propagate_columns") or kwargs.get("propagate_cols") or []
        # Avoid duplicate feature names; duplicates can produce nested Series when aggregating.
        feature_cols = list(dict.fromkeys(feature_cols))
        empty_idx = pd.DatetimeIndex([], name='trade_date')
        empty_cols = ['trade_date_idx', 'exit_date', 'quote_pnl', 'entry_value', 'exit_value', 'holding_days', 'log_return'] + list(feature_cols)
        if equity_df is None or equity_df.empty:
            return pd.DataFrame(columns=empty_cols, index=empty_idx)
        if 'equity_obj' not in equity_df.columns:
            raise ValueError("compress_trades_day expects a DataFrame with an 'equity_obj' column.")

        feature_map: dict[any, dict[str, float]] = {}
        if feature_cols and data is not None:
            entry_df = data.copy()
            # Drop duplicate columns to prevent object-valued means.
            entry_df = entry_df.loc[:, ~entry_df.columns.duplicated()]
            if "entered" in entry_df.columns:
                entry_df = entry_df[entry_df["entered"]]
            entry_df = entry_df.reset_index()
            if "trade_date_idx" in entry_df.columns:
                grouped = entry_df.groupby("trade_date_idx")
            elif "trade_date" in entry_df.columns:
                grouped = entry_df.groupby("trade_date")
            else:
                grouped = None

            if grouped is not None:
                for entry_id, group in grouped:
                    feature_map[entry_id] = {
                        col: group[col].mean() if col in group.columns else np.nan
                        for col in feature_cols
                    }

        equity_df = equity_df.sort_index()
        include_open = bool(kwargs.get('include_open', False))

        def _to_naive(ts) -> pd.Timestamp:
            if ts is None or pd.isna(ts):
                return pd.NaT
            ts = pd.to_datetime(ts)
            return ts.tz_localize(None) if getattr(ts, 'tzinfo', None) else ts

        def _mtm_log_return(entry_val: float, exit_val: float) -> float:
            """Mark-to-market log return ignoring cash buffers/credits; pure price change."""
            if entry_val == 0 or not np.isfinite(entry_val) or not np.isfinite(exit_val):
                return np.nan
            ratio = exit_val / entry_val
            if ratio <= 0 or not np.isfinite(ratio):
                return np.nan
            return np.log(ratio)

        def _entry_value(contracts: list[backtest_engine.TradedContract]) -> float:
            """Cost/credit at entry based on market value when first seen."""
            return sum(contract.market_value for contract in contracts)

        def _group_signature(contracts: list[backtest_engine.TradedContract]) -> tuple:
            """Fallback identifier when trade_date_idx is missing."""
            return tuple(
                sorted(
                    (
                        pd.Timestamp(contract.expiry_date_idx),
                        int(contract.call_put_idx),
                        float(contract.strike_idx),
                        float(np.sign(contract.scaled_position)),
                        float(abs(contract.scaled_position)),
                    )
                    for contract in contracts
                )
            )

        active_trades: dict[tuple[tuple, pd.Timestamp], dict[str, any]] = {}
        completed: list[dict[str, any]] = []

        for current_date, row in equity_df.iterrows():
            snapshot = row.get('equity_obj')
            if not isinstance(snapshot, SubStrategy):
                continue

            current_date_naive = _to_naive(current_date)
            todays_contracts = snapshot.active_contracts or []

            grouped_contracts: dict[tuple[tuple, pd.Timestamp], dict[str, any]] = {}
            coarse_groups: dict[tuple[any, pd.Timestamp], list[backtest_engine.TradedContract]] = {}
            for contract in todays_contracts:
                entry_date = _to_naive(contract.entry_date)
                coarse_key = (getattr(contract, 'trade_date_idx', None), entry_date)
                coarse_groups.setdefault(coarse_key, []).append(contract)

            for (entry_id, entry_date), contracts in coarse_groups.items():
                if not contracts:
                    continue
                if entry_id is None:
                    identifier = ('sig', _group_signature(contracts))
                else:
                    identifier = ('id', entry_id)
                entry_key = (identifier, entry_date)
                grouped_contracts[entry_key] = {
                    'contracts': contracts,
                    'trade_date_idx': entry_id,
                }

            seen_keys: set[tuple[tuple, pd.Timestamp]] = set()
            for entry_key, payload in grouped_contracts.items():
                seen_keys.add(entry_key)
                contracts = payload['contracts']
                current_value = sum(contract.market_value for contract in contracts)
                exit_hint = _to_naive(contracts[0].exit_date)
                trade_idx = payload['trade_date_idx']
                state = active_trades.get(entry_key)
                if state is None:
                    entry_value = _entry_value(contracts)
                    active_trades[entry_key] = {
                        'entry_value': entry_value,
                        'last_value': current_value,
                        'entry_date': entry_key[1],
                        'trade_date_idx': trade_idx,
                        'last_seen': current_date_naive,
                        'exit_hint': exit_hint,
                    }
                else:
                    state['last_value'] = current_value
                    state['last_seen'] = current_date_naive
                    state['exit_hint'] = exit_hint

            missing_keys = [key for key in list(active_trades.keys()) if key not in seen_keys]
            for key in missing_keys:
                state = active_trades.pop(key)
                exit_date = state['last_seen'] or state['exit_hint']
                completed.append(
                    {
                        'trade_date': state['entry_date'],
                        'trade_date_idx': state['trade_date_idx'],
                        'exit_date': exit_date,
                        'entry_value': state.get('entry_value', np.nan),
                        'exit_value': state.get('last_value', np.nan),
                        'quote_pnl': _mtm_log_return(state['entry_value'], state['last_value']),
                        'holding_days': (exit_date - state['entry_date']).days if pd.notna(exit_date) else np.nan,
                        **feature_map.get(state.get('trade_date_idx'), {}),
                    }
                )

        if include_open:
            for state in active_trades.values():
                exit_date = state['last_seen'] or state['exit_hint']
                completed.append(
                    {
                        'trade_date': state['entry_date'],
                        'trade_date_idx': state['trade_date_idx'],
                        'exit_date': exit_date,
                        'entry_value': state.get('entry_value', np.nan),
                        'exit_value': state.get('last_value', np.nan),
                        'quote_pnl': _mtm_log_return(state['entry_value'], state['last_value']),
                        'holding_days': (exit_date - state['entry_date']).days if pd.notna(exit_date) else np.nan,
                        **feature_map.get(state.get('trade_date_idx'), {}),
                    }
                )

        if not completed:
            return pd.DataFrame(columns=empty_cols, index=empty_idx)

        results = pd.DataFrame(completed)
        results['trade_date'] = pd.to_datetime(results['trade_date']).dt.tz_localize(None)
        results['exit_date'] = pd.to_datetime(results['exit_date']).dt.tz_localize(None)
        results = results.sort_values(['trade_date', 'trade_date_idx']).set_index('trade_date')
        # Convenience alias for modelling consistency
        results['log_return'] = results['quote_pnl']
        return results

    @all_stocks
    @staticmethod
    def combined_multiday_entry(per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        # Backwards compatibility: defer to multi_stock implementation.
        return Backtest.multi_stock(per_stock_equity, **kwargs)

    @all_stocks
    @staticmethod
    def filter_by(per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> dict[str, pd.DataFrame]:
        """
        Cross-sectional entry filter applied before ``multi_stock``.

        Keeps only the top ``n_taken`` ticks each day by ``col_name`` (using ``max`` or
        ``min``), optionally restricting the ranking universe to Nasdaq constituents when
        ``ndx_filter`` is True. Existing positions continue to flow through even if a tick
        drops out of the eligible set on a later date; only fresh entries are blocked.
        """
        if not isinstance(per_stock_equity, dict):
            raise TypeError("filter_by expects a mapping of ticker -> equity DataFrame.")

        col_name = kwargs.get("col_name")
        if not col_name:
            raise ValueError("filter_by requires a 'col_name' to rank by.")

        direction = str(kwargs.get("sdirection", "max")).lower()
        if direction not in {"max", "min"}:
            raise ValueError("sdirection must be 'max' or 'min'.")
        reverse = direction == "max"

        try:
            n_taken = int(kwargs.get("n_taken", 1))
        except (TypeError, ValueError):
            n_taken = 0
        if n_taken <= 0:
            return per_stock_equity

        ndx_filter = bool(kwargs.get("ndx_filter", False))

        def _norm_date(ts) -> pd.Timestamp:
            ts = pd.to_datetime(ts)
            try:
                return ts.tz_localize(None)
            except TypeError:
                return ts.tz_convert(None)

        top_per_date = kwargs.get("top_per_date")
        top_series: pd.Series | None = None
        if ndx_filter and top_per_date is not None:
            if isinstance(top_per_date, pd.Series):
                top_series = top_per_date.copy()
            elif isinstance(top_per_date, (list, tuple)):
                top_dict = {}
                for item in top_per_date:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        top_dict[pd.to_datetime(item[0])] = item[1]
                if top_dict:
                    top_series = pd.Series(top_dict)
            if top_series is not None:
                top_series.index = top_series.index.map(_norm_date)
                top_series = top_series.sort_index()

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

        all_dates = sorted(
            {_norm_date(idx) for df in per_stock_equity.values() if df is not None and not df.empty for idx in df.index}
        )
        if not all_dates:
            return per_stock_equity

        norm_index_map: dict[str, pd.DatetimeIndex] = {}
        for tick, df in per_stock_equity.items():
            if df is None or df.empty:
                continue
            norm_index_map[tick] = pd.DatetimeIndex([_norm_date(idx) for idx in df.index])

        selected_by_date: dict[pd.Timestamp, set[str]] = {}
        saw_metric = False

        for date in all_dates:
            constituents = _constituents_for(date) if ndx_filter else None
            candidates: list[tuple[float, str]] = []
            for tick, df in per_stock_equity.items():
                norm_index = norm_index_map.get(tick)
                if df is None or df.empty or norm_index is None:
                    continue
                matches = np.nonzero(norm_index == date)[0]
                if matches.size == 0:
                    continue
                row = df.iloc[matches[0]] if matches.size == 1 else df.iloc[matches[-1]]
                if constituents is not None and tick not in constituents:
                    continue
                if col_name not in row:
                    continue
                saw_metric = True
                try:
                    metric = float(row[col_name])
                except (TypeError, ValueError):
                    metric = np.nan
                if not np.isfinite(metric):
                    continue
                candidates.append((metric, tick))

            if not candidates:
                continue

            candidates.sort(key=lambda pair: pair[0], reverse=reverse)
            selected_by_date[date] = set(tick for _, tick in candidates[:n_taken])

        if not saw_metric:
            raise ValueError(f"Column '{col_name}' not found in per-stock equity frames.")
        if not selected_by_date:
            return per_stock_equity

        filtered: dict[str, pd.DataFrame] = {}
        active_positions: dict[str, set[backtest_engine.PositionKey]] = {tick: set() for tick in per_stock_equity.keys()}

        for tick, df in per_stock_equity.items():
            if df is None or df.empty:
                filtered[tick] = df
                continue

            df_copy = df.copy()
            updated_objs: list[backtest_engine.SubStrategy | object] = []

            for idx, row in df_copy.iterrows():
                norm_idx = _norm_date(idx)
                snapshot = row.get("equity_obj")
                if not isinstance(snapshot, backtest_engine.SubStrategy):
                    updated_objs.append(snapshot)
                    continue

                allowed_entries = selected_by_date.get(norm_idx, set())
                constituents = _constituents_for(norm_idx) if ndx_filter else None
                current_keys = active_positions.setdefault(tick, set())
                kept_contracts: list[backtest_engine.TradedContract] = []
                seen_keys: set[backtest_engine.PositionKey] = set()

                for contract in snapshot.active_contracts:
                    entry_date = _norm_date(contract.entry_date)
                    is_entry_today = entry_date == norm_idx
                    key = contract.position_key
                    is_new = key not in current_keys
                    allow_entry = tick in allowed_entries
                    if constituents is not None and tick not in constituents:
                        allow_entry = False
                    if not is_new or (allow_entry and is_entry_today):
                        kept_contracts.append(contract)
                        seen_keys.add(key)

                current_keys.intersection_update(seen_keys)
                current_keys.update(seen_keys)

                updated_objs.append(
                    backtest_engine.SubStrategy(
                        date=snapshot.date,
                        active_contracts=kept_contracts,
                        current_cash=snapshot.current_cash,
                    )
                )

            df_copy["equity_obj"] = updated_objs
            filtered[tick] = df_copy

        return filtered

    @all_stocks
    @staticmethod
    def multi_stock(per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        if not isinstance(per_stock_equity, dict):
            raise TypeError("multi_stock expects a mapping of ticker -> equity DataFrame.")
        capital_fraction = float(kwargs.get('capital_per_trade', kwargs.get('capital_per_day', 0.05)))
        allocation_basis = kwargs.get('allocation_basis')
        if allocation_basis is None:
            allocation_basis = kwargs.get('entry_cost_size', 'short')
        fixed_notional_exposure = kwargs.get('fixed_notional_exposure')
        entry_notional_param = kwargs.get('entry_notional', False)
        entry_notional = entry_notional_param if isinstance(entry_notional_param, str) else bool(entry_notional_param)
        fixed_notional_value = None
        if fixed_notional_exposure is not None:
            try:
                fixed_notional_value = float(fixed_notional_exposure)
            except (TypeError, ValueError):
                fixed_notional_value = None
        return backtest_engine.combine_multi_equity(
            per_stock_equity,
            initial_capital=float(kwargs.get('initial_capital', 100_000.0)),
            capital_per_trade=capital_fraction,
            capital_allocation=kwargs.get('capital_allocation'),
            entry_cost_size=allocation_basis,
            top_per_date=kwargs.get('top_per_date'),
            fixed_notional_exposure=fixed_notional_value,
            entry_notional=entry_notional,
        )

    @all_stocks
    def multi_stock_single_ep(self, per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Combine equity curves but allocate the full daily notional to a single ticker chosen by
        the lowest available E/P ratio on that date.
        """
        ep_lookup: dict[str, pd.Series] = {}
        ep_source = kwargs.get("ep_ratio_map")

        def _ep_series_for_tick(tick: str) -> pd.Series | None:
            series = None
            if isinstance(ep_source, dict) and tick in ep_source:
                series = ep_source[tick]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
            elif getattr(self.manager, "data", None) is not None:
                raw = self.manager.data.get(tick)
                if raw is not None and not raw.empty and "ep_ratio" in raw.columns and "trade_date" in raw.columns:
                    series = pd.Series(raw["ep_ratio"].values, index=pd.to_datetime(raw["trade_date"], errors="coerce"))
            if series is None:
                return None
            series = pd.to_numeric(series, errors="coerce")
            series.index = pd.to_datetime(series.index, errors="coerce")
            try:
                series.index = series.index.tz_localize(None)
            except TypeError:
                pass
            daily = series.groupby(series.index.normalize()).mean().dropna()
            return daily

        for tick in per_stock_equity.keys():
            ep_series = _ep_series_for_tick(tick)
            if ep_series is not None and not ep_series.empty:
                ep_lookup[tick] = ep_series

        def _single_ep_allocation(date: pd.Timestamp, allocation_inputs: dict[str, float]) -> dict[str, float]:
            if not allocation_inputs:
                return {}
            candidates: list[tuple[float, str]] = []
            for tick in allocation_inputs.keys():
                series = ep_lookup.get(tick)
                if series is None or series.empty:
                    continue
                try:
                    value = series.loc[:date].iloc[-1]
                except Exception:
                    continue
                if np.isfinite(value):
                    candidates.append((value, tick))

            if candidates:
                candidates.sort(key=lambda pair: pair[0])
                chosen_tick = candidates[0][1]
            else:
                chosen_tick = next(iter(allocation_inputs))

            return {tick: (1.0 if tick == chosen_tick else 0.0) for tick in allocation_inputs}

        args = dict(kwargs)
        args["capital_allocation"] = _single_ep_allocation
        return Backtest.multi_stock(per_stock_equity, **args)

    @all_stocks
    @staticmethod
    def model_pass_through(per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Flatten per-stock compressed trade frames into a single DataFrame with a ``tick`` column.
        Intended for piping ``compress_trades_day`` outputs straight into modelling code.
        """
        frames: list[pd.DataFrame] = []

        for tick, df in per_stock_equity.items():
            if df is None or df.empty:
                continue

            tmp = df.copy()
            if 'trade_date' in tmp.columns:
                trade_dates = pd.to_datetime(tmp['trade_date'], errors='coerce')
            else:
                trade_dates = pd.to_datetime(tmp.index, errors='coerce')
                tmp = tmp.reset_index().rename(columns={'index': 'trade_date'})

            # Handle both Series and DatetimeIndex without relying on .dt for Index
            if isinstance(trade_dates, pd.DatetimeIndex):
                try:
                    trade_dates = trade_dates.tz_localize(None)
                except TypeError:
                    trade_dates = trade_dates.tz_convert(None)
            else:
                try:
                    trade_dates = trade_dates.dt.tz_localize(None)
                except TypeError:
                    trade_dates = trade_dates.dt.tz_convert(None)

            tmp['trade_date'] = trade_dates
            tmp['tick'] = tick
            frames.append(tmp)

        if not frames:
            return pd.DataFrame(columns=['tick', 'trade_date'])

        combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        sort_cols = [col for col in ['trade_date', 'tick', 'trade_date_idx'] if col in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)
        return combined
