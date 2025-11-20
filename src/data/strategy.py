"""
Definitions for trade strategies
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import logging
import pandas as pd
import numpy as np
import inspect

from scipy import stats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import DataManager


class StrategyWrapper:
    def __init__(self, ticks: list[str] | str, strategy: Strategy):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.strategy: Strategy = strategy
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run transformation methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Features'

        for arg in args:
            method = getattr(self.strategy, arg, None)
            if not callable(method):
                raise AttributeError(f"'{arg}' is not a valid method of Strategy")
            methods.append(method)

        manager = self.strategy.manager
        if manager.load_lazy:
            for method in methods:
                manager.add_method(self.ticks, method, kwargs)
        else:
            for method in methods:
                manager.run_method(self.ticks, method, kwargs)


class Strategy:
   
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> StrategyWrapper:
        return StrategyWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        StrategyWrapper(self.manager.universe.ticks, self)(*args, **kwargs)

    def add(self, name: str, func: Callable) -> None:
        """Dynamically add a new feature method."""
        ref_sig = inspect.signature(self.earnings_calendar_spread)
        func_sig = inspect.signature(func)

        if ref_sig != func_sig:
            raise TypeError(
                f"Function '{func.__name__}' signature {func_sig} does not match expected {ref_sig}"
            )
        setattr(self, name, func.__get__(self))
        return None
    
    @staticmethod
    def ratio_spread(data, **kwargs) -> pd.DataFrame:

        lower_ttm = kwargs.get('lower_ttm', 90)
        upper_ttm = kwargs.get('upper_ttm', 150)
        delta_atm = kwargs.get('delta_atm', 0.45)
        delta_otm = kwargs.get('delta_otm', 0.15)
        otm_ratio = kwargs.get('otm_ratio', 2)
        hold_period = kwargs.get('hold_period', 30)
        call_put = kwargs.get('call_put', 'p')

        data['entered'] = False
        data['position'] = 0

        def pick_daily_contracts(day_chain, **kwargs) -> pd.DataFrame:
            fitler = (
                (day_chain['delta'].abs() <= delta_atm) &
                day_chain['ttm'].between(lower_ttm, upper_ttm) &
                (day_chain['call_put'] == call_put) &
                (day_chain['days_until_last_trade'] > hold_period)
            )
            atm_candidates = day_chain.loc[fitler]
            
            if atm_candidates.empty:
                return day_chain
            
            nearest_atm = atm_candidates.iloc[
                (atm_candidates['delta'].abs() - delta_atm).abs().argsort()[:1]
            ]
            
            otm_candidates = day_chain.loc[fitler & (day_chain['delta'].abs() > delta_otm)]
            
            if otm_candidates.empty:
                return day_chain
        
            nearest_otm = otm_candidates.iloc[(otm_candidates['delta'].abs() - delta_otm).abs().argsort()[:1]]

            if not nearest_atm.empty and not nearest_otm.empty:
                day_chain.loc[nearest_atm.index, 'entered'] = True
                day_chain.loc[nearest_otm.index, 'entered'] = True
                day_chain.loc[nearest_atm.index, 'position'] = -1
                day_chain.loc[nearest_otm.index, 'position'] = otm_ratio


            return day_chain

        grouped = data.groupby('trade_date')
        processed = grouped.apply(pick_daily_contracts)
        result = (
            processed[processed['entered']]
            .drop(columns=['entered'])
            .reset_index(level='trade_date', drop=True)
        )
        # Add these two lines before assigning back into data:
        processed_aligned = processed.droplevel(0).reindex(data.index)

        data['entered'] = processed_aligned['entered']
        data['position'] = processed_aligned['position']

        return result
    
    @staticmethod
    def earnings_calendar_spread(data, **kwargs):
        """Calendar spread around earnings: short near-term, long farther expiry."""
        entry_offset = kwargs.get("entry_offset", 1)
        exit_offset = kwargs.get("exit_offset", 0)
        exit_tolerance = kwargs.get("exit_tolerance", 3)
        call_delta = kwargs.get("call_delta", 0.35)
        put_delta = kwargs.get("put_delta", -0.35)
        ttm_short = kwargs.get("ttm_short", 14)
        ttm_long = kwargs.get("ttm_long", 30)
        position_size = kwargs.get("position", 1)

        cols = [
            "trade_exit_date",
            "position",
            "mid_price_entry",
            "mid_price_exit",
            "pnl",
            "theta_entry",
        ]
        if data.empty:
            return pd.DataFrame(index=data.index, columns=cols)

        base_index = data.index
        entries = []

        # --- entry candidates ---
        entry_window = sorted((0, entry_offset))
        exit_window = sorted((exit_offset, exit_offset + exit_tolerance))
        entry_mask = data["bdays_to_earnings"].between(*entry_window)
        exit_mask = data["bdays_since_earnings"].between(*exit_window)

        entry_df = data.loc[entry_mask].copy()
        exit_df = data.loc[exit_mask].copy()

        if entry_df.empty or exit_df.empty:
            return pd.DataFrame(index=base_index, columns=cols)

        def pick_leg(frame, ttm_filter, delta_filter):
            leg = frame.loc[ttm_filter(frame)].loc[delta_filter]
            if leg.empty:
                return leg
            # one contract per trade_date per call/put direction
            return (
                leg.reset_index()
                .sort_values("trade_date_idx")
                .groupby(["trade_date_idx", "call_put"])
                .apply(lambda g: g.loc[g["ttm"].idxmin()])
                .droplevel(-1)
            )

        short_calls = pick_leg(
            entry_df,
            lambda df: df["ttm"] <= ttm_short,
            lambda df: df["delta"] >= call_delta,
        )
        long_calls = pick_leg(
            entry_df,
            lambda df: df["ttm"] >= ttm_long,
            lambda df: df["delta"] >= call_delta,
        )
        short_puts = pick_leg(
            entry_df,
            lambda df: df["ttm"] <= ttm_short,
            lambda df: df["delta"] <= put_delta,
        )
        long_puts = pick_leg(
            entry_df,
            lambda df: df["ttm"] >= ttm_long,
            lambda df: df["delta"] <= put_delta,
        )
        def sanitize_leg(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            if not set(base_index.names).issubset(df.columns):
                df = df.reset_index()
            else:
                df = df.copy()
            df.index = pd.RangeIndex(len(df))
            return df

        short_calls = sanitize_leg(short_calls)
        long_calls = sanitize_leg(long_calls)
        short_puts = sanitize_leg(short_puts)
        long_puts = sanitize_leg(long_puts)

        join_keys = {"trade_date_idx"}

        def build_pairs(short_leg, long_leg, call_put):
            merged = short_leg.merge(
                long_leg,
                on=["trade_date_idx", "call_put"],
                suffixes=("_short", "_long"),
            )
            records = []
            for _, row in merged.iterrows():
                idx_short = tuple(
                    row[name] if name in join_keys else row[f"{name}_short"]
                    for name in base_index.names
                )
                idx_long = tuple(
                    row[name] if name in join_keys else row[f"{name}_long"]
                    for name in base_index.names
                )
                short_row = data.loc[idx_short].copy()
                long_row = data.loc[idx_long].copy()

                records.append(
                    {
                        "index": idx_short,
                        "trade_exit_date": None,
                        "position": -position_size,
                        "mid_price_entry": short_row["mid_price"],
                        "theta_entry": short_row.get("theta", np.nan),
                    }
                )
                records.append(
                    {
                        "index": idx_long,
                        "trade_exit_date": None,
                        "position": position_size,
                        "mid_price_entry": long_row["mid_price"],
                        "theta_entry": long_row.get("theta", np.nan),
                    }
                )
            return records

        entries.extend(build_pairs(short_calls, long_calls, "c"))
        entries.extend(build_pairs(short_puts, long_puts, "p"))

        if not entries:
            return pd.DataFrame(index=base_index, columns=cols)

        result = pd.DataFrame(entries)
        if result.empty:
            return pd.DataFrame(index=base_index, columns=cols)
        result = result.set_index("index").reindex(base_index)

        # --- exit matching ---
        exit_info = (
            exit_df.reset_index()
            .sort_values("bdays_since_earnings")
            .groupby(["strike_idx", "expiry_date_idx", "call_put_idx"], as_index=False)
            .first()[["strike_idx", "expiry_date_idx", "call_put_idx", "trade_date_idx", "mid_price"]]
            .rename(
                columns={
                    "trade_date_idx": "trade_exit_date",
                    "mid_price": "mid_price_exit",
                }
            )
        )

        result = result.reset_index()
        result = result.drop(columns=["trade_exit_date", "mid_price_exit"], errors="ignore")
        result = result.merge(
            exit_info,
            on=["strike_idx", "expiry_date_idx", "call_put_idx"],
            how="left",
        )
        result = result.set_index(base_index.names).reindex(base_index)

        result["pnl"] = result["position"] * (
            result["mid_price_exit"] - result["mid_price_entry"]
        )
        result = result.loc[:, ~result.columns.duplicated()]
        return result[cols].loc[result["mid_price_entry"].notna()]
   
