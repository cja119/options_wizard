"""
Definitions for trade strategies
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import pandas as pd
import numpy as np
import inspect

from scipy import stats

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
        """
        Implements a ratio spread option strategy that works for both calls and puts.
        Long 1 option near target delta (by absolute value), short N options at a
        strike giving ~zero cost. Exits after a fixed hold period (trading days).
        """
        entry_ttm = kwargs.get("entry_ttm", 90)
        ttm_tol = kwargs.get("ttm_tol", 5)
        delta_long = kwargs.get("moneyness_long", 0.35)
        delta_short = kwargs.get("moneyness_short", 0.10)
        short_ratio = kwargs.get("short_ratio", 3)
        delta_tol = kwargs.get("moneyness_tol", 0.05)
        call_put = kwargs.get("call_put", "p")
        hold_period = kwargs.get("hold_period", 30)
        position = kwargs.get("position", 1)

        data = data.copy()
        data["abs_delta"] = data["delta"].abs()
        data['data_index'] = data.index

        filtered_df = data[
            (data["call_put"] == call_put)
            & (data["ttm"].between(entry_ttm - ttm_tol, entry_ttm + ttm_tol))
            & (data["abs_delta"].between(delta_short, delta_long))
        ]

        if filtered_df.empty:
            return pd.DataFrame(columns=[
                "trade_exit_date", "position", "mid_price_entry", "mid_price_exit", "pnl"
            ])

        delta_lower = delta_long - delta_tol
        delta_upper = delta_long + delta_tol

        long_df = filtered_df[
            (filtered_df["abs_delta"] > delta_lower) &
            (filtered_df["abs_delta"] < delta_upper)
        ]

        if long_df.empty:
            return pd.DataFrame(columns=[
                "trade_exit_date", "position", "mid_price_entry", "mid_price_exit", "pnl"
            ])

        longs = long_df.loc[long_df.groupby("expiry_date")["ttm"].idxmin()]
        longs = longs.loc[longs.groupby("expiry_date")["abs_delta"].idxmin()]

        shorts = []
        for (trade_date, expiry_date), long_row in longs.groupby(["trade_date", "expiry_date"]):
            subset = filtered_df[
                (filtered_df["trade_date"] == trade_date) &
                (filtered_df["expiry_date"] == expiry_date)
            ]
            subset = subset.drop(index=long_row["data_index"].values, errors="ignore")
            if subset.empty:
                continue
            target_price = long_row["mid_price"].values[0] / short_ratio
            short_candidate = subset.iloc[
                (subset["mid_price"] - target_price).abs().argsort()[:1]
            ]
            shorts.append(short_candidate)

        if not shorts:
            return pd.DataFrame(columns=[
                "trade_exit_date", "position", "mid_price_entry", "mid_price_exit", "pnl"
            ]).set_index(pd.Index([], name=data.index.names[0]))

        shorts = pd.concat(shorts).copy()
        longs = longs.copy()

        longs["position"] = +1 * position
        shorts["position"] = -short_ratio * position

        entries = pd.concat([longs, shorts])
        entry_prices = entries["mid_price"].copy()

        exit_prices = pd.Series(index=entries.index, dtype=float)
        exit_date = pd.Series(index=entries.index, dtype=int)
        hold_td = pd.Timedelta(days=hold_period)

        for idx, row in entries.iterrows():
            trade_idx = row.name[1] if isinstance(row.name, tuple) else row.name
            candidate_exits = data[
                (data["trade_date"] > trade_idx) &
                (data["trade_date"] <= trade_idx + hold_td) &
                (data["strike"] == row["strike"]) &
                (data["expiry_date"] == row["expiry_date"]) &
                (data["call_put"] == row["call_put"])
            ]
            if not candidate_exits.empty:
                exit_row = candidate_exits.iloc[-1]
                exit_prices.loc[idx] = exit_row["mid_price"]
                exit_date.loc[idx] = (
                    exit_row.name[1] if isinstance(exit_row.name, tuple) else exit_row.name
                )
            else:
                exit_prices.loc[idx] = np.nan
                exit_date.loc[idx] = np.nan

        entries["mid_price_exit"] = exit_prices
        entries["trade_exit_date"] = exit_date
        entries["pnl"] = entries["position"] * (entries["mid_price_exit"] - entry_prices)

        results = entries.set_index("data_index")[
            ["trade_exit_date", "position", "mid_price", "mid_price_exit", "pnl"]
        ].rename(columns={"mid_price": "mid_price_entry"})
        results = results.reindex(data.index)

        results = results.drop(columns=["abs_delta"], errors="ignore")
        return results


    
    @staticmethod
    def earnings_calendar_spread(data, **kwargs):
        """Implements an earnings calendar spread strategy."""

        entry_offset = kwargs.get("entry_offset", 1)
        exit_offset = kwargs.get("exit_offset", 0)
        exit_tolerance = kwargs.get("exit_tolerance", 3)
        position = kwargs.get("position", 1)  
        call_delta = kwargs.get("delta", 0.5)
        put_delta = kwargs.get("put_delta", -0.5)
        ttm_short = kwargs.get("ttm_short", 14)
        ttm_long = kwargs.get("ttm_long", 30)

        filtered_df = data.copy()[(data['bdays_to_earnings'] <= entry_offset) | (data['bdays_since_earnings'] <= exit_offset)]
        dbf_mask = (filtered_df['bdays_to_earnings'] == entry_offset)
        after_mask = (filtered_df['bdays_since_earnings'] >= exit_offset  & (filtered_df['bdays_since_earnings'] < exit_offset + exit_tolerance))
        post_data = filtered_df[after_mask].copy()

        short_call = filtered_df[(dbf_mask) & (filtered_df['ttm'] <= ttm_short) & (filtered_df['delta'] >= call_delta) & (filtered_df['call_put'] == 'c')]
        short_put = filtered_df[(dbf_mask) & (filtered_df['ttm'] <= ttm_short) & (filtered_df['delta'] <= put_delta) & (filtered_df['call_put'] == 'p')]
        long_call = filtered_df[(dbf_mask) & (filtered_df['ttm'] >= ttm_long) & (filtered_df['delta'] >= call_delta) & (filtered_df['call_put'] == 'c')]
        long_put = filtered_df[(dbf_mask) & (filtered_df['ttm'] >= ttm_long) & (filtered_df['delta'] <= put_delta) & (filtered_df['call_put'] == 'p')]

        # Take nearest delta if multiple options meet criteria
        if not short_call.empty:
            short_call = short_call.loc[short_call.groupby('trade_date')['delta'].idxmin()]
        if not short_put.empty:
            short_put = short_put.loc[short_put.groupby('trade_date')['delta'].idxmin()]
        if not long_call.empty:
            long_call = long_call.loc[long_call.groupby('trade_date')['delta'].idxmin()]
        if not long_put.empty:
            long_put = long_put.loc[long_put.groupby('trade_date')['delta'].idxmin()]

        short_call = short_call.loc[short_call.groupby('trade_date')['expiry_date'].idxmax()]
        short_put = short_put.loc[short_put.groupby('trade_date')['expiry_date'].idxmax()]
        long_call = long_call.loc[long_call.groupby('trade_date')['expiry_date'].idxmin()]
        long_put = long_put.loc[long_put.groupby('trade_date')['expiry_date'].idxmin()]

        for df, side in zip([short_call, short_put, long_call, long_put], [-1, -1, 1, 1]):
            df['position'] = side * position
            df['data_index'] = df.index

        entries = pd.concat([short_call, short_put, long_call, long_put])

        # Use a Series instead of a list, keyed by the entries index
        exit_prices = pd.Series(index=entries.index, dtype=float)
        exit_date = pd.Series(index=entries.index, dtype=int)
        entry_prices = entries['mid_price'].copy()

        for idx, row in entries.iterrows():
            candidate_exits = post_data[
                (post_data['strike'] == row['strike']) &
                (post_data['expiry_date'] == row['expiry_date']) &
                (post_data['call_put'] == row['call_put'])
            ]
            if not candidate_exits.empty:
                exit_prices.loc[idx] = candidate_exits.loc[candidate_exits['ttm'].idxmin(), 'mid_price']
                exit_date.loc[idx] = candidate_exits.loc[candidate_exits['ttm'].idxmin()].name[1]  # trade_date index
            else:
                exit_prices.loc[idx] = np.nan

        entries['mid_price_exit'] = exit_prices
        entries['data_index_exit'] = exit_date
        entries['pnl'] = entries['position'] * (entries['mid_price_exit'] - entry_prices) / entry_prices

        results = entries.set_index('data_index')[['data_index_exit', 'position', 'mid_price', 'mid_price_exit', 'pnl']]
        results = results.reindex(data.index) 
        results = results.rename(columns={'mid_price': 'mid_price_entry', 'data_index_exit': 'trade_exit_date'}) 

        return results

