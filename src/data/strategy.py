"""
Definitions for trade strategies
"""
from __future__ import annotations
from typing import Callable

import pandas as pd
import numpy as np

from src.data.manager import DataManager
from src.data.utils import US_BD


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

    @staticmethod
    def three_by_ones(data, **kwargs) -> pd.DataFrame:
        """
        Implements a 3x1 ratio option strategy.
        Long 1 option near target delta, short 3 options at a strike giving ~zero cost.
        Exits after a fixed hold period (in trading days).
        """

        # --- PARAMETERS ---
        entry_ttm = kwargs.get("entry_ttm", 90)
        ttm_tol = kwargs.get("ttm_tol", 5)
        delta_long = kwargs.get("moneyness_one", 0.35)
        delta_tol = kwargs.get("moneyness_tol", 0.05)
        call_put = kwargs.get("call_put", "p")
        hold_period = kwargs.get("hold_period", 10)  # number of trading days to hold
        position = kwargs.get("position", 1)

        # --- FILTER INITIAL DATA ---
        filtered_df = data.copy()[
            (data['call_put'] == call_put)
            & (data['ttm'] <= entry_ttm + ttm_tol)
            & (data['ttm'] >= entry_ttm - ttm_tol)
        ]

        # --- FIND LONG LEG ---
        delta_lower = delta_long - delta_tol
        delta_upper = delta_long + delta_tol

        long_df = filtered_df[
            (filtered_df['delta'] > delta_lower) &
            (filtered_df['delta'] < delta_upper)
        ]

        # choose per expiration: shortest TTM, smallest delta within band
        longs = long_df.loc[long_df.groupby('exdate')['ttm'].idxmin()]
        longs = longs.loc[longs.groupby('exdate')['delta'].idxmin()]

        # --- FIND SHORT LEG ---
        # Want the short strike such that 3*long_mid ≈ short_mid (zero-cost)
        shorts = []
        for exdate, long_row in longs.groupby('exdate'):
            subset = filtered_df[filtered_df['exdate'] == exdate]
            if subset.empty:
                continue

            target_price = long_row['mid_price'].values[0] * 3
            short_candidate = subset.iloc[(subset['mid_price'] - target_price).abs().argsort()[:1]]
            shorts.append(short_candidate)

        if shorts:
            shorts = pd.concat(shorts)
        else:
            return pd.DataFrame()  # no valid shorts found

        # --- ASSIGN POSITIONS ---
        longs = longs.copy()
        shorts = shorts.copy()
        longs['position'] = +1 * position
        shorts['position'] = -3 * position

        entries = pd.concat([longs, shorts])
        entries['data_index'] = entries.index
        entry_prices = entries['mid_price'].copy()

        # --- EXIT LOGIC (BASED ON HOLD PERIOD) ---
        exit_prices = pd.Series(index=entries.index, dtype=float)
        exit_date = pd.Series(index=entries.index, dtype=int)

        for idx, row in entries.iterrows():
            trade_idx = row.name[1] if isinstance(row.name, tuple) else row.name
            candidate_exits = data[
                (data['trade_date'] > trade_idx) &
                (data['trade_date'] <= trade_idx + hold_period) &
                (data['strike'] == row['strike']) &
                (data['exdate'] == row['exdate']) &
                (data['call_put'] == row['call_put'])
            ]

            if not candidate_exits.empty:
                # Exit on the last available day within hold period
                exit_row = candidate_exits.iloc[-1]
                exit_prices.loc[idx] = exit_row['mid_price']
                exit_date.loc[idx] = exit_row.name[1] if isinstance(exit_row.name, tuple) else exit_row.name
            else:
                exit_prices.loc[idx] = np.nan
                exit_date.loc[idx] = np.nan

        # --- COMPUTE RESULTS ---
        entries['mid_price_exit'] = exit_prices
        entries['trade_exit_date'] = exit_date
        entries['pnl'] = entries['position'] * (entries['mid_price_exit'] - entry_prices)

        results = entries.set_index('data_index')[
            ['trade_exit_date', 'position', 'mid_price', 'mid_price_exit', 'pnl']
        ]
        results = results.rename(columns={'mid_price': 'mid_price_entry'})

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

