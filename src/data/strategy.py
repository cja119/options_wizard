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
    def __init__(self, manager):
        super().__init__(manager)
        return None
    
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> StrategyWrapper:
        return StrategyWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        StrategyWrapper(self.manager.universe.ticks, self)(*args, **kwargs)

    
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

