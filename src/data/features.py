"""
Data featuresfunctions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List

import pandas as pd
import numpy as np

from src import data

if TYPE_CHECKING:
    from src.data.manager import DataManager

def dataprep(method: Callable) -> Callable:
    """Decorator to mark a feature method as a data preparation step."""
    method._is_dataprep = True
    return method

class FeaturesWrapper:
    def __init__(self, ticks: list[str] | str, features: Features):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.features: Features = features
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run features methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Features'

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

class Features:
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> FeaturesWrapper:
        return FeaturesWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        FeaturesWrapper(self.manager.universe.ticks, self)(*args, **kwargs)
        return None

    @staticmethod
    def iv_rv_ratio(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the ratio of implied volatility to realized volatility."""
        iv_rv_ratio = data['implied_volatility'] / data['realized_volatility']
        iv_rv_ratio.name = 'iv_rv_ratio'
        return iv_rv_ratio
    
    @staticmethod
    def mac_vol(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the moving average crossover of volume."""
        lookback_short = kwargs.get("lookback_short", 5)
        lookback_long = kwargs.get("lookback_long", 30)

        ema_short = (
            data.groupby('trade_date')['volume']
                .transform(lambda x: x.ewm(span=lookback_short, min_periods=1).mean())
        )
        ema_long = (
            data.groupby('trade_date')['volume']
                .transform(lambda x: x.ewm(span=lookback_long, min_periods=1).mean())
        )

        mac_vol = (ema_short - ema_long) / ema_long
        mac_vol_df = mac_vol.to_frame(name='mac_volume')  # <-- return as DataFrame
        return mac_vol_df

    @staticmethod
    def avg_vol(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the average volume per day."""
        lookback_days = kwargs.get("lookback_days", 30)
        avg_vol = (
            data.groupby('trade_date')['volume']
                .transform(lambda x: x.rolling(window=lookback_days, min_periods=1).mean())
        )
        avg_vol.name = 'avg_volume'
        return avg_vol

    @staticmethod
    def term_structure_slope(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """
        Compute the average slope of the term structure of implied volatility per day
        for options within a delta (moneyness) range.
        """

        delta_low = kwargs.get("delta_low", -0.45)
        delta_high = kwargs.get("delta_high", 0.45)
        otm_limit = kwargs.get("otm_limit", 0.05)
        near_expiry_days = kwargs.get("near_expiry_days", 14)
        far_expiry_days = kwargs.get("far_expiry_days", 30)
        
        subset = data[(data['delta'] >= delta_low) & (data['delta'] <= delta_high)].copy()
        subset = subset[(subset['delta'].abs() >= otm_limit)]
        
        def slope_for_day(df):
            near_candidates = df[df['ttm'] <= near_expiry_days]
            far_candidates = df[df['ttm'] >= far_expiry_days]

            if near_candidates.empty or far_candidates.empty:
                return np.nan

            # choose nearest to bounds
            near_row = near_candidates.loc[near_candidates['ttm'].idxmax()]
            far_row = far_candidates.loc[far_candidates['ttm'].idxmin()]
            
            # handle case where multiple rows have same ttm
            near_dupes = near_candidates[near_candidates["ttm"] == near_candidates["ttm"].max()]
            far_dupes = far_candidates[far_candidates["ttm"] == far_candidates["ttm"].min()]

            if len(near_dupes) > 1 or len(far_dupes) > 1:
                raise Exception(f"Multiple near/far expiry matches on {df.name}: "
                    f"near={near_dupes['delta']}, far={far_dupes['delta']}")

            ttm_near = float(near_row["ttm"])
            ttm_far = float(far_row["ttm"])
            iv_near = float(near_row["implied_volatility"])
            iv_far = float(far_row["implied_volatility"])

            if ttm_far == ttm_near:
                return np.nan

            slope = (iv_far - iv_near) / (ttm_far - ttm_near)

            return slope

        result = (
            subset.groupby(['trade_date', 'strike_idx', 'call_put'])
                .apply(slope_for_day)
                .groupby('trade_date')
                .mean()
                .reset_index(name='avg_slope')
        )


        slope_map = result.set_index('trade_date')['avg_slope']
        data_trade_dates = data.index.get_level_values('trade_date_idx')
        avg_slope_series = data_trade_dates.map(slope_map)

        return pd.Series(avg_slope_series.values, index=data.index, name='avg_slope')
    
    @staticmethod
    def days_to_earnings(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the number of days to the next earnings date."""
        trade_dates = data['trade_date']
        next_earnings = data['next_earnings']
        mask = next_earnings.notna()
        data.loc[mask, 'bdays_to_earnings'] = np.busday_count(
                trade_dates[mask].values.astype('datetime64[D]'),
                next_earnings[mask].values.astype('datetime64[D]')
            )
        return data['bdays_to_earnings']

    @staticmethod
    def days_since_earnings(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the number of days since the last earnings date."""
        trade_dates = data['trade_date']
        last_earnings = data['last_earnings']
        mask = last_earnings.notna()
        data.loc[mask, 'bdays_since_earnings'] = np.busday_count(
                last_earnings[mask].values.astype('datetime64[D]'),
                trade_dates[mask].values.astype('datetime64[D]')
            )
        return data['bdays_since_earnings']

    @staticmethod
    @dataprep
    def prepare_data(outputs: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Prepare data by filling missing values and sorting."""

        def combine_pnl(group: pd.DataFrame) -> pd.Series:
                       
            entry_price_magnitude = group['mid_price_entry'].abs().sum()
            entry_position = (group['position'] * group['mid_price_entry']).sum()
            exit_position = (group['position'] * group['mid_price_exit']).sum()
            
            combined_pnl = (exit_position - entry_position) / entry_price_magnitude
            average_iv_ratio = group['iv_rv_ratio'].mean()
            average_slope = group['avg_slope'].mean()
            mac_vol = group['mac_volume'].mean()

            return pd.Series({
                'combined_pnl': combined_pnl,
                'average_iv_ratio': average_iv_ratio,
                'average_slope': average_slope,
                'mac_volume': mac_vol
            })
        
        outputs = outputs.dropna()
        outputs = outputs.groupby('trade_date_idx', group_keys=False).apply(combine_pnl)
        outputs.index.name = 'trade_date_idx'

        return outputs