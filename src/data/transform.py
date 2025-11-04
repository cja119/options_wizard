"""
Data transformation functions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.data.manager import DataManager


class TransformWrapper:
    def __init__(self, ticks: list[str] | str, transform: Transformer):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.transform: Transformer = transform
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run transformation methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Transformer'

        for arg in args:
            method = getattr(self.transform, arg, None)
            if not callable(method):
                raise AttributeError(f"'{arg}' is not a valid method of Transformer")
            methods.append(method)

        manager = self.transform.manager
        if manager.load_lazy:
            for method in methods:
                manager.add_method(self.ticks, method, kwargs)
        else:
            for method in methods:
                manager.run_method(self.ticks, method, kwargs)

class Transformer:
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> TransformWrapper:
        return TransformWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        TransformWrapper(self.manager.universe.ticks, self)(*args, **kwargs)
        return None

    @staticmethod
    def drop_stale_options(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Drops options that are stale (e.g., no volume or open interest)"""
        volume_threshold = kwargs.get("volume_threshold", 0)
        oi_threshold = kwargs.get("open_interest_threshold", 0)
        data = data[data.isna().sum(axis=1) == 0]
        if "volume" in data.columns:
            data = data[data["volume"] > volume_threshold]
        if "open_interest" in data.columns:
            data = data[data["open_interest"] > oi_threshold]
        if "bid_price" in data.columns:
            data = data[data["bid_price"] > 0.01]
        if "ask_price" in data.columns:
            data = data[data["ask_price"] > 0.01]
        return data
    
    @staticmethod
    def scale_by_splits(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Scales option prices for stock splits"""
        splits = kwargs.get("splits", {})
        for date, ratio in splits.items():
            data.loc[
                pd.to_datetime(data['trade_date']) < pd.Timestamp(date),
                ['strike', 'bid_price', 'ask_price', 'last_trade_price']
            ] /= ratio
        return data
    
    @staticmethod
    def train_test_split(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Splits data into training and testing sets based on a specified test size."""
        test_size = kwargs.get("test_size", 0.2)
        drop_out_of_sample = kwargs.get("drop_out_of_sample", False)
        data = data.sort_values('trade_date')
        unique_dates = data['trade_date'].drop_duplicates().sort_values()
        split_index = int(len(unique_dates) * (1 - test_size))
        split_date = unique_dates.iloc[split_index]
        if drop_out_of_sample:
            data = data[data['trade_date'] < split_date]
        else:
            data['set'] = ['train' if date < split_date else 'test' for date in data['trade_date']]
        return data

    @staticmethod
    def add_ttm(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Adds time to maturity (TTM) column to the data."""
        data['ttm'] = (pd.to_datetime(data['expiry_date']) - pd.to_datetime(data['trade_date'])).dt.days
        return data
    
    @staticmethod
    def filter_ttms(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Filters options based on minimum and maximum time to maturity (TTM)."""
        min_ttm = kwargs.get("min_ttm", 0)
        max_ttm = kwargs.get("max_ttm", float('inf'))
        if 'ttm' not in data.columns:
            data = Transformer.add_ttm(data, **kwargs)
        data = data[(data['ttm'] >= min_ttm) & (data['ttm'] <= max_ttm)]
        return data
    
    @staticmethod
    def implied_volatility(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Calculates the implied volatility from the midpoint of bis and ask implied vols"""
        data['implied_volatility'] = (data['bid_implied_volatility'] + data['ask_implied_volatility']) / 2
        return data
    
    @staticmethod
    def mid_price(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Calculates the mid price from the bid and ask prices."""
        data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2
        return data
    
    @staticmethod
    def safe_mean(series: pd.Series) -> float:
        """Compute mean, return NaN if all values are NaN."""
        if series.dropna().empty:
            return np.nan
        return series.mean()

    
    @staticmethod
    def safe_sum(series: pd.Series) -> float:
        """Compute sum, return NaN if all values are NaN."""
        if series.dropna().empty:
            return np.nan
        return series.sum()

    @staticmethod
    def prepare_data(outputs: pd.DataFrame, data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Prepares data by computing specified features and target variables."""
        features: List[Tuple[str, Dict[str, Any]]] = kwargs.get('features', [])
        target: Tuple[str, Dict[str, Any]] | None = kwargs.get('target', None)
        outputs['trade_date'] = data.loc[data.index, 'trade_date']
        filters: Dict[str, Tuple[Any, ...]] = kwargs.get('filters', {})
        shifts : Dict[str, int] = kwargs.get('shifts', {})

        # Map aggregation names to functions
        agg_map = {
            'mean': Transformer.safe_mean,
            'sum': Transformer.safe_sum
        }

        agg_dict = {}
        for feature_name, feature_kwargs in features:
            if feature_name not in outputs.columns:
                raise ValueError(f"Feature '{feature_name}' not found in outputs.")
            how = feature_kwargs.get('how', 'mean')
            if how not in agg_map:
                raise ValueError(f"Aggregation '{how}' not supported.")
            agg_dict[feature_name] = agg_map[how]

        if target is not None:
            target_name, target_kwargs = target
            if target_name not in outputs.columns:
                raise ValueError(f"Target '{target_name}' not found in outputs.")
            how = target_kwargs.get('how', 'mean')
            if how not in agg_map:
                raise ValueError(f"Aggregation '{how}' not supported.")
            agg_dict[target_name] = agg_map[how]

        # Group by trade_date and apply safe aggregations
        if shifts:
            for feature_name, shift in shifts.items():
                if feature_name not in outputs.columns:
                    raise ValueError(f"Feature '{feature_name}' not found in outputs.")
                outputs[feature_name] = outputs.groupby('trade_date')[feature_name].shift(shift)


        outputs = outputs.groupby('trade_date').agg(agg_dict).reset_index()

        if filters:
            mask = pd.Series(False, index=outputs.index)
            for feature_name, filter_tuple in filters.items():
                if feature_name not in outputs.columns and feature_name not in data.columns:
                    raise ValueError(f"Feature '{feature_name}' not found in data columns.")
                
                for condition in filter_tuple:
                    if condition == 'null':
                        mask |= outputs[feature_name].isna()
                    else:
                        mask |= outputs[feature_name] == condition

            # Keep only rows NOT masked
            outputs = outputs[~mask].copy()

        return outputs