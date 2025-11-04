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
    def butterfly_spread(data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate butterfly spread returns for all feasible trades in a DataFrame.
        Fully kwargs-compatible.
        """
        from src.strategies.butterfly_spread import butterfly_returns

        data_copy = data.copy()
        cp_flag = kwargs.get('call_put', 'c')
        ttm = kwargs.get('ttm', 30)
        hold = kwargs.get('hold_days', 15)
        name = kwargs.get('name', f'butterfly_spread_ttm{ttm}_hold{hold}')

        filtered_data = data_copy[
            (data_copy['call_put'] == cp_flag) & 
            (data_copy['ttm'] >= ttm - hold) &
            (data_copy['ttm'] <= ttm)
        ]
        res = butterfly_returns(filtered_data, **kwargs)
        res.name= name
        return res