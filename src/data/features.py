"""
Data featuresfunctions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.data.manager import DataManager


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
    def _compute_forward_vol_ratios(group: pd.DataFrame, **kwargs) -> pd.Series:
        group = group.sort_values("ttm")
        ttms = group["ttm"].values
        sigmas = group["implied_volatility"].values

        ratios = [np.nan]  
        for i in range(len(ttms) - 1):
            t1, t2 = ttms[i], ttms[i + 1]
            s1, s2 = sigmas[i], sigmas[i + 1]
            if t2 == t1 or s1 == 0:
                ratios.append(np.nan)
                continue
            fwd_vol = np.sqrt((s2**2 * t2 - s1**2 * t1) / (t2 - t1))
            ratios.append(fwd_vol / s1)

        mask = np.ones(len(ratios), dtype=bool)
        if kwargs.get("near_ttm") is not None:
            idx = min(max(np.searchsorted(ttms, kwargs["near_ttm"], side="right") - 1, 0), len(ratios)-1)
            mask[:] = False
            mask[idx] = True
        if kwargs.get("far_ttm") is not None:
            idx = min(max(np.searchsorted(ttms, kwargs["far_ttm"], side="left"), 0), len(ratios)-1)
            mask[:] = mask & False
            mask[idx] = True

        ratios = [ratios[i] if mask[i] else np.nan for i in range(len(ratios))]
        return pd.Series(ratios, index=group.index)


    @staticmethod
    def forward_vol_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
        return data.groupby("trade_date", group_keys=False)\
                .apply(Features._compute_forward_vol_ratios, **kwargs)\
                .rename("forward_vol_ratio")
