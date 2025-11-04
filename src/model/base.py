"""
Base model class
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional

import pandas as pd

from src.universe import Universe
from src.data.manager import DataManager

class ModelWrapper:
    def __init__(self, ticks: list[str] | str, model: BaseModel):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.model: BaseModel = model
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run model methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Model'

        for arg in args:
            method = getattr(self.model, arg, None)
            if not callable(method):
                raise AttributeError(f"'{arg}' is not a valid method of Model")
            methods.append(method)

        manager = self.model.data_manager
        if manager.load_lazy:
            for method in methods:
                manager.add_method(self.ticks, method, kwargs)
        else:
            for method in methods:
                manager.run_method(self.ticks, method, kwargs)

class BaseModel(ABC):
    def __init__(self, data_manager: DataManager):
        self.data_manager: DataManager = data_manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> ModelWrapper:
        return ModelWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        ModelWrapper(self.data_manager.universe.ticks, self)(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def fit(X_cols: list[str], y_col: str) -> None:
        pass
    
    @staticmethod
    @abstractmethod
    def serialise(**kwargs) -> None:
        pass

    @staticmethod
    @abstractmethod
    def fit_predict(X_cols: list[str], y_col: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass