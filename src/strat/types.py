"""
Type definitions for strategy implementation
"""

from __future__ import annotations

import os
from pathlib import Path
from enum import Enum
from typing import override 
from abc import ABC, abstractmethod

import polars as pl

SAVE_PATH: Path = Path(os.getcwd()) / "tmp"

class FuncType(Enum):
    LOAD = "load_function"
    DATA = "data_function"
    OUTPUT = "output_function"
    STRAT = "strategy_function"
    MODEL = "model_function"

class BaseType(ABC):
    def __init__(self, data: None | pl.DataFrame = None, tick: str = "") -> None:
        self._data: None | pl.DataFrame = data
        self._tick: str = tick
    
    #--- Abstract Methods --- #
    @abstractmethod
    def __add__(self, other: BaseType) -> BaseType:
        pass

    # --- External Interface --- #
    def __iadd__(self, other: BaseType) -> BaseType:
        return self.__add__(other)
    
    def __call__(self) -> None | pl.DataFrame:
        return self._data

    def save(self) -> None:
        save_path: Path = SAVE_PATH / f"{self._name}_{self._tick}.parquet"
        if save_path.parent.exists() is False:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        self._data.write_parquet(save_path)
        return None            
    
    @classmethod
    def load(cls, tick: str) -> BaseType:
        load_path: Path = SAVE_PATH / f"{cls._name}_{tick}.parquet"
        if load_path.exists() is False:
            return cls(data=None, tick=tick)
        data = pl.read_parquet(load_path)
        return cls(data=data, tick=tick)
        

class DataType(BaseType):
    _name = "data"
    
    # --- Override Methods --- #
    @override
    def __add__(self, other: DataType) -> DataType:
        assert isinstance(other, DataType), "Can only add DataType to DataType"
        self._data = other._data
        return self

class OutputType(BaseType):
    _name = "output"
    
    # --- Override Methods --- #
    @override
    def __add__(self, other: OutputType) -> OutputType:
        assert isinstance(other, OutputType), "Can only add OutputType to OutputType"
        if self._data is None:
            self._data = other._data
        else:
            self._data = pl.concat([self._data, other._data], how="diagonal")
        return self

class StratType(BaseType):
    _name = "strategy"
    
    # --- Override Methods --- #
    @override
    def __add__(self, other: StratType) -> StratType:
        assert isinstance(other, StratType), "Can only add StratType to StratType"
        if self._data is None:
            self._data = other._data
        else:
            self._data = pl.concat([self._data, other._data], how="diagonal")
        return self

class ModelType(BaseType):
    _name = "model"

    # --- Override Methods --- #
    @override
    def __add__(self, other: ModelType) -> ModelType:
        assert isinstance(other, ModelType), "Can only add ModelType to ModelType"
        if self._data is None:
            self._data = other._data
        else:
            self._data = pl.concat([self._data, other._data], how="vertical")
        return self

