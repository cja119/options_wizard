"""
Type definitions for strategy implementation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from enum import Enum
from typing import Deque, List, override, Optional, Dict, Any
from abc import ABC, abstractmethod
import structlog

import polars as pl

from options_wizard.data.trade import *
from options_wizard.data.date import *
from options_wizard.data.contract import *

# Save artifacts relative to the current working directory (where the script is run)
SAVE_PATH: Path = Path.cwd() / "tmp"
SAVE_PATH.mkdir(exist_ok=True, parents=True)
logger = structlog.get_logger(__name__)


class SaveType(str, Enum):
    PICKLE: str = "pickle"
    PARQUET: str = "parquet"


class FuncType(str, Enum):
    LOAD = "load_function"
    DATA = "data_function"
    OUTPUT = "output_function"
    STRAT = "strategy_function"
    MODEL = "model_function"


class BaseType(ABC):
    def __init__(
        self, data: None | pl.DataFrame | List | Deque = None, tick: str = ""
    ) -> None:
        if isinstance(data, (list, Deque)):
            data = self._from_list(data)
        self._data: None | pl.DataFrame | pl.LazyFrame = data
        self._tick: str = tick

    # --- Abstract Methods --- #
    @abstractmethod
    def __add__(self, other: BaseType) -> BaseType:
        pass

    # --- External Interface --- #
    def __iadd__(self, other: BaseType) -> BaseType:
        return self.__add__(other)

    def __call__(self) -> None | pl.DataFrame:
        return self._data

    def save(self, save_type: SaveType = SaveType.PARQUET, suffix: str = "") -> None:
        if save_type == SaveType.PARQUET:
            save_path: Path = SAVE_PATH / f"{self._name}_{self._tick}_{suffix}.parquet"
            if save_path.parent.exists() is False:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            self._data.write_parquet(save_path)
        elif save_type == SaveType.PICKLE:
            import dill

            save_path: Path = SAVE_PATH / f"{self._name}_{self._tick}_{suffix}.pkl"
            if save_path.parent.exists() is False:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            dill.dump(self, open(save_path, "wb"))
        return None

    def isempty(self) -> bool:
        if self._data is None:
            return True
        if isinstance(self._data, pl.DataFrame):
            return self._data.is_empty()
        if isinstance(self._data, pl.LazyFrame):
            return False
        if isinstance(self._data, (list, Deque)):
            return len(self._data) == 0
        else:
            return False

    @classmethod
    def load(
        cls, tick: str, save_type: SaveType = SaveType.PARQUET, suffix: str = ""
    ) -> BaseType:
        if save_type == SaveType.PARQUET:
            load_path: Path = SAVE_PATH / f"{cls._name}_{tick}_{suffix}.parquet"
            if load_path.exists() is False:
                return cls(data=None, tick=tick)
            data = pl.read_parquet(load_path)
            return cls(data=data, tick=tick)
        elif save_type == SaveType.PICKLE:
            import dill

            load_path: Path = SAVE_PATH / f"{cls._name}_{tick}_{suffix}.pkl"
            if load_path.exists() is False:
                return cls(data=None, tick=tick)
            return dill.load(open(load_path, "rb"))

    def _from_list(self, data: List | Deque) -> pl.DataFrame:
        raise NotImplementedError("Subclasses must implement _from_list method")


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
    dc_type = EntryData

    @override
    def save(self, save_type: SaveType = SaveType.PARQUET, suffix: str = "") -> None:
        if save_type == SaveType.PARQUET:
            save_path = SAVE_PATH / f"{self._name}_{self._tick}_{suffix}.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if self._data is None:
                pl.DataFrame({"entry": []}).write_parquet(save_path)
                return

            rows = [json.dumps(row) for row in self._data]
            df = pl.DataFrame({"entry": rows})
            df.write_parquet(save_path)
        elif save_type == SaveType.PICKLE:
            import dill

            save_path: Path = SAVE_PATH / f"{self._name}_{self._tick}_{suffix}.pkl"
            if save_path.parent.exists() is False:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            dill.dump(self, open(save_path, "wb"))

    @override
    def __add__(self, other: StratType) -> StratType:
        assert isinstance(other, StratType)
        if self._data is None:
            self._data = other._data
        else:
            self._data = pl.concat([self._data, other._data], how="vertical")
        return self
    
    def __iter__(self):
        return iter(self._data)

    @override
    @classmethod
    def load(
        cls, tick: str, save_type: SaveType = SaveType.PARQUET, suffix: str = "",
        wrapper: Optional[Callable] = None, overrides: Optional[Dict[str, Any]]= None
    ) -> StratType:
        if save_type == SaveType.PARQUET:
            path = SAVE_PATH / f"{cls._name}_{tick}_{suffix}.parquet"
            if not path.exists():
                return cls(data=None, tick=tick)

            df = pl.read_parquet(path)
            entries = df["entry"].to_list()

            objs = []
            for js in entries:
                raw_dict = json.loads(js)
                obj = cls.dc_type.from_dict(raw_dict)
                objs.append(obj)

            cls = cls(data=objs, tick=tick)

        elif save_type == SaveType.PICKLE:
            import dill

            if SAVE_PATH.exists() is False:
                return cls(data=None, tick=tick)
            
            cls =  dill.load(open(SAVE_PATH / f"{cls._name}_{tick}_{suffix}.pkl", "rb"))

        if wrapper is not None:
            cls._data = cls.reconstruct(wrapper, overrides)

        return cls
        
    def reconstruct(self, wrapper=None, overrides: Optional[Dict[str, Any]]=None):
        if self._data is None:
            return []
        out = [self.dc_type.from_dict(obj) for obj in self._data]
        if wrapper:
            out = [wrapper(x) for x in out]

        if overrides:
            for item in out:
                for key, value in overrides.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                    else:
                        logger.warning(
                            "Override ignored; item does not have attribute",
                            item=str(item),
                            attribute=key,
                        )
        return out

    @staticmethod
    def _from_list(data):
        return [obj.to_dict() for obj in data]


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
