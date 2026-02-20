from __future__ import annotations

import logging
from typing import Callable, Any, Tuple, List, Dict
from functools import wraps
from enum import Enum
from tqdm import tqdm


from .types import (
    FuncType,
    DataType,
    OutputType,
    StratType,
    ModelType,
    SaveType,
)

FNS_SIG = List[Tuple[Callable, Dict[str, Any]]]


def wrap_fn(
    sig: FuncType,
    pipeline: Pipeline | None = None,
    kwargs: Dict[str, Any] = None,
    depends_on: List[Callable] = None,
) -> Callable:

    # --- Default Arguments --- #
    if kwargs is None:
        kwargs = {}
    if depends_on is None:
        depends_on = []

    # --- Decorator Definition --- #
    def decorator(function: Callable) -> Callable:
        function._wrap = sig

        # -- Pipeline Dependency Handling -- #
        if depends_on and not pipeline:
            raise ValueError(
                "If 'depends_on' is specified, a 'pipeline' must also be provided."
            )

        # -- Return the original function -- #
        @wraps(function)
        def wrapper(*args, **fkwargs):
            result = function(*args, **fkwargs)
            if result.isempty():
                return None
            return result

        # Mirror the signature metadata on the wrapper so pipeline checks work
        wrapper._wrap = sig  # type: ignore[attr-defined]

        # -- Pipeline Dependency Handling -- #
        if pipeline is not None:
            for dep in depends_on:
                if not pipeline.isin(dep):
                    pipeline(dep, **kwargs)
            # Register the wrapper (not the raw function) to avoid double adds
            pipeline(wrapper, **kwargs)

        return wrapper

    return decorator


class SingleTickProcessor:
    def __init__(
        self,
        tick: str,
        functions: FNS_SIG,
        saves: List[SaveFrames],
        save_type: SaveType | None = None,
        return_type: List[FuncType] | None = None,
    ) -> None:
        self._tick = tick
        self._functions = functions
        self._data: DataType = DataType(tick=tick)
        self._output: OutputType = OutputType(tick=tick)
        self._strat: StratType = StratType(tick=tick)
        self._model: ModelType = ModelType(tick=tick)
        self._saves: List[SaveFrames] = saves
        self._break: bool = False
        self._save_type: SaveType | None = save_type
        self._return_type: List[FuncType] | None = return_type

    # --- External Interface --- #
    def run(self):
        for func, kwargs in self._functions:
            kwargs = kwargs.copy()
            kwargs["tick"] = self._tick

            try:
                self._execute_function(func, kwargs)
            except Exception as e:
                logging.error(
                    "Exception in %s for tick %s: %s",
                    func.__name__,
                    self._tick,
                    e,
                    extra={"tick_name": self._tick},
                )
                self._exit()

            if self._break:
                break

        if not self._break:
            self._save(kwargs.get("suffix", ""))

        return self._return()

    # --- Internal Methods --- #
    def _exit(self) -> None:
        self._break = True

    def _execute_function(self, function: Callable, kwargs: Dict[str, Any]) -> None:

        if function._wrap == FuncType.LOAD:
            res = function(**kwargs)
            if res is not None:
                self._data += res
            else:
                self._exit()

        elif function._wrap == FuncType.DATA:
            res = function(self._data, **kwargs)
            if res is not None:
                self._data = res
            else:
                self._exit()

        elif function._wrap == FuncType.OUTPUT:
            res = function(self._data, self._output, **kwargs)
            if res is not None:
                self._output = res
            else:
                self._exit()

        elif function._wrap == FuncType.STRAT:
            res = function(self._data, **kwargs)
            if res is not None:
                self._strat = res
            else:
                self._exit()

        elif function._wrap == FuncType.MODEL:
            res = function(self._output, **kwargs)
            if res is not None:
                self._model = res
            else:
                self._exit()

    def _save(self, suffix: str) -> None:
        if self._saves and not self._save_type:
            raise ValueError(
                "Save type must be specified if saving any frames in the pipeline."
            )
        if SaveFrames.DATA in self._saves:
            self._data.save(self._save_type, suffix=suffix)
        if SaveFrames.OUTPUT in self._saves:
            self._output.save(self._save_type, suffix=suffix)
        if SaveFrames.STRAT in self._saves:
            self._strat.save(self._save_type, suffix=suffix)
        if SaveFrames.MODEL in self._saves:
            self._model.save(self._save_type, suffix=suffix)
        return None

    def _return(self) -> Any:
        # If we bailed early, signal failure so the progress tracker can count it.
        if getattr(self, "_break", False):
            return None

        ret_vals = []
        for ret_type in self._return_type:
            if ret_type == FuncType.DATA:
                ret_vals.append(self._data)
            elif ret_type == FuncType.OUTPUT:
                ret_vals.append(self._output)
            elif ret_type == FuncType.STRAT:
                ret_vals.append(self._strat)
            elif ret_type == FuncType.MODEL:
                ret_vals.append(self._model)
        if len(ret_vals) == 1:
            return ret_vals[0]
        elif len(ret_vals) > 1:
            return tuple(ret_vals)
        else:
            # Light-weight success flag so progress tracking treats the run as successful
            # without materializing large data when no explicit return_type is provided.
            return True


class SaveFrames(Enum):
    DATA = "data"
    OUTPUT = "output"
    STRAT = "strategy"
    MODEL = "model"


class Pipeline:
    def __init__(
        self,
        universe,
        saves: None | List[SaveFrames] | SaveFrames = None,
        save_type: SaveType | None = SaveType.PARQUET,
        return_type: FuncType | None | List[FuncType] = None,
    ) -> None:
        self.universe = universe
        self._functions: Dict[str, FNS_SIG] = {tick: [] for tick in universe.ticks}
        self._set_saves(saves),
        self._save_type: SaveType | None = save_type
        self._rets: Dict[str, Any] = {}
        self._return_type: List[FuncType] = (
            return_type
            if isinstance(return_type, list)
            else ([return_type] if return_type is not None else [])
        )

    # --- Run the strategy pipeline ---
    def run(self) -> None:
        self._run_pipeline()

    def __call__(self, function: Callable, **kwargs: Dict[str, Any]) -> None:
        self._signature_check(function)
        for tick in self.universe.ticks:
            self._functions[tick].append((function, kwargs))

    def isin(self, function: Callable) -> bool:
        self._signature_check(function)
        targets = {function, getattr(function, "__wrapped__", None)}
        for tick in self.universe.ticks:
            for func, _ in self._functions[tick]:
                if func in targets or getattr(func, "__wrapped__", None) in targets:
                    return True
        return False

    def outputs(
        self, type: FuncType
    ) -> List[DataType | OutputType | StratType | ModelType | None]:
        strats = []
        for tick in self.universe.ticks:
            ret = self._rets[tick]
            if type == FuncType.DATA:
                strats.append(ret if isinstance(ret, DataType) else None)
            elif type == FuncType.OUTPUT:
                strats.append(ret if isinstance(ret, OutputType) else None)
            elif type == FuncType.STRAT:
                strats.append(ret if isinstance(ret, StratType) else None)
            elif type == FuncType.MODEL:
                strats.append(ret if isinstance(ret, ModelType) else None)
        return strats

    # --- Internal Methods --- #

    def _run_pipeline(self) -> None:
        failed = 0
        for tick in self.universe.ticks:
            logging.info(f"Running strategy construction.", extra={"tick_name": tick})
            result = self._run_single(tick)
            self._rets[tick] = result

            # failure logic
            if result is None:
                failed += 1
            elif isinstance(result, tuple) and all(r is None for r in result):
                logging.warning(f"Strategy backtest failed", extra={"tick_name": tick})
                failed += 1

    def _run_single(
        self, tick: str
    ) -> None | DataType | OutputType | StratType | ModelType | Tuple[Any, ...]:
        processor = SingleTickProcessor(
            tick, self._functions[tick], self._saves, self._save_type, self._return_type
        )
        rets = processor.run()
        return rets

    def _signature_check(self, function: Callable) -> None:
        has_wrap = hasattr(function, "_wrap") or hasattr(
            getattr(function, "__wrapped__", None), "_wrap"
        )
        assert has_wrap, (
            f"Function '{function.__name__}' is missing a signature decorator. "
            "Please use one of the following decorators: "
            "@wrap(FuncType.LOAD), @wrap(FuncType.DATA), "
            "@wrap(FuncType.OUTPUT), @wrap(FuncType.STRAT), @wrap(FuncType.MODEL)."
        )

    def _set_saves(self, saves: None | List[SaveFrames] | SaveFrames) -> None:
        if saves is None:
            self._saves = []
        elif isinstance(saves, SaveFrames):
            self._saves = [saves]
        else:
            self._saves = saves
        return None
