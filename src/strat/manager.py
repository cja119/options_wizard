from __future__ import annotations

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
)

FNS_SIG = List[Tuple[Callable, Dict[str, Any]]]


def wrap_fn(
    sig: FuncType,
    pipeline: Pipeline | None = None,
    kwargs: Dict[str, Any] = None,
    depends_on: List[Callable] = None,
) -> Callable:
    if kwargs is None:
        kwargs = {}
    if depends_on is None:
        depends_on = []

    def decorator(function: Callable) -> Callable:
        function._wrap = sig

        # -- Pipeline Dependency Handling -- #
        if depends_on and not pipeline:
            raise ValueError(
                "If 'depends_on' is specified, a 'pipeline' must also be provided."
            )
        if pipeline is not None:
            for dep in depends_on:
                if not pipeline.isin(dep):
                    pipeline(dep, **kwargs)
            pipeline(function, **kwargs)

        # -- Return the original function -- #
        @wraps(function)
        def wrapper(*args, **fkwargs):
            result = function(*args, **fkwargs) 
            if result.isempty():
                return None
            return result
        return wrapper
    return decorator

class SingleTickProcessor:
    def __init__(self, tick: str, functions: FNS_SIG, saves: List[SaveFrames]) -> None:
        self._tick = tick
        self._functions = functions
        self._data: DataType = DataType(tick=tick)
        self._output: OutputType = OutputType(tick=tick)
        self._strat: StratType = StratType(tick=tick)
        self._model: ModelType = ModelType(tick=tick)
        self._saves: List[SaveFrames] = saves
        self._break: bool = False

    # --- External Interface --- #
    def run(self) -> None:
        for func, kwargs in self._functions:
            kwargs = kwargs.copy()
            kwargs["tick"] = self._tick
            try:
                self._execute_function(func, kwargs)
            except Exception as e:
                print(f"Error executing function '{func.__name__}' for tick '{self._tick}': {e}")
                self._exit()
            if self._break:
                break
        if not self._break:
            self._save()
        pass

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

    def _save(self) -> None:
        if SaveFrames.DATA in self._saves:
            self._data.save()
        if SaveFrames.OUTPUT in self._saves:
            self._output.save()
        if SaveFrames.STRAT in self._saves:
            self._strat.save()
        if SaveFrames.MODEL in self._saves:
            self._model.save()
        return None


class SaveFrames(Enum):
    DATA = "data"
    OUTPUT = "output"
    STRAT = "strategy"
    MODEL = "model"


class Pipeline:
    def __init__(
        self, universe, saves: None | List[SaveFrames] | SaveFrames = None
    ) -> None:
        self.universe = universe
        self._functions: Dict[str, FNS_SIG] = {tick: [] for tick in universe.ticks}
        self._set_saves(saves)

    # --- Run the strategy pipeline ---
    def run(self) -> None:
        self._run_pipeline()

    def __call__(self, function: Callable, **kwargs: Dict[str, Any]) -> None:
        self._signature_check(function)
        for tick in self.universe.ticks:
            self._functions[tick].append((function, kwargs))

    def isin(self, function: Callable) -> bool:
        self._signature_check(function)
        for tick in self.universe.ticks:
            if any(func == function for func, _ in self._functions[tick]):
                return True
        return False

    # --- Internal Methods --- #
    def _run_pipeline(self) -> None:
        for tick in tqdm(self.universe.ticks):
            self._run_single(tick)

    def _run_single(self, tick: str) -> None:
        processor = SingleTickProcessor(tick, self._functions[tick], self._saves)
        processor.run()

    def _signature_check(self, function: Callable) -> None:
        assert hasattr(function, "_wrap"), (
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
