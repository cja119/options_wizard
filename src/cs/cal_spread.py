"""
Calendar spread strategy based on earnings report dates
"""

from typing import Tuple, List, Dict, override
from functools import partial

import options_wizard as ow


def add_cal_spread_methods(pipeline: ow.Pipeline, kwargs) -> None:

    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)

    @ow.wrap_fn(ow.FuncType.LOAD)
    def load_data(kwargs, tick: str) -> ow.BaseType:
        from .cal_spread import load_data

        return load_data(tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data])
    def in_universe(data: ow.DataType, kwargs, tick: str) -> ow.DataType:
        from .put_spread import in_universe

        return in_universe(data, tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[in_universe])
    def filter_out(data: ow.DataType, kwargs, tick: str) -> ow.DataType:
        from .cal_spread import filter_out

        return filter_out(data, tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[filter_out])
    def ttms(data: ow.DataType, kwargs, tick: str) -> ow.DataType:
        from .cal_spread import ttms

        return ttms(data, tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[ttms])
    def underlying_close(data: ow.DataType, kwargs, tick: str) -> ow.DataType:
        from .cal_spread import underlying_close

        return underlying_close(data, tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[underlying_close])
    def scale_splits(data: ow.DataType, kwargs, tick: str) -> ow.DataType:
        from .cal_spread import scale_splits

        return scale_splits(data, tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.OUTPUT, depends_on=[scale_splits])
    def realised_vol(
        data: ow.DataType, output: ow.OutputType, kwargs, tick: str
    ) -> ow.OutputType:
        from .cal_spread import realised_vol

        return realised_vol(data, output, tick, **kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[scale_splits])
    def earnigns_dates(data: ow.DataType, kwargs, tick: str) -> ow.DataType:
        return ow.DataType()  # Placeholder for actual implementation

    @ow.wrap_fn(ow.FuncType.OUTPUT, depends_on=[earnigns_dates])
    def days_to_earnings(
        data: ow.DataType, output: ow.OutputType, kwargs, tick: str
    ) -> ow.OutputType:
        return ow.OutputType()  # Placeholder for actual implementation

    @ow.wrap_fn(ow.FuncType.OUTPUT, depends_on=[scale_splits])
    def days_since_earnings(
        data: ow.DataType, output: ow.OutputType, kwargs, tick: str
    ) -> ow.OutputType:
        return ow.OutputType()  # Placeholder for actual implementation

    @ow.wrap_fn(ow.FuncType.OUTPUT, depends_on=[scale_splits])
    def term_structure_slope(
        data: ow.DataType, output: ow.OutputType, kwargs, tick: str
    ) -> ow.OutputType:
        return ow.OutputType()  # Placeholder for actual implementation

    @ow.wrap_fn(ow.FuncType.OUTPUT, depends_on=[scale_splits])
    def average_implied_vol(
        data: ow.DataType, output: ow.OutputType, kwargs, tick: str
    ) -> ow.OutputType:
        return ow.OutputType()  # Placeholder for actual implementation

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[term_structure_slope])
    def calendar_spread(data: ow.DataType, kwargs, tick: str) -> ow.StratType:
        return ow.StratType()  # Placeholder for actual implementation
