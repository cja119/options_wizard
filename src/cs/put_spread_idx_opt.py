"""
Optimized IDX Spread module
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import options_wizard as ow
import polars as pl
from dotenv import load_dotenv
from typing import List

# Load .env once and cache lookups to avoid repeated disk reads
load_dotenv()
_ENV_CACHE: dict[str, str] = {}


def _env_path(key: str) -> str:
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = os.getenv(key, "")
    return _ENV_CACHE[key]


# -----------------------------------------------------------
#                CONFIG
# -----------------------------------------------------------

OPT_RENAME_MAP = {
    "date": "trade_date",
    "exdate": "expiry_date",
    "cp_flag": "call_put",
    "strike_price": "strike",
    "impl_volatility": "bid_implied_volatility",
    "best_bid": "bid_price",
    "best_offer": "ask_price",
}

# Columns we actually need; reading only these cuts parquet scan time
OPT_BASE_COLS = list(OPT_RENAME_MAP.keys()) + [
    "delta",
    "gamma",
    "vega",
    "theta",
    "volume",
    "open_interest",
]


# ===========================================================
#     TOP-LEVEL LOGIC FUNCTIONS (IMPORTABLE ANYWHERE)
# ===========================================================


def load_index_data_logic(**kwargs) -> ow.DataType:
    tick = kwargs.get("tick", "")
    idx_opt_path = _env_path(f"{tick}".upper() + "_OPTIONS")

    if not idx_opt_path or not Path(idx_opt_path).is_file():
        return ow.DataType(pl.LazyFrame(), tick)

    df = (
        pl.scan_parquet(idx_opt_path)
        .select(OPT_BASE_COLS)
        .rename(OPT_RENAME_MAP)
        .with_columns(
            [
                pl.col("trade_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
                pl.col("expiry_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
            ]
        )
        .with_columns(
            [
                pl.col("call_put").str.to_lowercase(),
                pl.col("bid_implied_volatility").alias("ask_implied_volatility"),
            ]
        )
    )

    return ow.DataType(df, tick)


def in_universe_dummy_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    tick = kwargs.get("tick", "")
    df = data()
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    df = df.with_columns(pl.lit(True).alias("in_universe"))
    return ow.DataType(df, tick=tick)


def idx_futures_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    tick = kwargs.get("tick", "")
    fut_path = _env_path(f"{tick}".upper() + "_FUTURES")

    if not fut_path or not Path(fut_path).is_file():
        return data

    trade_date_expr = pl.coalesce(
        [
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
        ]
    )

    fut = (
        pl.scan_parquet(fut_path)
        .select(["date", "close"])
        .with_columns(
            trade_date_expr.alias("trade_date")
        )
        .rename({"close": "underlying_close"})
        .select(["trade_date", "underlying_close"])
    )

    df = data()
    if isinstance(df, pl.LazyFrame):
        joined = df.join(fut, on="trade_date", how="left")
    else:
        joined = df.join(fut.collect(), on="trade_date", how="left")
    return ow.DataType(joined, tick)


def filter_gaps_opt(data: ow.DataType, **kwargs) -> ow.DataType:
    """
    Faster gap checker:
    - avoids Python map_elements
    - computes n_missing via expected vs observed counts
    - retains days_until_last_trade for downstream filters
    """
    df = data()
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    df = df.with_columns(pl.col("trade_date").cast(pl.Date))

    keys: List[str] = ["call_put", "strike", "expiry_date"]

    calendar = df.select("trade_date").unique().sort("trade_date").with_row_index("day_idx")

    bounds = (
        df.group_by(keys)
        .agg(
            [
                pl.col("trade_date").min().alias("start"),
                pl.col("trade_date").max().alias("end"),
                pl.col("trade_date").n_unique().alias("present_count"),
            ]
        )
        .join(calendar.rename({"trade_date": "start"}), on="start", how="left")
        .rename({"day_idx": "start_idx"})
        .join(calendar.rename({"trade_date": "end"}), on="end", how="left")
        .rename({"day_idx": "end_idx"})
        .with_columns((pl.col("end_idx") - pl.col("start_idx") + 1).alias("expected_count"))
        .with_columns((pl.col("expected_count") - pl.col("present_count")).alias("n_missing"))
    )

    df2 = (
        df.join(bounds.select(keys + ["end", "n_missing"]), on=keys, how="left")
        .with_columns((pl.col("end") - pl.col("trade_date")).alias("days_until_last_trade"))
    )

    return ow.DataType(df2, kwargs.get("tick", ""))


# Simple wrappers kept local to avoid imports from other modules
def perc_spread_opt(data: ow.DataType, **kwargs) -> ow.DataType:
    df = data()
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    df = df.with_columns(
        (
            (pl.col("ask_price") - pl.col("bid_price"))
            / ((pl.col("ask_price") + pl.col("bid_price")) / 2)
        ).alias("perc_spread")
    )
    return ow.DataType(df, kwargs.get("tick", ""))


def ttms_opt(data: ow.DataType, **kwargs) -> ow.DataType:
    df = data()
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    df = df.with_columns(
        (
            (pl.col("expiry_date").cast(pl.Date) - pl.col("trade_date").cast(pl.Date)).dt.total_days()
        ).alias("ttm")
    )
    return ow.DataType(df, kwargs.get("tick", ""))


def log_moneyness_opt(data: ow.DataType, **kwargs) -> ow.DataType:
    df = data()
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    df = df.with_columns((pl.col("strike") / pl.col("underlying_close")).log().alias("log_moneyness"))
    return ow.DataType(df, kwargs.get("tick", ""))


def filter_out_opt(data: ow.DataType, **kwargs) -> ow.DataType:
    import operator as op
    from functools import reduce

    keep = kwargs.get("keep_val", "")
    col = kwargs.get("keep_col", "")
    oper = kwargs.get("keep_oper", "")
    tick = kwargs.get("tick", "")

    if isinstance(keep, str):
        keep = [keep]
    if isinstance(col, str):
        col = [col]
    if not isinstance(oper, list):
        oper = [oper]

    df = data()
    df = df.lazy() if isinstance(df, pl.DataFrame) else df
    exprs = [o(pl.col(c), d) for d, c, o in zip(keep, col, oper)]
    df = df.filter(reduce(op.and_, exprs)) if exprs else df
    return ow.DataType(df, tick)


# ===========================================================
#     PIPELINE REGISTRATION (BOTTOM OF FILE)
# ===========================================================


def add_idx_spread_methods_opt(pipeline: ow.Pipeline, kwargs) -> None:
    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)
    from .put_spread import options_entry, options_trade

    # -----------------------
    # LOAD
    # -----------------------
    @ow.wrap_fn(ow.FuncType.LOAD)
    def load_index_data(**fn_kwargs) -> ow.DataType:
        return load_index_data_logic(**fn_kwargs)

    # -----------------------
    # DATA
    # -----------------------
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def ttms_wrapped(data: ow.DataType, **fn_kwargs):
        return ttms_opt(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def filter_gaps_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return filter_gaps_opt(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_index_data])
    def filter_out_wrapped(data: ow.DataType, **fn_kwargs):
        return filter_out_opt(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def idx_futures_wrapped(data: ow.DataType, **fn_kwargs):
        return idx_futures_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def in_universe_wrapped(data: ow.DataType, **fn_kwargs):
        return in_universe_dummy_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[ttms_wrapped, filter_out_wrapped, idx_futures_wrapped])
    def perc_spread_wrapped(data: ow.DataType, **fn_kwargs):
        return perc_spread_opt(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[filter_gaps_wrapped, in_universe_wrapped])
    def log_moneyness_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return log_moneyness_opt(data, **fn_kwargs)

    @ow.wrap_fn(
        ow.FuncType.DATA,
        depends_on=[load_index_data, ttms_wrapped, filter_out_wrapped, perc_spread_wrapped, idx_futures_wrapped],
    )
    def options_entry_wrapped(data: ow.DataType, **fn_kwargs):
        return options_entry(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[options_entry_wrapped])
    def options_trade_wrapped(data: ow.DataType, **fn_kwargs):
        return options_trade(data, **fn_kwargs)

    return None


__all__ = [
    "load_index_data_logic",
    "in_universe_dummy_logic",
    "idx_futures_logic",
    "add_idx_spread_methods_opt",
]
