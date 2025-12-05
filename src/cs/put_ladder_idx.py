"""
IDX Spread module — clean structure for importability
"""

import options_wizard as ow
import polars as pl
from pathlib import Path
import os
from dotenv import load_dotenv
from functools import partial


# -----------------------------------------------------------
#                CONFIG
# -----------------------------------------------------------

OPT_RENAME_MAP = {
    'date': 'trade_date',
    'exdate': 'expiry_date',
    'cp_flag': 'call_put',
    'strike_price': 'strike',
    'impl_volatility': 'bid_implied_volatility',
    'best_bid': 'bid_price',
    'best_offer': 'ask_price',
}

OPT_DROP_MAP = [
    'secid', 'symbol', 'symbol_flag', 'last_date', 'optionid', 'cfadj',
    'am_settlement', 'contract_size', 'ss_flag', 'forward_price',
    'expiry_indicator', 'root', 'suffix', 'cusip', 'ticker', 'sic',
    'index_flag', 'exchange_d', 'class', 'issue_type', 'industry_group',
    'issuer', 'div_convention', 'exercise_style', 'am_set_flag',
]

# ===========================================================
#     TOP-LEVEL LOGIC FUNCTIONS (IMPORTABLE ANYWHERE)
# ===========================================================

def load_index_data_logic(**kwargs) -> ow.DataType:
    load_dotenv()
    tick = kwargs.get("tick", "")
    idx_opt_path = os.getenv(f"{tick}".upper() + "_OPTIONS", "")

    if not idx_opt_path or not Path(idx_opt_path).is_file():
        return pl.LazyFrame()

    df = (
        pl.scan_parquet(idx_opt_path)
        .rename(OPT_RENAME_MAP)
        .drop(OPT_DROP_MAP)
        .with_columns(pl.col("bid_implied_volatility").alias("ask_implied_volatility"))
    )

    df = df.with_columns([
        pl.col("trade_date").str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
        pl.col("expiry_date").str.strptime(pl.Date, format="%d/%m/%Y", strict=False)
    ])

    df = df.with_columns(
        pl.col("call_put").str.to_lowercase()
    )

    return ow.DataType(df, tick)


def filter_out_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import filter_out as ps_filter_out
    return ps_filter_out(data, **kwargs)


def ttms_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import ttms as ps_ttms
    return ps_ttms(data, **kwargs)

def in_universe_dummy_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    tick = kwargs.get("tick", "")
    df = data()
    df = df.with_columns(pl.lit(True).alias("in_universe"))
    return ow.DataType(df, tick=tick)

def idx_futures_logic(data: ow.DataType, **kwargs) -> ow.DataType:

    load_dotenv()
    tick = kwargs.get("tick", "")
    fut_path = os.getenv(f"{tick}".upper() + "_FUTURES", "")

    if not fut_path or not Path(fut_path).is_file():
        return data

    fut = (
        pl.scan_parquet(fut_path)
        .rename({"date": "trade_date", "close": "underlying_close"})
        .select(["trade_date", "underlying_close"])
        )
    
    joined = data().join(fut, on="trade_date", how="left")
    return ow.DataType(joined, tick)


def scale_splits_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import scale_splits as ps_scale_splits
    return ps_scale_splits(data, **kwargs)


def perc_spread_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import perc_spread as ps_perc_spread
    return ps_perc_spread(data, **kwargs)


def ladder_spread_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    """
    Generate ratio spread entries per day using Polars.
    """
    import polars as pl

    lower_ttm = kwargs.get("lower_ttm", 90)
    upper_ttm = kwargs.get("upper_ttm", 150)
    delta_atm = kwargs.get("delta_atm", 0.45)
    delta_mtm = kwargs.get("delta_mtm", 0.30)
    delta_otm = kwargs.get("delta_otm", 0.15)
    atm_ratio = kwargs.get("atm_ratio", 2)
    mtm_ratio = kwargs.get("mtm_ratio", -1)
    otm_ratio = kwargs.get("otm_ratio", 2)
    hold_period = kwargs.get("hold_period", 30)
    delta_tol = kwargs.get("delta_tol", 0.02)
    call_put = kwargs.get("call_put", "p")
    tick = kwargs.get("tick", "")

    df = data()

    eligible = df.filter(
        (pl.col("call_put") == call_put)
        & (pl.col("ttm").is_between(lower_ttm, upper_ttm))
        & ((pl.col("delta").abs()) <= delta_atm + delta_tol)
        & ((pl.col("delta").abs()) >= delta_otm - delta_tol)
        & (pl.col("n_missing") == 0)
        & (pl.col("days_until_last_trade") > hold_period)
        & (pl.col("in_universe") == True)
    )

    base_cols = [
        c for c in df.collect_schema().names() if c not in ("entered", "position")
    ]

    atm = (
        eligible.filter(pl.col("delta").abs() >= delta_atm - delta_tol)
        .with_columns(pl.col("perc_spread").abs().alias("atm_rank"))
        .sort(["trade_date", "atm_rank"])
        .group_by("trade_date")
        .head(1)
        .with_columns(
            [pl.lit(True).alias("entered"), pl.lit(atm_ratio).alias("position")]
        )
        .select(base_cols + ["entered", "position"])
    )

    mtm = (
        eligible.filter(pl.col("delta").abs().is_between(delta_mtm - delta_tol, delta_mtm + delta_tol))
        .with_columns(pl.col("perc_spread").abs().alias("mtm_rank"))
        .sort(["trade_date", "mtm_rank"])
        .group_by("trade_date")
        .head(1)
        .with_columns(
            [pl.lit(True).alias("entered"), pl.lit(mtm_ratio).alias("position")]
        )
        .select(base_cols + ["entered", "position"])
    )


    otm = (
        eligible.filter(pl.col("delta").abs() < delta_otm + delta_tol)
        .with_columns(pl.col("perc_spread").abs().alias("otm_rank"))
        .sort(["trade_date", "otm_rank"])
        .group_by("trade_date")
        .head(1)
        .with_columns(
            [pl.lit(True).alias("entered"), pl.lit(otm_ratio).alias("position")]
        )
        .select(base_cols + ["entered", "position"])
    )

    entries = pl.concat([atm, mtm, otm], how="vertical")

    df_out = df.join(
        entries.lazy(),
        on=df.columns
        if isinstance(df, pl.DataFrame)
        else df.collect_schema().names(),
        how="left",
    ).with_columns(
        [pl.col("entered").fill_null(False), pl.col("position").fill_null(0)]
    )

    return ow.DataType(df_out, tick)


def fixed_hold_trade_logic(data: ow.DataType, **kwargs) -> ow.StratType:
    from .put_spread import fixed_hold_trade as ps_fixed
    return ps_fixed(data, **kwargs)

def filter_gaps_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import filter_gaps as ps_filter_gaps
    return ps_filter_gaps(data, **kwargs)

# ===========================================================
#     PIPELINE REGISTRATION (BOTTOM OF FILE)
# ===========================================================

def add_idx_ladder_methods(pipeline: ow.Pipeline, kwargs) -> None:

    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)

    # -----------------------
    # LOAD
    # -----------------------
    @ow.wrap_fn(ow.FuncType.LOAD)
    def load_index_data(**fn_kwargs) -> ow.DataType:
        return load_index_data_logic(**fn_kwargs)

    # -----------------------
    # DATA — imported logic
    # -----------------------
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def ttms(data: ow.DataType, **fn_kwargs):
        return ttms_logic(data, **fn_kwargs)
    
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def filter_gaps_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return filter_gaps_logic(data, **fn_kwargs)
        

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_index_data])
    def filter_out(data: ow.DataType, **fn_kwargs):
        return filter_out_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def idx_futures(data: ow.DataType, **fn_kwargs):
        return idx_futures_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def in_universe_dummy(data: ow.DataType, **fn_kwargs):
        return in_universe_dummy_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[ttms, filter_out, idx_futures])
    def perc_spread(data: ow.DataType, **fn_kwargs):
        return perc_spread_logic(data, **fn_kwargs)

    @ow.wrap_fn(
        ow.FuncType.DATA,
        depends_on=[load_index_data, ttms, filter_out, perc_spread, idx_futures],
    )
    def ladder_spread(data: ow.DataType, **fn_kwargs):
        return ladder_spread_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[ladder_spread])
    def fixed_hold_trade(data: ow.DataType, **fn_kwargs):
        return fixed_hold_trade_logic(data, **fn_kwargs)

    return None
