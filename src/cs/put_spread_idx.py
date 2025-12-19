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

# ===========================================================
#     PIPELINE REGISTRATION (BOTTOM OF FILE)
# ===========================================================

def add_idx_spread_methods(pipeline: ow.Pipeline, kwargs) -> None:

    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)
    from .put_spread import (
        options_entry,
        perc_spread,
        filter_gaps,
        filter_out,
        options_trade,
        log_moneyness,
        ttms,
    )

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
    def ttms_wrapped(data: ow.DataType, **fn_kwargs):
        return ttms(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def filter_gaps_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return filter_gaps(data, **fn_kwargs)
        
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_index_data])
    def filter_out_wrapped(data: ow.DataType, **fn_kwargs):
        return filter_out(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def idx_futures_wrapped(data: ow.DataType, **fn_kwargs):
        return idx_futures_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def in_universe_wrapped(data: ow.DataType, **fn_kwargs):
        return in_universe_dummy_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[ttms_wrapped, filter_out_wrapped, idx_futures_wrapped])
    def perc_spread_wrapped(data: ow.DataType, **fn_kwargs):
        return perc_spread(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[filter_gaps_wrapped, in_universe_wrapped])
    def log_moneyness_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return log_moneyness(data, **fn_kwargs)

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
