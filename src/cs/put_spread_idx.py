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

def vix_term_structure(data: ow.DataType, **kwargs) -> ow.DataType:
    
    from dotenv import load_dotenv
    import os
    import sys

    load_dotenv()
    path = os.getenv("VIX_FUTURES", "")
    tick = kwargs.get("tick", "")

    if not path or not Path(path).is_file():
        raise FileNotFoundError("VIX futures data not found at specified path.")
    
    vix_fut = (
            pl.scan_parquet(path)
            .select(["trade_date", "UX1 Index", "UX3 Index", "UX6 Index"])
            .rename({"UX1 Index": "1m_vix_fut", "UX3 Index": "3m_vix_fut", "UX6 Index": "6m_vix_fut"})
        )
    vix_fut = vix_fut.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, format="%d/%m/%Y", strict=False)
        )

    vix_fut = vix_fut.with_columns(
            ((pl.col("6m_vix_fut") - pl.col("1m_vix_fut")) / 5).alias("grad")
        )
    vix_fut = vix_fut.with_columns(
            ((pl.col("1m_vix_fut") + pl.col("6m_vix_fut") * (2/3) - pl.col("3m_vix_fut") * (1 - (2/3))) / (0.5*(2*3 + 2**2))).alias("curvature")
    )

    vix_fut = vix_fut.collect()

    joined = data().join(vix_fut, on="trade_date", how="left")

    return ow.DataType(joined, tick=tick)


def scale_splits_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import scale_splits as ps_scale_splits
    return ps_scale_splits(data, **kwargs)


def perc_spread_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import perc_spread as ps_perc_spread
    return ps_perc_spread(data, **kwargs)


def ratio_spread_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import ratio_spread as ps_ratio_spread
    return ps_ratio_spread(data, **kwargs)


def fixed_hold_trade_logic(data: ow.DataType, **kwargs) -> ow.StratType:
    from .put_spread import fixed_hold_trade as ps_fixed
    return ps_fixed(data, **kwargs)

def filter_gaps_logic(data: ow.DataType, **kwargs) -> ow.DataType:
    from .put_spread import filter_gaps as ps_filter_gaps
    return ps_filter_gaps(data, **kwargs)

# ===========================================================
#     PIPELINE REGISTRATION (BOTTOM OF FILE)
# ===========================================================

def add_idx_spread_methods(pipeline: ow.Pipeline, kwargs) -> None:

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
    def ratio_spread(data: ow.DataType, **fn_kwargs):
        return ratio_spread_logic(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[ratio_spread])
    def fixed_hold_trade(data: ow.DataType, **fn_kwargs):
        return fixed_hold_trade_logic(data, **fn_kwargs)

    return None
