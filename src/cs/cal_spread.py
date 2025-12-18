"""
IDX Spread module — clean structure for importability
"""

import options_wizard as ow
import polars as pl
from pathlib import Path
import os
from dotenv import load_dotenv
from functools import partial
import time


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
#                   Function Definitions
# ===========================================================.


def earnings_dates(data: ow.DataType, **kwargs) -> ow.DataType:
    import os
    import polars as pl
    import pandas as pd

    tick = kwargs.get("tick", "")
    exchange = kwargs.get("exchange", ow.Exchange.NASDAQ)

    df_raw = data()
    df = df_raw.lazy() if isinstance(df_raw, pl.DataFrame) else df_raw

    stats = df.select(
        pl.col("trade_date").min().alias("min_td"),
        pl.col("trade_date").max().alias("max_td"),
    ).collect()
    min_td, max_td = stats["min_td"][0], stats["max_td"][0]

    cal = ow.market_dates(
        ow.DateObj.from_datetime(min_td),
        ow.DateObj.from_datetime(max_td),
        exchange=exchange,
    )
    cal_lf = (
        pl.DataFrame({"trade_date": [d.to_datetime().date() for d in cal]})
        .lazy()
        .with_row_index(name="td_idx")
        .sort("trade_date")
    )

    # ---- NEW: CSV earnings dates ----
    earn_dates_path = os.getenv("EARN_DATES", "").split(os.pathsep)[0]
    if not earn_dates_path:
        raise ValueError("EARN_DATES env var not set")

    earn_dates = pd.read_csv(earn_dates_path)[tick]
    earn_dates = (
        pd.to_datetime(earn_dates, format="%Y%m%d", errors="coerce")
        .dt.date
        .dropna()
    )

    if earn_dates.empty:
        raise ValueError(f"No earnings dates found for {tick}")

    earns_lf = (
        pl.DataFrame({"earn_date": earn_dates})
        .with_columns(pl.col("earn_date").cast(pl.Date))  
        .sort("earn_date")
        .lazy()
    )

    # ---- existing logic (unchanged) ----
    floor_hits = earns_lf.join_asof(
        cal_lf, left_on="earn_date", right_on="trade_date", strategy="backward"
    ).select(["trade_date", "td_idx"])

    ceil_hits = earns_lf.join_asof(
        cal_lf, left_on="earn_date", right_on="trade_date", strategy="forward"
    ).select(["trade_date", "td_idx"])

    earnings_td = (
        pl.concat([floor_hits, ceil_hits], how="vertical")
        .drop_nulls()
        .unique()
        .sort("trade_date")
    )

    df_sorted = (
        df.with_columns(pl.col("trade_date").cast(pl.Date))
        .sort("trade_date")
        .join(cal_lf, on="trade_date", how="left")
    )

    df_for_asof = df_sorted.sort("trade_date")
    earnings_for_asof = earnings_td.sort("trade_date")

    with_next = df_for_asof.join_asof(
        earnings_for_asof, on="trade_date", strategy="forward", suffix="_next"
    )
    with_prev = with_next.join_asof(
        earnings_for_asof, on="trade_date", strategy="backward", suffix="_prev"
    )

    out = (
        with_prev.rename({"td_idx_next": "next_td_idx", "td_idx_prev": "prev_td_idx"})
        .with_columns(
            [
                (pl.col("next_td_idx") - pl.col("td_idx")).alias("days_to_next_earnings"),
                (pl.col("td_idx") - pl.col("prev_td_idx")).alias("days_since_last_earnings"),
            ]
        )
    )

    return ow.DataType(out, tick)

def is_consolidating(data: ow.DataType, **kwargs) -> ow.DataType:

    df = data()
    tick = kwargs.get("tick", "")

    df = (
        df
        # --- Ensure no nulls in close before log ---
        .with_columns(
            pl.col("underlying_close")
            .fill_null(strategy="forward")
            .alias("ucl")
        )
        # --- Log returns ---
        .with_columns(
            (pl.col("ucl").log() - pl.col("ucl").log().shift(1))
            .alias("log_return")
        )
        # --- Rolling vol windows ---
        .with_columns([
            pl.col("log_return").rolling_std(window_size=20).alias("volatility_20d"),
            pl.col("log_return").rolling_std(window_size=60).alias("volatility_60d"),
        ])
        # --- Percentile of vol20 ---
        .with_columns(
            pl.col("volatility_20d")
            .rolling_quantile(window_size=252, quantile=0.2, interpolation="nearest")
            .alias("vol20_p20")
        )
        # --- Consolidation flag ---
        .with_columns(
            (
                (pl.col("volatility_20d") < 0.5 * pl.col("volatility_60d")) &
                (pl.col("volatility_20d") < pl.col("vol20_p20"))
            )
            .alias("is_consolidating")
        )
    )

    return ow.DataType(df, tick=tick)


def rec_high_low(data: ow.DataType, **kwargs) -> ow.DataType:
    import polars as pl
    import numpy as np
    from collections import deque
    from datetime import timedelta

    frame = data()
    tick = kwargs.get("tick", "")

    # Collect LazyFrame to DataFrame for Python-side loop
    if isinstance(frame, pl.LazyFrame):
        df = frame.collect()
        return_lazy = True
    else:
        df = frame
        return_lazy = False

    # Ensure correct dtypes
    df = df.with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("ttm").cast(pl.Int64),
    )

    # ============================================================
    # 1) Build daily underlying series (one row per trade_date)
    # ============================================================
    # underlying_close may be null on some dates → fill forward/backward
    ts = (
        df
        .group_by("trade_date")
        .agg([
            pl.col("underlying_close").first().alias("underlying_close"),
            pl.col("ttm").max().alias("ttm"),    # longest window per day
        ])
        .sort("trade_date")
        .with_columns(
            pl.col("underlying_close").fill_null(strategy="forward")
                                       .fill_null(strategy="backward")
        )
    )

    # Extract arrays for sliding-window logic
    dates = ts["trade_date"].to_list()
    prices = ts["underlying_close"].to_list()
    ttms = ts["ttm"].to_list()

    n = len(ts)
    recent_high = np.empty(n)
    recent_low = np.empty(n)

    maxdq = deque()
    mindq = deque()

    # ============================================================
    # 2) Compute rolling high/low using deques (O(n))
    # ============================================================
    for i in range(n):
        ttm_i = int(ttms[i]) if ttms[i] is not None else 0
        start_date = dates[i] - timedelta(days=ttm_i)

        # Remove stale indices outside window
        while maxdq and dates[maxdq[0]] < start_date:
            maxdq.popleft()
        while mindq and dates[mindq[0]] < start_date:
            mindq.popleft()

        # Insert current index for max deque
        while maxdq and prices[maxdq[-1]] <= prices[i]:
            maxdq.pop()
        maxdq.append(i)

        # Insert current index for min deque
        while mindq and prices[mindq[-1]] >= prices[i]:
            mindq.pop()
        mindq.append(i)

        recent_high[i] = prices[maxdq[0]]
        recent_low[i] = prices[mindq[0]]

    # Add results back to the daily series
    ts2 = ts.with_columns([
        pl.Series("recent_high", recent_high),
        pl.Series("recent_low", recent_low),
        (pl.Series("recent_high", recent_high) -
         pl.Series("recent_low", recent_low)).alias("recent_high_low")
    ])

    # ============================================================
    # 3) Join daily metrics back to *all option rows*
    # ============================================================
    out_df = df.join(ts2, on="trade_date", how="left")

    if return_lazy:
        out_df = out_df.lazy()

    return ow.DataType(out_df, tick=tick)

# ===========================================================
#                   Strategy Evaluation Pipeline
# ===========================================================


def add_cal_spread_methods(pipeline: ow.Pipeline, kwargs) -> None:

    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)

    from .put_spread import (
        perc_spread,
        filter_gaps,
        filter_out,
        ttms,
        load_data,
        options_entry,
        options_trade,
        scale_splits,
        log_moneyness,
        underlying_close,
        in_universe,
    )

    @ow.wrap_fn(ow.FuncType.LOAD)
    def load_data_wrapped(**fn_kwargs) -> ow.DataType:
        return load_data(**fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def ttms_wrapped(data: ow.DataType, **fn_kwargs):
        return ttms(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[])
    def filter_gaps_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return filter_gaps(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA)
    def underlying_close_wrapped(data: ow.DataType, **fn_kwargs):
        return underlying_close(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def earnings_dates_wrapped(data: ow.DataType, **fn_kwargs):
        return earnings_dates(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def filter_out_wrapped(data: ow.DataType, **fn_kwargs):
        return filter_out(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[earnings_dates_wrapped, filter_out_wrapped])
    def is_consolidating_wrapped(data: ow.DataType, **fn_kwargs):
        return is_consolidating(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA)
    def scale_splits_wrapped(data: ow.DataType, **fn_kwargs):
        return scale_splits(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def in_universe_wrapped(data: ow.DataType, **fn_kwargs):
        return in_universe(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA)
    def rec_high_low_wrapped(data: ow.DataType, **fn_kwargs):
        return rec_high_low(data, **fn_kwargs)
    
    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def log_moneyness_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return log_moneyness(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[ttms_wrapped, filter_out_wrapped, underlying_close_wrapped])
    def perc_spread_wrapped(data: ow.DataType, **fn_kwargs):
        return perc_spread(data, **fn_kwargs)

    @ow.wrap_fn(
        ow.FuncType.DATA,
        depends_on=[
            load_data_wrapped,
            ttms_wrapped,
            filter_out_wrapped,
            perc_spread_wrapped,
            underlying_close_wrapped,
            rec_high_low_wrapped,
        ],
    )
    def options_entry_wrapped(data: ow.DataType, **fn_kwargs):
        return options_entry(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[options_entry_wrapped])
    def options_trade_wrapped(data: ow.DataType, **fn_kwargs):
        return options_trade(data, **fn_kwargs)
    
    return None
