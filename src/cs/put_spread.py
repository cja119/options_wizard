"""
Methods for put spread strategy
"""

from typing import Tuple, List, Dict, override
import options_wizard as ow
from functools import partial
import numpy as np


# ============================================================
#   TOP-LEVEL LOGIC FUNCTIONS (IMPORTABLE, UNDECORATED)
# ============================================================


def load_data(**kwargs) -> ow.DataType:
    """Loads in data for the tick"""
    from dotenv import load_dotenv
    import os
    import polars as pl

    max_date = kwargs.get("max_date", None)

    load_dotenv()
    tick = kwargs.get("tick", "")
    tick_path = os.getenv("TICK_PATH", "")
    try:
        data = pl.scan_parquet(os.path.join(tick_path, f"{tick}.parquet"))
    except FileNotFoundError:
        # Missing tick data: return empty so the pipeline can short-circuit cleanly
        return ow.DataType(None, tick)

    if max_date is not None:
        data = data.filter(pl.col("trade_date") <= max_date)

    return ow.DataType(data, tick)


def in_universe(data: ow.DataType, **kwargs) -> ow.DataType:
    """Marks whether each row is in the universe."""
    from dotenv import load_dotenv
    import os
    import polars as pl
    import pandas as pd
    from pathlib import Path

    load_dotenv()
    n_const = kwargs.get("n_const", 50)
    tick = kwargs.get("tick", "")

    na_vals = ["NaN", "nan", "#N/A", "#N/A N/A", "#N/A Invalid Security"]
    cache_dir = Path(os.getcwd()) / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_parquet = cache_dir / "universe_frame.parquet"
    cache_pickle = cache_dir / "universe_frame.pkl"

    universe_frame = None

    if cache_parquet.exists():
        universe_frame = pd.read_parquet(cache_parquet)
    elif cache_pickle.exists():
        universe_frame = pd.read_pickle(cache_pickle)
    else:
        index_df = pd.read_csv(os.getenv("INDEX_CONSTITUENTS_PATH"), na_values=na_vals)

        mktcap_df = pd.read_csv(os.getenv("MKT_CAP_PATH"), na_values=na_vals)

        index_df = index_df.drop(columns=index_df.columns[0])
        mktcap_df = mktcap_df.drop(columns=mktcap_df.columns[0])

        index_df = index_df.rename(columns={index_df.columns[0]: "date"})
        mktcap_df = mktcap_df.rename(columns={mktcap_df.columns[0]: "date"})

        index_df["date"] = pd.to_datetime(index_df["date"], dayfirst=True)
        mktcap_df["date"] = pd.to_datetime(mktcap_df["date"], dayfirst=True)

        index_df = index_df.set_index("date")
        mktcap_df = mktcap_df.set_index("date")

        index_long = index_df.stack().reset_index()
        index_long = index_long.rename(columns={"level_1": "col_idx", 0: "ticker"})

        mktcap_long = mktcap_df.stack().reset_index()
        mktcap_long = mktcap_long.rename(columns={"level_1": "col_idx", 0: "mktcap"})

        merged = pd.merge(index_long, mktcap_long, on=["date", "col_idx"], how="outer")
        panel = merged.pivot(index="date", columns="ticker", values="mktcap")

        universe_frame = panel.loc[:, ~panel.columns.str.match(r"^\d")]
        try:
            universe_frame.to_parquet(cache_parquet)
        except Exception:
            cache_pickle.unlink(missing_ok=True)
            universe_frame.to_pickle(cache_pickle)

    if not isinstance(universe_frame.index, pd.DatetimeIndex):
        universe_frame.index = pd.to_datetime(universe_frame.index)
    universe_frame.index = pd.to_datetime(universe_frame.index).tz_localize(None)

    df = data()
    if isinstance(df, pl.LazyFrame):
        stats = df.select(
            pl.col("trade_date").min().alias("min_td"),
            pl.col("trade_date").max().alias("max_td"),
        ).collect()
        start_date = stats["min_td"][0]
        end_date = stats["max_td"][0]
    else:
        start_date = df.select(pl.col("trade_date").min()).item()
        end_date = df.select(pl.col("trade_date").max()).item()

    # If there is no data for this ticker in the requested window, mark it
    # as out-of-universe so downstream filters drop it cleanly.
    if start_date is None or end_date is None:
        df_out = df.with_columns(pl.lit(False).alias("in_universe"))
        return ow.DataType(df_out, tick)

    universe_frame = universe_frame.sort_index()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    daily_index = pd.date_range(start_ts, end_ts, freq="D")

    # Track membership separately so that once a ticker drops out of the index
    # its mkt cap is not forward-filled into future dates.
    membership = (
        universe_frame.notna()
        .reindex(daily_index)
        .astype("boolean")
        .ffill()
        .fillna(False)
        .astype(bool)
    )

    daily_panel = universe_frame.reindex(daily_index).ffill().where(membership)

    top_n_per_date = daily_panel.apply(
        lambda row: row.dropna().nlargest(n_const).index.tolist(), axis=1
    )
    in_universe_mask = top_n_per_date.apply(lambda tickers: tick in tickers)

    universe_map = pl.from_pandas(
        in_universe_mask.rename("in_universe")
        .reset_index()
        .rename(columns={"index": "trade_date"})
    ).with_columns(pl.col("trade_date").dt.date())

    df_out = df.join(universe_map.lazy(), on="trade_date", how="left").with_columns(
        pl.col("in_universe").fill_null(False)
    )

    return ow.DataType(df_out, tick)


def perc_spread(data: ow.DataType, **kwargs) -> ow.DataType:
    """Adds percentage spread column to data."""
    import polars as pl

    df = data().with_columns(
        (
            (pl.col("ask_price") - pl.col("bid_price"))
            / ((pl.col("ask_price") + pl.col("bid_price")) / 2)
        ).alias("perc_spread")
    )
    return ow.DataType(df, kwargs.get("tick", ""))


def filter_gaps(data: ow.DataType, **kwargs) -> ow.DataType:
    import polars as pl

    df = data()
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    df = df.with_columns(pl.col("trade_date").cast(pl.Date))
    keys = ["call_put", "strike", "expiry_date"]

    calendar = (
        df.select("trade_date").unique().sort("trade_date").with_row_index("day_idx")
    )
    cal_dates = calendar["trade_date"].to_list()

    contract_bounds = df.group_by(keys).agg(
        [
            pl.col("trade_date").min().alias("start"),
            pl.col("trade_date").max().alias("end"),
            pl.col("trade_date").alias("present_dates"),
        ]
    )

    bounds = (
        contract_bounds.join(
            calendar.rename({"trade_date": "start"}), on="start", how="left"
        )
        .rename({"day_idx": "start_idx"})
        .join(calendar.rename({"trade_date": "end"}), on="end", how="left")
        .rename({"day_idx": "end_idx"})
    )

    def compute_missing(s):
        p = set(s["present_dates"])
        if s["start_idx"] is None or s["end_idx"] is None:
            return []
        expected = cal_dates[s["start_idx"] : s["end_idx"] + 1]
        return [d for d in expected if d not in p]

    bounds = bounds.with_columns(
        pl.struct(["present_dates", "start_idx", "end_idx"])
        .map_elements(compute_missing)
        .alias("missing_days")
    ).with_columns(pl.col("missing_days").list.len().alias("n_missing"))

    bounds = bounds.with_columns((pl.col("end") - pl.col("start")).alias("_tmp"))

    df2 = df.join(
        bounds.select(keys + ["end", "missing_days", "n_missing"]),
        on=keys,
        how="left",
    ).with_columns(
        (pl.col("end") - pl.col("trade_date")).alias("days_until_last_trade")
    )

    return ow.DataType(df2.lazy(), kwargs.get("tick", ""))


def ttms(data: ow.DataType, **kwargs) -> ow.DataType:
    """Adds time to maturity column to data."""
    import polars as pl

    df = data().with_columns(
        (
            (
                pl.col("expiry_date").cast(pl.Date) - pl.col("trade_date").cast(pl.Date)
            ).dt.total_days()
        ).alias("ttm")
    )
    return ow.DataType(df, kwargs.get("tick", ""))


def filter_out(data: ow.DataType, **kwargs) -> ow.DataType:
    """Filters rows based on column/value conditions."""
    import polars as pl
    from typing import List, Any
    from functools import reduce
    import operator as op

    keep: str | List[Any] = kwargs.get("keep_val", "")
    col: str | List[str] = kwargs.get("keep_col", "")
    oper = kwargs.get("keep_oper", "")
    tick = kwargs.get("tick", "")

    if isinstance(keep, str):
        keep = [keep]
    if isinstance(col, str):
        col = [col]
    if not isinstance(oper, list):
        oper = [oper]

    df = data()
    exprs = [o(pl.col(c), d) for d, c, o in zip(keep, col, oper)]
    df = df.filter(reduce(op.and_, exprs))
    return ow.DataType(df, tick)


def underlying_close(data: ow.DataType, **kwargs) -> ow.DataType:
    """Adds underlying close price to data."""
    import yfinance as yf
    import polars as pl
    import pandas as pd
    import time

    tick = kwargs.get("tick", "")

    def _is_rate_limited(err: Exception) -> bool:
        resp = getattr(err, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 429:
            return True
        return "429" in str(err)

    def _with_backoff(fn, attempts: int = 5, base_delay: float = 1.0):
        for i in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 — we need to inspect for 429
                if _is_rate_limited(exc) and i < attempts:
                    time.sleep(base_delay * i)
                    continue
                raise

    ticker = yf.Ticker(tick)

    df = data()
    if isinstance(df, pl.LazyFrame):
        stats = df.select(
            pl.col("trade_date").min().alias("min_td"),
            pl.col("trade_date").max().alias("max_td"),
        ).collect()
        start = stats["min_td"][0]
        end = stats["max_td"][0]
        df = df.collect()
    else:
        start = df.select(pl.col("trade_date").min()).item()
        end = df.select(pl.col("trade_date").max()).item()

    hist = _with_backoff(
        lambda: ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            interval="1d",
        )
    )

    if hist is None or hist.empty:
        return ow.DataType(None, tick)

    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"])
    if hist.empty:
        return ow.DataType(None, tick)

    splits = hist["Stock Splits"].fillna(0).replace(0, 1.0)
    split_factor = splits.iloc[::-1].cumprod().iloc[::-1]

    hist["Date"] = hist["Date"].dt.tz_localize(None)
    hist["split_factor"] = split_factor
    hist["raw_underlying_close"] = hist["Close"] * split_factor

    hist_pl = (
        pl.from_pandas(hist)
        .with_columns(pl.col("Date").dt.date().alias("trade_date"))
        .rename({"Close": "underlying_close"})
        .select(
            ["trade_date", "underlying_close", "split_factor", "raw_underlying_close"]
        )
    )

    merged_df = df.join(hist_pl, on="trade_date", how="left")

    return ow.DataType(merged_df.lazy(), tick)


def log_moneyness(data: ow.DataType, **kwargs) -> ow.DataType:
    """Calculates log-moneyness for each option row."""
    import polars as pl

    df = data()

    df = df.with_columns(
        (pl.col("strike") / pl.col("underlying_close")).log().alias("log_moneyness")
    )

    return ow.DataType(df, kwargs.get("tick", ""))


def scale_splits(data: ow.DataType, **kwargs) -> ow.DataType:
    """Scales data for stock splits using Polars expressions."""
    import polars as pl

    df = data()
    tick = kwargs.get("tick", "")

    cols_to_reduce = ["last_trade_price", "bid_price", "ask_price", "strike"]
    cols_to_increase = ["volume", "open_interest"]

    df = df.with_columns(
        [(pl.col(c) / pl.col("split_factor")).alias(c) for c in cols_to_reduce]
        + [(pl.col(c) * pl.col("split_factor")).alias(c) for c in cols_to_increase]
    )

    return ow.DataType(df, tick)


def options_entry(data: ow.DataType, **kwargs) -> ow.DataType:
    import polars as pl

    df_raw = data()
    # Normalize to LazyFrame for consistent joins
    df = df_raw.lazy() if isinstance(df_raw, pl.DataFrame) else df_raw
    specs = kwargs.get("specs", [])
    tick = kwargs.get("tick", "")

    def _safe_apply(value, fn):
        """Guard user filters against nulls from sparse greeks data."""
        return False if value is None else bool(fn(value))

    base_cols = [
        c for c in df.collect_schema().names() if c not in ("entered", "position")
    ]

    all_entries = []

    for spec in specs:
        # Bind all spec-level callables as default args so the lazy Polars
        # expressions don't end up using the last spec's filters (late binding).
        call_put = spec.call_put
        lm_fn = spec.lm_fn
        ttm_fn = spec.ttm
        abs_delta_fn = spec.abs_delta
        entry_cond_fn = (
            spec.entry_cond if isinstance(spec.entry_cond, List) else [spec.entry_cond]
        )
        open_interest_min = spec.open_interest_min
        volume_min = spec.volume_min
        entry_col = (
            spec.entry_col if isinstance(spec.entry_col, List) else [spec.entry_col]
        )
        minimise_col = spec.entry_min
        max_hold = spec.max_hold_period
        position = spec.position

        # ---- base eligibility filters ----
        eligible = df.filter(
            (pl.col("call_put") == call_put)
            & pl.struct(["log_moneyness"]).map_elements(
                lambda s, fn=lm_fn: _safe_apply(s["log_moneyness"], fn)
            )
            & pl.struct(["ttm"]).map_elements(
                lambda s, fn=ttm_fn: _safe_apply(s["ttm"], fn)
            )
            & pl.struct(["delta"]).map_elements(
                lambda s, fn=abs_delta_fn: (
                    False if s["delta"] is None else bool(fn(abs(s["delta"])))
                )
            )
            & (pl.col("n_missing") == 0)
            & (pl.col("days_until_last_trade") > max_hold)
            & (pl.col("in_universe") == True)
            & (pl.col("open_interest") >= open_interest_min)
            & (pl.col("volume") >= volume_min)
        )

        # ---- optional entry condition ----
        if entry_col and all(col is not None for col in entry_col):
            for col, cond in zip(entry_col, entry_cond_fn):
                eligible = eligible.filter(
                    pl.col(col).map_elements(lambda x, fn=cond: _safe_apply(x, fn))
                )

        # ---- pick best per day ----
        entries = (
            eligible.with_columns(pl.col(minimise_col).abs().alias("rank_metric"))
            .sort(["trade_date", "rank_metric"])
            .group_by("trade_date")
            .head(1)
            .with_columns(
                [
                    pl.lit(True).alias("entered"),
                    pl.lit(position).alias("position"),
                    pl.lit(max_hold).alias("max_hold_period"),
                ]
            )
            .select(base_cols + ["entered", "position", "max_hold_period"])
        )

        all_entries.append(entries)

    if all_entries:
        entries = pl.concat(all_entries, how="vertical")

        # Only allow entries on dates where every spec produced a candidate.
        # This guards against partially-filled days slipping through when,
        # for example, one leg fails its entry filter.
        if len(specs) > 0:
            entries = (
                entries.with_columns(
                    pl.count().over("trade_date").alias("_entries_per_day")
                )
                .filter(pl.col("_entries_per_day") == len(specs))
                .drop("_entries_per_day")
            )
    else:
        entries = df.head(0).with_columns(
            [
                pl.lit(False).alias("entered"),
                pl.lit(0.0).alias("position"),
                pl.lit(0).alias("max_hold_period"),
            ]
        )

    df_out = df.join(
        entries.lazy(),
        on=df.collect_schema().names(),
        how="left",
    ).with_columns(
        pl.col("entered").fill_null(False),
        pl.col("position").fill_null(0),
        pl.col("max_hold_period").fill_null(0),
    )

    return ow.DataType(df_out, tick)


def options_trade(data: ow.DataType, **kwargs) -> ow.StratType:
    from collections import deque
    from functools import lru_cache
    import polars as pl
    from options_wizard import (
        DateObj,
        Option,
        OptionType,
        Spot,
        EntryData,
        PositionType,
        PriceSeries,
    )

    specs = kwargs.get("specs", [])
    tick = kwargs.get("tick", "")
    other_data = kwargs.get("other_data", [])

    # map spec by (call_put, position) for finding exit rules
    spec_lookup = {(s.call_put, s.position): s for s in specs}

    @lru_cache(None)
    def date_obj(dt):
        return DateObj(day=dt.day, month=dt.month, year=dt.year)

    df_raw = data()
    df = df_raw.collect() if isinstance(df_raw, pl.LazyFrame) else df_raw
    df = df.sort("trade_date")

    grouped = df.group_by(["call_put", "strike", "expiry_date"])
    contract_map = {(k[0], k[1], k[2]): group for k, group in grouped}

    trade_entries = df.filter(pl.col("entered") == True)
    trades = deque()

    for trade in trade_entries.iter_rows(named=True):

        spec = spec_lookup.get((trade["call_put"], trade["position"]), None)
        if spec is None:
            continue

        exit_cond_fn = (
            spec.exit_cond if isinstance(spec.exit_cond, List) else [spec.exit_cond]
        )
        exit_col = spec.exit_col if isinstance(spec.exit_col, List) else [spec.exit_col]
        max_hold = trade["max_hold_period"]

        key = (trade["call_put"], trade["strike"], trade["expiry_date"])
        group = contract_map.get(key)
        if group is None:
            continue

        start = trade["trade_date"]
        final_exit_date = start + pl.duration(days=max_hold)

        feasible = group.filter(
            (pl.col("trade_date") >= start) & (pl.col("trade_date") <= final_exit_date)
        )

        rows = feasible.to_dicts()
        if not rows:
            continue

        price_series = PriceSeries(tick=tick)
        exit_date = None

        for r in rows:
            d = date_obj(r["trade_date"])
            expiry = date_obj(r["expiry_date"])

            underlying = Spot(
                bid=r["underlying_close"],
                ask=r["underlying_close"],
                volume=0.0,
                date=d,
                tick=tick,
            )

            mid_iv = None
            if (
                r["bid_implied_volatility"] is not None
                and r["ask_implied_volatility"] is not None
            ):
                mid_iv = (r["bid_implied_volatility"] + r["ask_implied_volatility"]) / 2

            others = {}
            for datum in other_data:
                others[datum] = r[datum]

            option_contract = Option(
                bid=r["bid_price"],
                ask=r["ask_price"],
                volume=r["volume"],
                date=d,
                tick=tick,
                option_type=OptionType.CALL if r["call_put"] == "c" else OptionType.PUT,
                strike=r["strike"],
                expiry=expiry,
                iv=mid_iv,
                underlying=underlying,
                rfr=None,
                delta=r["delta"],
                gamma=r["gamma"],
                vega=r["vega"],
                theta=r["theta"],
                rho=r.get("rho"),
                other=others,
            )

            price_series.add(option_contract)

            # ---- exit rule: exit_cond() applied correctly ----
            if exit_col and all(col is not None for col in exit_col):
                fired = any(cond(r[col]) for cond, col in zip(exit_cond_fn, exit_col))
            else:
                fired = any(cond(r) for cond in exit_cond_fn)

            if fired and exit_date is None:
                exit_date = d

        if exit_date is None:
            exit_date = d  # last available date

        entry = EntryData(
            entry_date=date_obj(start),
            position_type=(
                PositionType.LONG if trade["position"] > 0 else PositionType.SHORT
            ),
            price_series=price_series,
            exit_date=exit_date,
            position_size=abs(trade["position"]),
            features=None,
            tick=tick,
        )

        trades.append(entry)

    return ow.StratType(trades, tick)


# ============================================================
#                   Strategy Evaluation Pipeline
# ============================================================


def add_put_spread_methods(pipeline: ow.Pipeline, kwargs) -> None:
    """
    Attach put-spread methods to a pipeline.

    Uses @ow.wrap_fn decorators *here* so we can bind
    pipeline + kwargs dynamically, while keeping all logic
    functions importable at module scope.
    """
    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)

    @ow.wrap_fn(ow.FuncType.LOAD)
    def load_data_wrapped(**fn_kwargs) -> ow.DataType:
        return load_data(**fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def in_universe_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return in_universe(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def perc_spread_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return perc_spread(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def filter_gaps_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return filter_gaps(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def ttms_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return ttms(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def filter_out_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return filter_out(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def underlying_close_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return underlying_close(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def scale_splits_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return scale_splits(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA, depends_on=[load_data_wrapped])
    def log_moneyness_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return log_moneyness(data, **fn_kwargs)

    @ow.wrap_fn(
        ow.FuncType.DATA,
        depends_on=[
            load_data_wrapped,
            filter_gaps_wrapped,
            ttms_wrapped,
            filter_out_wrapped,
            perc_spread_wrapped,
            underlying_close_wrapped,
            scale_splits_wrapped,
            in_universe_wrapped,
        ],
    )
    def options_entry_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return options_entry(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[options_entry_wrapped])
    def options_trade_wrapped(data: ow.DataType, **fn_kwargs) -> ow.StratType:
        return options_trade(data, **fn_kwargs)

    return None
