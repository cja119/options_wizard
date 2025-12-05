"""
Methods for put spread strategy
"""

from typing import Tuple, List, Dict, override
import options_wizard as ow
from functools import partial
import numpy as np


class FixedHoldNotional(ow.PositionBase):
    _carry = 0.0
    _notional_exposure = None

    @override
    def size_function(self, entering_trades: List[ow.Trade]) -> Dict[ow.Trade, float]:
        protected_notional = self._kwargs.get("protected_notional", 1_000_000)
        hold_period = self._kwargs.get("hold_period", 30)
        notional = protected_notional / (hold_period)

        # If nothing to enter today, roll the daily budget forward once
        if not entering_trades:
            self._carry += notional
            return {}

        # Deploy the accumulated budget across all trades entering today
        notional += self._carry
        self._carry = 0.0

        # Anchor sizing on short-leg exposure per ticker so each name gets the
        # same notional allocation on a given day, regardless of how many legs
        # it uses.
        short_exp_by_tick = {}
        for trade in entering_trades:
            if trade.entry_data.position_type != ow.PositionType.SHORT:
                continue
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            if entry_px is None or entry_px.delta is None or entry_px.underlying is None:
                continue
            exp = abs(
                entry_px.delta * trade.entry_data.position_size * entry_px.underlying.ask
            )
            short_exp_by_tick[entry_px.tick] = short_exp_by_tick.get(entry_px.tick, 0.0) + exp

        if not short_exp_by_tick:
            # No meaningful anchor to size off today; roll budget forward.
            self._carry += notional
            return {trade: 0.0 for trade in entering_trades}

        per_tick_budget = notional / len(short_exp_by_tick)

        sizes = {}
        for trade in entering_trades:
            entry_px = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            tick = entry_px.tick if entry_px is not None else None
            tick_exp = short_exp_by_tick.get(tick, 0.0)
            if tick_exp == 0.0:
                sizes[trade] = 0.0
                print("Trade ZERO size due to zero exposure anchor.")
                continue
            sizes[trade] = per_tick_budget / tick_exp

        return sizes

    @override
    def exit_trigger(
        self,
        live_trades: List[ow.Trade],
        current_date: ow.DateObj,
        scheduled_exits: List[ow.Trade],
    ) -> Tuple[List[ow.Trade], List[ow.Cashflow]]:
        exit_cashflows = []
        trades_to_close = set()
        max_days_open = self._kwargs.get("hold_period", 30)
        for trade in live_trades:
            if trade.days_open(current_date) >= max_days_open:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)

        for trade in scheduled_exits:
            if trade not in trades_to_close:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)
        return trades_to_close, exit_cashflows


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
    data = pl.scan_parquet(os.path.join(tick_path, f"{tick}.parquet"))

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
        index_df = pd.read_csv(
            os.getenv("INDEX_CONSTITUENTS_PATH"), na_values=na_vals
        )

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
        mktcap_long = mktcap_long.rename(
            columns={"level_1": "col_idx", 0: "mktcap"}
        )

        merged = pd.merge(
            index_long, mktcap_long, on=["date", "col_idx"], how="outer"
        )
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

    daily_panel = (
        universe_frame.reindex(daily_index)
        .ffill()
        .where(membership)
    )

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
        df.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_row_index("day_idx")
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

    bounds = bounds.with_columns(
        (pl.col("end") - pl.col("start")).alias("_tmp")
    )

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
                pl.col("expiry_date").cast(pl.Date)
                - pl.col("trade_date").cast(pl.Date)
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

    tick = kwargs.get("tick", "")
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

    hist = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        interval="1d",
    ).reset_index()

    hist["Date"] = hist["Date"].dt.tz_localize(None)

    hist_pl = (
        pl.from_pandas(hist)
        .with_columns(pl.col("Date").dt.date().alias("trade_date"))
        .rename({"Close": "underlying_close"})
        .select(["trade_date", "underlying_close"])
    )

    merged_df = df.join(hist_pl, on="trade_date", how="left")

    return ow.DataType(merged_df.lazy(), tick)


def scale_splits(data: ow.DataType, **kwargs) -> ow.DataType:
    """Scales data for stock splits using Polars expressions."""
    import yfinance as yf
    import polars as pl
    import pandas as pd

    apply_adjustment = kwargs.get("apply_split_adjustment", True)
    if not apply_adjustment:
        return data

    tick = kwargs.get("tick", "")
    ticker = yf.Ticker(tick)
    splits_override = kwargs.get("splits_override")
    splits = splits_override if splits_override is not None else ticker.splits

    if splits is None or getattr(splits, "empty", False):
        return data

    df = data()
    if isinstance(splits, pl.DataFrame):
        splits = splits.rename({"date": "Date", "split": "Split"})
    elif isinstance(splits, pd.Series):
        splits = splits.reset_index().rename(
            columns={"index": "Date", "Stock Splits": "Split"}
        )
    elif isinstance(splits, pd.DataFrame):
        splits = splits.rename(
            columns={"index": "Date", "Stock Splits": "Split", "split": "Split"}
        )
    else:
        splits = pd.DataFrame(splits).reset_index().rename(
            columns={"index": "Date", "Stock Splits": "Split", "split": "Split"}
        )

    splits["Date"] = splits["Date"].dt.normalize()  
    splits = splits.sort_values("Date")

    cols_to_reduce = ["last_trade_price", "bid_price", "ask_price", "strike", "underlying_close"]
    cols_to_increase = ["volume", "open_interest", "num_shares"]

    df = df.with_columns(pl.lit(100).alias("num_shares"))

    for date, ratio in zip(splits["Date"], splits["Split"]):
        df = df.with_columns(
            [
                pl.when(pl.col("trade_date") < date)
                .then(pl.col(col) / ratio)
                .otherwise(pl.col(col))
                .alias(col)
                for col in cols_to_reduce
            ]
            + [
                pl.when(pl.col("trade_date") < date)
                .then(pl.col(col) * ratio)
                .otherwise(pl.col(col))
                .alias(col)
                for col in cols_to_increase
            ]
        )

    return ow.DataType(df, tick)

def ratio_spread(data: ow.DataType, **kwargs) -> ow.DataType:
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
        c for c in df.collect_schema().names()
        if c not in ("entered", "position")
    ]

    all_entries = []

    for spec in specs:
        # Bind all spec-level callables as default args so the lazy Polars
        # expressions don't end up using the last spec's filters (late binding).
        call_put = spec.call_put
        strike_fn = spec.strike
        ttm_fn = spec.ttm
        abs_delta_fn = spec.abs_delta
        entry_cond_fn = spec.entry_cond
        entry_col = spec.entry_col
        minimise_col = spec.entry_min
        max_hold = spec.max_hold_period
        position = spec.position

        # ---- base eligibility filters ----
        eligible = df.filter(
            (pl.col("call_put") == call_put)
            & pl.struct(["strike"]).map_elements(
                lambda s, fn=strike_fn: _safe_apply(s["strike"], fn)
            )
            & pl.struct(["ttm"]).map_elements(
                lambda s, fn=ttm_fn: _safe_apply(s["ttm"], fn)
            )
            & pl.struct(["delta"]).map_elements(
                lambda s, fn=abs_delta_fn: (
                    False
                    if s["delta"] is None
                    else bool(fn(abs(s["delta"])))
                )
            )
            & (pl.col("n_missing") == 0)
            & (pl.col("days_until_last_trade") > max_hold)
            & (pl.col("in_universe") == True)
        )

        # ---- optional entry condition ----
        if entry_col is not None:
            eligible = eligible.filter(
                pl.col(entry_col).map_elements(
                    lambda x, fn=entry_cond_fn: fn(x)
                )
            )
        else:
            eligible = eligible.filter(
                pl.struct(df.collect_schema().names()).map_elements(
                    lambda row, fn=entry_cond_fn: fn(row)
                )
            )
        
        # ---- pick best per day ----
        entries = (
            eligible
            .with_columns(pl.col(minimise_col).abs().alias("rank_metric"))
            .sort(["trade_date", "rank_metric"])
            .group_by("trade_date")
            .head(1)
            .with_columns([
                pl.lit(True).alias("entered"),
                pl.lit(position).alias("position"),
                pl.lit(max_hold).alias("max_hold_period")
            ])
            .select(base_cols + ["entered", "position", "max_hold_period"])
        )

        all_entries.append(entries)

    if all_entries:
        entries = pl.concat(all_entries, how="vertical")
    else:
        entries = df.head(0).with_columns(
            [
                pl.lit(False).alias("entered"),
                pl.lit(0.0).alias("position"),
                pl.lit(0).alias("max_hold_period"),
            ]
        )

    df_out = (
        df.join(
            entries.lazy(),
            on=df.collect_schema().names(),
            how="left",
        )
        .with_columns(
            pl.col("entered").fill_null(False),
            pl.col("position").fill_null(0),
            pl.col("max_hold_period").fill_null(0),
        )
    )

    return ow.DataType(df_out, tick)


def fixed_hold_trade(data: ow.DataType, **kwargs) -> ow.StratType:
    from collections import deque
    from functools import lru_cache
    import polars as pl
    from options_wizard import (
        DateObj, Option, OptionType, Spot,
        EntryData, PositionType, PriceSeries,
    )

    specs = kwargs.get("specs", [])
    tick = kwargs.get("tick", "")

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

        exit_cond_fn = spec.exit_cond
        exit_col = spec.exit_col
        max_hold = trade["max_hold_period"]

        key = (trade["call_put"], trade["strike"], trade["expiry_date"])
        group = contract_map.get(key)
        if group is None:
            continue

        start = trade["trade_date"]
        final_exit_date = start + pl.duration(days=max_hold)

        feasible = group.filter(
            (pl.col("trade_date") >= start)
            & (pl.col("trade_date") <= final_exit_date)
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
                tick=tick
            )

            mid_iv = None
            if r['bid_implied_volatility'] is not None and r['ask_implied_volatility'] is not None:
                mid_iv = (r["bid_implied_volatility"] + r["ask_implied_volatility"]) / 2

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
                num_underlying=r.get("num_shares"),
                rfr=None,
                delta=r["delta"],
                gamma=r["gamma"],
                vega=r["vega"],
                theta=r["theta"],
                rho=r.get("rho"),
            )

            price_series.add(option_contract)

            # ---- exit rule: exit_cond() applied correctly ----
            if exit_col is not None:
                fired = exit_cond_fn(r[exit_col])
            else:
                fired = exit_cond_fn(r)

            if fired and exit_date is None:
                exit_date = d

        if exit_date is None:
            exit_date = d  # last available date


        entry = EntryData(
            entry_date=date_obj(start),
            position_type=PositionType.LONG if trade["position"] > 0 else PositionType.SHORT,
            price_series=price_series,
            exit_date=exit_date,
            position_size=abs(trade["position"]),
            features=None,
            tick=tick
        )

        trades.append(entry)

    return ow.StratType(trades, tick)

# ============================================================
#                   PIPELINE REGISTRATION 
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
    def ratio_spread_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return ratio_spread(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT, depends_on=[ratio_spread_wrapped])
    def fixed_hold_trade_wrapped(data: ow.DataType, **fn_kwargs) -> ow.StratType:
        return fixed_hold_trade(data, **fn_kwargs)


    return None
