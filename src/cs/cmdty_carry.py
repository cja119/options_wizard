from dataclasses import dataclass, field
from tokenize import group
from typing import List, Dict, Deque
from collections import deque
from functools import partial
import options_wizard as ow
import polars as pl

# ============================================================
#   TOP-LEVEL LOGIC FUNCTIONS (IMPORTABLE, UNDECORATED)
# ============================================================

def load_data(**kwargs) -> ow.DataType:
    """Loads in the commodity futures data"""
    import os
    from dotenv import load_dotenv
    import polars as pl

    tick = kwargs.get("tick", None)

    load_dotenv()
    cmdty_path = os.getenv("CMDTY_PATH", "").split(os.pathsep)[0]
    files = [
        f.path
        for f in os.scandir(cmdty_path)
        if f.is_file() and f.name.endswith(".parquet")
    ]

    df = None
    for file in files:
        if tick == file.split("\\")[-1].replace("_FUT.parquet", ""):
            df = pl.scan_parquet(file)

    if df is None:
        raise ValueError(f"Tick {tick} not found in CMDTY_PATH")

    df = (
        df.with_columns(
            pl.col("LAST_TRADEABLE_DT").str.strptime(
                pl.Date, format="%d/%m/%Y", strict=False
            )
        )
        .with_columns(
            pl.col("FUT_NOTICE_FIRST").str.strptime(
                pl.Date, format="%d/%m/%Y", strict=False
            )
        )
        .with_columns(pl.col("Date").cast(pl.Date))
    )

    return ow.DataType(data=df, tick=tick)

def days_to_anchor(data: ow.DataType, **kwargs) -> ow.DataType:

    import pandas as pd
    import polars as pl
    import options_wizard as ow

    df = data._data
    tick = kwargs.get("tick", None)
    min_dte = kwargs.get("min_dte", 21)
    

    min_day = pd.Timestamp(df.select(pl.col("Date").min()).collect().item())
    max_day = pd.Timestamp(
        df.select(pl.col("LAST_TRADEABLE_DT").max()).collect().item()
    )

    if max_day > pd.Timestamp.now():
        max_day = pd.Timestamp.now().date()
    df = df.filter(pl.col("LAST_TRADEABLE_DT") <= max_day)

    exchange = df.select(pl.col("EXCH_CODE")).unique().collect().item()

    dates = ow.market_dates(exchange=exchange, lower=min_day, upper=max_day)

    calendar = (
        pl.LazyFrame({"Date": dates})
        .with_columns(pl.col("Date").cast(pl.Date))
        .with_row_index("TradeDateIdx")
        .sort("Date")
    )

    df = df.sort("Date")

    df = (
        df.join_asof(calendar, on="Date", strategy="backward")
        .filter(pl.col("TradeDateIdx").is_not_null())
        .rename({"TradeDateIdx": "DateIdx"})
    )

    calendar_ltd = calendar.rename(
        {"Date": "LAST_TRADEABLE_DT", "TradeDateIdx": "LtdIdx"}
    ).sort("LAST_TRADEABLE_DT")

    df = df.sort("LAST_TRADEABLE_DT")

    df = df.join_asof(calendar_ltd, on="LAST_TRADEABLE_DT", strategy="backward").filter(
        pl.col("LtdIdx").is_not_null()
    )

    df = df.with_columns((pl.col("LtdIdx") - pl.col("DateIdx")).alias("DAYS_TO_ANCHOR"))
    df = df.with_columns(
        pl.col("DAYS_TO_ANCHOR").min().over("Date").alias("DAYS_TO_FRONT_ANCHOR")
    )

    return ow.DataType(data=df, tick=tick)

def carry_entry(data: "ow.DataType", **kwargs) -> "ow.DataType":
    import polars as pl
    import options_wizard as ow
    import numpy as np
    from functools import lru_cache
    
    # Date helper function
    @lru_cache(maxsize=None)
    def date_obj(dt):
        return ow.DateObj(day=dt.day, month=dt.month, year=dt.year)
    
    @lru_cache(maxsize=None)
    def last_price(contract_series):
        dt = contract_series.select("Date").last().item()
        return ow.DateObj(day=dt.day, month=dt.month, year=dt.year)

    # Parameters
    carry_spec = kwargs.get("carry_spec", None)
    tick = kwargs.get("tick", None)
    vol_window = kwargs.get("vol_window", 60)
    front_carry = True if carry_spec.metric == "FRONT_RELATIVE" else False
    
    # Unpack carry spec
    roll_target = carry_spec.roll_target
    tenor_targets = carry_spec.tenor_targets
    exposure_targets = carry_spec.exposure_targets
    spread = carry_spec.spread_override_bps
    
    # Collect data
    base = data._data.collect()
    grouped = base.group_by("Contract")
    contract_map = {k[0]: v.sort("Date") for k, v in grouped}
    
    # Precompute last available Date per contract 
    last_dt_map = {}
    for cid, df_c in contract_map.items():
        dt = df_c.select(pl.col("Date").max()).item()
        last_dt_map[cid] = date_obj(dt)

    trades = []
    lg_rets = deque(maxlen=vol_window)
    exdates = deque(maxlen=vol_window)
    durations = deque(maxlen=vol_window)

    # Need to iterate in order
    dates = base.select("Date").unique().sort("Date").to_series().to_list()
    for date in dates:
        group = base.filter(pl.col("Date") == date).sort("DAYS_TO_ANCHOR")
        group = group.sort("DAYS_TO_ANCHOR")
        date = date[0] if isinstance(date, (list, tuple)) else date

        # Drop contracts within roll_target
        group = group.filter(pl.col("DAYS_TO_FRONT_ANCHOR") > roll_target)
        if group.height == 0:
            continue

        if group.height < max(tenor_targets):
            continue

        front_price = group.select(pl.col("PX_SETTLE").first()).item()
        front_dte = group.select(pl.col("DAYS_TO_ANCHOR").first()).item()
        front_multip = 1.0 #group.select(pl.col("FUT_CONT_SIZE").first()).item()

        cum_carry = 0.0
        contract_ids = []
        last_dts = []

        for i, row in enumerate(group.iter_rows(named=True)):
            if i == 0:
                continue

            denom = (row["DAYS_TO_ANCHOR"] - front_dte)
            if denom <= 0:
                continue

            carry = np.log(row["PX_SETTLE"] / front_price) * (252.0 / denom)

            tenor_pos = i + 1  # position in filtered curve
            if tenor_pos in tenor_targets:
                target_idx = tenor_targets.index(tenor_pos)
                cum_carry += carry * float(exposure_targets[target_idx])
                contract_ids.append(row["Contract"])

            if not front_carry:
                front_price = row["PX_SETTLE"]
                front_dte = row["DAYS_TO_ANCHOR"]

            last_dts.append(last_dt_map[row["Contract"]])

        rets = []
        for t, r, d in zip(exdates, lg_rets, durations):
            if t > date_obj(date):
                break
            rets.append(r / np.sqrt(d) if d > 0 else 0.0)

        spread_features = ow.SpreadFeatures(
            other_contracts=tuple(contract_ids),
            carry_score=cum_carry,
            notional_exposure = 2 * front_price * front_multip * abs(exposure_targets[0]),
            volatility = np.std(rets) * np.sqrt(365) if len(rets) > 1 else 0.0
        )
        exdate = min(last_dts)
        price_series = {}
        positions = {}
        position_sizes = {}

        # Now we have the cumulative carry metric for this date, we can log trades
        for i, row in enumerate(group.iter_rows(named=True)):
            
            tenor_pos = i + 1  # position in filtered curve

            if tenor_pos in tenor_targets:
                
                contract = row["Contract"]
                if tenor_pos == tenor_targets[0]:
                    positions[contract] = ow.PositionType.LONG if cum_carry < 0 else ow.PositionType.SHORT
                else:
                    positions[contract] = ow.PositionType.SHORT if cum_carry < 0 else ow.PositionType.LONG
                

                # Extracting all contracts for this tenor position
                contracts = contract_map[contract].filter((pl.col("Date") >= date) & (pl.col("Date") <= exdate.to_datetime()))
                target_idx = tenor_targets.index(tenor_pos)
                position_sizes[contract] = abs(float(exposure_targets[target_idx])) * (front_price / row["PX_SETTLE"])

                # Order in increasing date and creating a blank price series
                price_series[contract] = ow.PriceSeries(tick)

                # Now, iterate through future dats to build the price series
                for row in contracts.iter_rows(named=True):
                    
                    future_contract = ow.Future(
                        # --- Required Fields --- #
                        bid = row["PX_BID"] if spread is None else  (1 - spread / 20000) * row["PX_SETTLE"],
                        ask = row["PX_ASK"] if spread is None else (1 + spread / 20000) * row["PX_SETTLE"],
                        volume = row["VOLUME"],
                        date = date_obj(row["Date"]),
                        tick = tick,
                        expiry = date_obj(row["LAST_TRADEABLE_DT"]),
                        contract_id = row["Contract"],
                        
                        # -- Optional Fields --- #
                        open_interest= row["OPEN_INT"],
                        settlement_price= row["PX_SETTLE"],
                        contract_multiplier= row["FUT_CONT_SIZE"],
                        )
                    price_series[contract].add(future_contract)
                
                while any(price_series[contract].get(date_obj(exdate)) is None for contract in price_series):
                    exdate = date_obj(exdate.to_datetime() - np.timedelta64(1, 'D'))
        
        numerator = 0.0
        denominator = 2 * front_price
        duration = (date_obj(exdate).to_datetime() - date_obj(date).to_datetime()).days 
        for key in price_series:      
            entry = ow.EntryData(
                entry_date = date_obj(date),
                position_type = positions[key],
                price_series = price_series[key],
                exit_date = date_obj(exdate),
                tick = tick,
                position_size = position_sizes[key],
                features = spread_features
            )
            trades.append(entry)
        
            numerator += position_sizes[key] * positions[key] * (
                price_series[key].get(date_obj(exdate)).settlement_price 
                - price_series[key].get(date_obj(date)).settlement_price 
            )

        durations.append(duration)
        lg_rets.append( np.log(1 + numerator / denominator) if denominator != 0 else 0.0 ) 
        exdates.append(date_obj(exdate))

    return ow.StratType(trades, tick)


def add_cmdty_carry_methods(pipeline: ow.Pipeline, kwargs) -> None:
    """
    Attach cmdty-carry methods to a pipeline.

    Uses @ow.wrap_fn decorators *here* so we can bind
    pipeline + kwargs dynamically, while keeping all logic
    functions importable at module scope.
    """
    ow.wrap_fn = partial(ow.wrap_fn, pipeline=pipeline, kwargs=kwargs)

    @ow.wrap_fn(ow.FuncType.LOAD)
    def load_data_wrapped(**fn_kwargs) -> ow.DataType:
        return load_data(**fn_kwargs)

    @ow.wrap_fn(ow.FuncType.DATA)
    def days_to_anchor_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return days_to_anchor(data, **fn_kwargs)

    @ow.wrap_fn(ow.FuncType.STRAT)
    def carry_entries_wrapped(data: ow.DataType, **fn_kwargs) -> ow.DataType:
        return carry_entry(data, **fn_kwargs)

    return None
