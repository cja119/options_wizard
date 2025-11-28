import pandas as pd
import numpy as np
from typing import Any, Dict, List

def get_underlying(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds the underlying Yahoo Finance close price to each row in an options chain DataFrame.
    Does NOT modify the type or index of trade_date.
    """

    import yfinance as yf

    try:
        ticker = kwargs.get("tick", "")
        if not ticker:
            raise ValueError("Missing required argument: 'tick' (underlying symbol)")

        # Convert trade_date just for range and merge purposes
        temp_dates = pd.to_datetime(data['trade_date'], errors='coerce')
        try:
            temp_dates = temp_dates.dt.tz_localize(None)
        except TypeError:
            pass
        min_date, max_date = temp_dates.min(), temp_dates.max()

        if pd.isna(min_date) or pd.isna(max_date):
            raise ValueError("No valid trade_date values found in input data.")

        # --- Download underlying data ---
        close_prices = yf.download(
            ticker,
            start=min_date - pd.Timedelta(days=2),
            end=max_date + pd.Timedelta(days=2),
            progress=False,
            auto_adjust=False,
            interval="1d"
        )[["Close"]]

        # Reset index so 'Date' becomes a column for merging
        close_df = (
            close_prices
            .reset_index()
            .rename(columns={"Date": "merge_date", "Close": "underlying_close"})
        )

        # --- Flatten MultiIndex columns if necessary ---
        if isinstance(close_prices.columns, pd.MultiIndex):
            close_prices.columns = ['_'.join(col).strip() for col in close_prices.columns.values]

        # Now select Close column (handles 'Close' or 'Close_<TICKER>')
        close_col = [c for c in close_prices.columns if c.lower().startswith('close')]
        close_prices = close_prices[close_col].rename(columns={close_col[0]: 'Close'})
        close_df = close_prices.reset_index().rename(columns={'Date': 'merge_date', 'Close': 'underlying_close'})

        # Create a temporary datetime column for merging
        merge_dates = pd.to_datetime(data["trade_date"], errors="coerce")
        try:
            merge_dates = merge_dates.dt.tz_localize(None)
        except TypeError:
            pass
        close_df["merge_date"] = pd.to_datetime(close_df["merge_date"], errors="coerce")
        try:
            close_df["merge_date"] = close_df["merge_date"].dt.tz_localize(None)
        except TypeError:
            pass
        close_series = close_df.set_index("merge_date")["underlying_close"]

        data["underlying_close"] = merge_dates.map(close_series)
        return data

    except Exception as e:
        raise RuntimeError(f"Failed to get underlying data for {kwargs.get('tick', '')}: {e}")


def flag_chain_gaps(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Flags gaps in option chains and returns the list of missing dates per contract.
    """

    import pytz
    import exchange_calendars as ec

    gap_threshold: int = kwargs.get("gap_threshold", 1)
    # Remove timezone for consistency
    norm_trade_date = pd.to_datetime(data['trade_date'], errors='coerce')
    try:
        norm_trade_date = norm_trade_date.dt.tz_localize(None)
    except TypeError:
        pass
    data['norm_trade_date'] = norm_trade_date

    # Get NASDAQ calendar
    cal = ec.get_calendar("XNAS")
    if not hasattr(cal.tz, "key"):
        cal.tz = pytz.timezone("America/New_York")

    # Prepare a list to collect results
    results = []

    # Group by contract
    for (expiry, strike, call_put), group in data.groupby(['expiry_date', 'strike', 'call_put']):
        # Contract-specific trade dates
        observed = pd.to_datetime(group['norm_trade_date'].unique())
        observed = pd.DatetimeIndex(observed)
        expected_sessions = cal.sessions_in_range(observed.min(), observed.max())
        expected_sessions = pd.DatetimeIndex(expected_sessions)
        missing_dates = list(expected_sessions.difference(observed))

        results.append({
            'expiry_date': expiry,
            'strike': strike,
            'call_put': call_put,
            'missing_dates': missing_dates,
            'missing_count': len(missing_dates),
            'gap_flag': len(missing_dates) > gap_threshold
        })

    # Convert results to DataFrame
    missing_df = pd.DataFrame(results)

    if missing_df.empty:
        result = data.copy()
        result['missing_dates'] = [[] for _ in range(len(result))]
        result['missing_count'] = 0
        result['gap_flag'] = False
    else:
        missing_df = missing_df.set_index(['expiry_date', 'strike', 'call_put'])
        result = data.join(
            missing_df[['missing_dates', 'missing_count', 'gap_flag']],
            on=['expiry_date', 'strike', 'call_put']
        )

    last_trade = (
        data.groupby(['expiry_date', 'strike', 'call_put'])['norm_trade_date']
            .max()
            .rename("last_trade_date")
    )

    # Join back into the main result
    result = result.join(
        last_trade,
        on=['expiry_date', 'strike', 'call_put']
    )

    # Calendar-day difference
    result['days_until_last_trade'] = (
        result['last_trade_date'] - result['norm_trade_date']
    ).dt.days
    result.drop(columns=['norm_trade_date'], inplace=True, errors='ignore')

    if kwargs.get('drop_on_gap', False):
        result = result.loc[~result['gap_flag']].copy()

    return result


def flag_stale_options(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Drops options that are stale (e.g., no volume or open interest)"""
    volume_threshold = kwargs.get("volume_threshold", 0)
    oi_threshold = kwargs.get("open_interest_threshold", 0)
    data = data.loc[data.isna().sum(axis=1) == 0].copy()
    if "volume" in data.columns:
        data.loc[:, 'stale'] = data["volume"] <= volume_threshold
    if "open_interest" in data.columns:
        data.loc[:, 'stale'] = data.get('stale', False) | (data["open_interest"] <= oi_threshold)
    if "bid_price" in data.columns:
        data.loc[:, 'stale'] = data.get('stale', False) | (data["bid_price"] <= 0.01)
    if "ask_price" in data.columns:
        data.loc[:, 'stale'] = data.get('stale', False) | (data["ask_price"] <= 0.01)
    return data


def compute_rv(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Computes realized volatility from historical prices"""
    import yfinance as yf
    tick = kwargs.get("tick", "")
    period = kwargs.get("period", 60)

    trade_dates = pd.to_datetime(data['trade_date'])
    start_date = trade_dates.min() - pd.Timedelta(days=60)

    stock = yf.Ticker(tick)
    hist = stock.history(start=start_date)

    hist['log_return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist['rv'] = hist['log_return'].rolling(window=period).std() * np.sqrt(252)
    hist.index = pd.to_datetime(hist.index)
    rv_series = hist['rv'].reindex(trade_dates).reset_index(drop=True)
    data['realized_volatility'] = rv_series.values

    return data


def pe_ratio(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Adds PE ratio from Yahoo Finance"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    path = os.getenv("PE_PATH", "")
    tick = kwargs.get("tick", "")
    if not tick:
        raise ValueError("pe_ratio requires a tick symbol (e.g., tick='AAPL').")

    # Only load the date column plus the requested ticker to avoid pulling a huge CSV into memory.
    try:
        csv = pd.read_csv(path, usecols=["Date", tick])
    except ValueError:
        # Column missing in file; fall back to loading just Date to raise a clear KeyError below.
        csv = pd.read_csv(path, usecols=["Date"])
    csv.index = pd.to_datetime(csv['Date'], errors='coerce')
    try:
        csv.index = csv.index.tz_localize(None)
    except TypeError:
        pass

    if tick not in csv.columns:
        raise KeyError(f"Ticker '{tick}' not found in PE CSV columns.")

    # Use underlying price to scale forward-filled PE values.
    if 'underlying_close' not in data.columns:
        raise ValueError("pe_ratio requires 'underlying_close' in data for price scaling.")

    trade_dates = pd.to_datetime(data['trade_date'], errors='coerce')
    try:
        trade_dates = trade_dates.dt.tz_localize(None)
    except TypeError:
        pass
    valid_mask = ~trade_dates.isna()

    # One underlying price per trade date.
    prices_by_date = pd.Series(
        data.loc[valid_mask, 'underlying_close'].values,
        index=trade_dates[valid_mask]
    ).groupby(level=0).first().sort_index()

    pe_raw = csv[tick]
    pe_raw = pe_raw.loc[~pe_raw.index.isna()]
    if pe_raw.index.has_duplicates:
        pe_raw = pe_raw.groupby(pe_raw.index).last().sort_index()
    else:
        pe_raw = pe_raw.sort_index()

    pe_aligned = pe_raw.reindex(prices_by_date.index)

    scaled_pe = pd.Series(index=prices_by_date.index, dtype=float)
    last_pe = np.nan
    last_price = np.nan
    for date in prices_by_date.index:
        pe_val = pe_aligned.loc[date]
        price_today = prices_by_date.loc[date]
        if not pd.isna(pe_val):
            last_pe = pe_val
            last_price = price_today
            scaled_pe.loc[date] = pe_val
        else:
            if pd.isna(last_pe) or pd.isna(price_today) or pd.isna(last_price) or last_price == 0:
                scaled_pe.loc[date] = np.nan
            else:
                scaled_pe.loc[date] = last_pe * price_today / last_price

    data['pe_ratio'] = trade_dates.map(scaled_pe)
    return data


def scale_by_splits(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Scales option prices for stock splits"""
    splits = kwargs.get("splits", {})
    if splits is None or len(splits) == 0:
        import yfinance as yf
        try:
            ticker = kwargs.get("tick", "")
            stock = yf.Ticker(ticker)
            splits = stock.splits.to_dict()
            splits = dict(sorted(splits.items()))
        except Exception:
            return data

    trade_dates = pd.to_datetime(data['trade_date'], errors='coerce')
    try:
        trade_dates_naive = trade_dates.dt.tz_localize(None)
    except TypeError:
        trade_dates_naive = trade_dates
    cols_to_scale = ['strike', 'bid_price', 'ask_price', 'last_trade_price']
    cols_to_scale = [c for c in cols_to_scale if c in data.columns]

    for date, ratio in splits.items():
        try:
            split_date = pd.to_datetime(date)
            if getattr(split_date, 'tzinfo', None) is not None:
                split_date = split_date.tz_localize(None)
            mask = trade_dates_naive < split_date
            data.loc[mask, cols_to_scale] /= ratio
        except Exception:
            pass
    return data


def underlying_skew(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Calculates the skew of the underlying over a rolling window"""
    window_size = kwargs.get("window_size", 30)
    min_periods = kwargs.get("min_periods", max(3, window_size // 2))

    if "underlying_close" not in data.columns:
        raise ValueError("underlying_skew requires 'underlying_close'. Run get_underlying first.")

    trade_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    try:
        trade_dates = trade_dates.dt.tz_localize(None)
    except TypeError:
        pass

    # Deduplicate per-day closes to avoid overweighting days with more contracts.
    daily_close = pd.Series(
        pd.to_numeric(data["underlying_close"], errors="coerce").values,
        index=trade_dates
    ).groupby(level=0).first().sort_index()

    log_returns = np.log(daily_close / daily_close.shift(1))
    daily_skew = log_returns.rolling(window=window_size, min_periods=min_periods).skew()

    # Broadcast back to the full frame by trade_date so every row on the day gets the same value.
    data = data.copy()
    data["underlying_skew"] = trade_dates.map(daily_skew)
    return data


def avg_log_return(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    In-place addition of single-metric ALR:

        ALR = mean(log returns over window)

    Works on options-level data without creating a full copy.
    """

    window = kwargs.get("window_size", 90)
    min_periods = kwargs.get("min_periods", max(30, window // 2))

    # --- validate required columns ---
    required = {"trade_date", "ticker", "underlying_close"}
    if not required.issubset(data.columns):
        raise ValueError(f"Requires columns: {required}")

    # ensure datetime
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")

    # --- one daily close per ticker ---
    daily = (
        data.groupby(["ticker", "trade_date"])["underlying_close"]
            .first()
            .unstack("ticker")
            .sort_index()
    )

    # --- compute daily log returns ---
    log_ret = np.log(daily / daily.shift(1))

    # --- rolling mean log return ---
    avg_ret = log_ret.rolling(window, min_periods=min_periods).mean()

    # --- broadcast back WITHOUT copying main frame ---
    avg_stacked = avg_ret.stack()   # index = (trade_date, ticker)

    data["avg_log_return"] = (
        data.set_index(["trade_date", "ticker"])
            .index.map(avg_stacked)
    )

    # --- cleanup ---
    del daily, log_ret, avg_ret, avg_stacked

    return data


def add_tail_selective_performance(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    In-place addition of single-metric TSP:

        TSP = median(returns) / |ES_5%|

    Works on options-level data without creating a full copy.
    """

    window = kwargs.get("window_size", 90)
    min_periods = kwargs.get("min_periods", max(30, window // 2))

    # --- validate required columns ---
    required = {"trade_date", "ticker", "underlying_close"}
    if not required.issubset(data.columns):
        raise ValueError(f"Requires columns: {required}")

    # ensure datetime
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")

    # --- get ONE daily close per ticker ---
    daily = (
        data.groupby(["ticker", "trade_date"])["underlying_close"]
            .first()
            .unstack("ticker")
            .sort_index()
    )

    # --- compute daily log returns ---
    log_ret = np.log(daily / daily.shift(1))

    # --- rolling median (typical performance) ---
    median_ret = log_ret.rolling(window, min_periods=min_periods).median()

    # --- expected shortfall (5%) for downside tail ---
    q05 = log_ret.rolling(window, min_periods=min_periods).quantile(0.05)
    es_5 = (
        log_ret.where(log_ret <= q05)
            .rolling(window, min_periods=min_periods)
            .mean()
    )

    # --- single metric ---
    tsp = median_ret / es_5.abs()

    # --- broadcast back WITHOUT copying the main frame ---
    tsp_stacked = tsp.stack()   # index = (trade_date, ticker)

    data["tsp_metric"] = data.set_index(["trade_date", "ticker"]) \
                            .index.map(tsp_stacked)

    # --- clean up to free memory ---
    del daily, log_ret, median_ret, q05, es_5, tsp, tsp_stacked

    return data


def drop_contract(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Drops a contract type"""
    drop = kwargs.get('drop', None)
    if drop:
        data = data.loc[data['call_put'] != drop.lower()].copy()
    return data


def train_test_split(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Splits data into training and testing sets based on a specified test size."""
    test_size = kwargs.get("test_size", 0.2)
    drop_out_of_sample = kwargs.get("drop_out_of_sample", False)
    drop_in_sample = kwargs.get("drop_in_sample", False)
    upper_date = kwargs.get("upper_date", None)
    data = data.sort_values('trade_date')
    unique_dates = data['trade_date'].drop_duplicates().sort_values()
    split_index = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates.iloc[split_index]
    if upper_date is not None:
        split_date = min(split_date, pd.to_datetime(upper_date))
    if drop_out_of_sample:
        data = data[data['trade_date'] < split_date]
    elif drop_in_sample:
        data = data[data['trade_date'] >= split_date]
    else:
        data['set'] = ['train' if date < split_date else 'test' for date in data['trade_date']]
    return data


def add_ttm(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Adds time to maturity (TTM) column to the data."""
    data['ttm'] = (pd.to_datetime(data['expiry_date']) - pd.to_datetime(data['trade_date'])).dt.days
    return data


def filter_ttms(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Filters options based on minimum and maximum time to maturity (TTM)."""
    min_ttm = kwargs.get("min_ttm", 0)
    max_ttm = kwargs.get("max_ttm", float('inf'))
    if 'ttm' not in data.columns:
        data = add_ttm(data, **kwargs)
    data = data[(data['ttm'] >= min_ttm) & (data['ttm'] <= max_ttm)]
    return data


def implied_volatility(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Calculates the implied volatility from the midpoint of bis and ask implied vols"""
    data['implied_volatility'] = (data['bid_implied_volatility'] + data['ask_implied_volatility']) / 2
    return data


def mid_price(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Calculates the mid price from the bid and ask prices."""
    data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2
    return data


def safe_mean(series: pd.Series) -> float:
    """Compute mean, return NaN if all values are NaN."""
    if series.dropna().empty:
        return np.nan
    return series.mean()


def pull_earnings_dates(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Fast version: adds 'last_earnings' and 'next_earnings' columns for each trade_date.
    Uses vectorized numpy search instead of merge_asof.
    """
    import yfinance as yf

    tick = kwargs.get("tick", "")
    if not tick:
        raise ValueError("You must provide a 'tick' argument (e.g., tick='AAPL').")

    # --- Fetch and prepare earnings data ---
    ticker = yf.Ticker(tick)
    earnings_dates = ticker.get_earnings_dates(limit=100).reset_index()
    earnings_series = pd.to_datetime(earnings_dates["Earnings Date"], errors='coerce')
    if getattr(earnings_series.dt, "tz", None):
        earnings_series = earnings_series.dt.tz_localize(None)
    earnings_dates = earnings_series.sort_values().to_numpy()

    # --- Prepare trade dates ---
    trade_dates = pd.to_datetime(data["trade_date"], errors='coerce')
    try:
        trade_dates = trade_dates.dt.tz_localize(None)
    except TypeError:
        pass
    trade_dates = trade_dates.to_numpy()

    # --- Find next and last earnings using searchsorted (O(n log m)) ---
    idx_next = np.searchsorted(earnings_dates, trade_dates, side="left")
    idx_last = idx_next - 1

    next_earnings = np.where(
        idx_next < len(earnings_dates),
        earnings_dates[idx_next],
        pd.NaT
    )
    last_earnings = np.where(
        idx_last >= 0,
        earnings_dates[idx_last],
        pd.NaT
    )

    # --- Attach results back to the DataFrame ---
    result = data.copy()
    result["next_earnings"] = next_earnings
    result["last_earnings"] = last_earnings

    return result


def safe_sum(series: pd.Series) -> float:
    """Compute sum, return NaN if all values are NaN."""
    if series.dropna().empty:
        return np.nan
    return series.sum()


def to_datetime(data: pd.DataFrame, columns: List[str], **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Converts specified columns to datetime format."""
    tz = kwargs.get('tz', 'America/New_York')
    for col in columns:
        converted = pd.to_datetime(data[col], errors='coerce')
        try:
            data[col] = converted.dt.tz_localize(tz)
        except TypeError:
            data[col] = converted.dt.tz_convert(tz)

        idx_name = f"{col}_idx"
        if isinstance(data.index, pd.MultiIndex) and idx_name in data.index.names:
            level_pos = data.index.names.index(idx_name)
            level_values = data.index.levels[level_pos]
            level_converted = pd.to_datetime(level_values, errors='coerce')
            try:
                level_converted = level_converted.tz_localize(tz)
            except TypeError:
                level_converted = level_converted.tz_convert(tz)
            data.index = data.index.set_levels(level_converted, level=level_pos)
        elif data.index.name == idx_name:
            index_converted = pd.to_datetime(data.index, errors='coerce')
            try:
                data.index = index_converted.tz_localize(tz)
            except TypeError:
                data.index = index_converted.tz_convert(tz)
    return data


def set_index(data: pd.DataFrame, index_cols: List[str], **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Sets the DataFrame index to the specified columns."""
    drop = kwargs.get('drop', False)  # default: keep index columns
    dedupe = kwargs.get('dedupe', False)
    if dedupe:
        data = data.drop_duplicates(subset=index_cols, keep='first')
    data = data.set_index(index_cols, drop=drop)
    data.index.names = [f"{i}_idx" for i in index_cols]
    return data

def iv_rv_ratio(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Compute the ratio of implied volatility to realized volatility."""
    iv_rv = data['implied_volatility'] / data['realized_volatility']
    return iv_rv.to_frame(name='iv_rv_ratio')


def propagate_column(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Copy one or more existing columns from the raw data into the feature outputs unchanged.
    Useful when a later stage (e.g., backtests) expects those fields to be present.
    """
    cols = kwargs.get("columns") or kwargs.get("column") or kwargs.get("feature_names")
    if cols is None:
        raise ValueError("propagate_column requires 'columns' (str or list).")
    if isinstance(cols, str):
        cols = [cols]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")
    propagated = data.loc[:, cols].copy()
    # Drop any nested/non-scalar values so downstream modeling doesn't see arrays/Series.
    mask_nested = propagated.applymap(lambda v: not pd.api.types.is_scalar(v))
    if mask_nested.any().any():
        propagated = propagated.where(~mask_nested, np.nan)
    # Standardise to numeric to avoid object columns leaking through.
    propagated = propagated.apply(pd.to_numeric, errors="coerce")
    return propagated


def mac_vol(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Daily volume MAC (EMA_short − EMA_long) / EMA_long."""
    lookback_short = kwargs.get("lookback_short", 5)
    lookback_long = kwargs.get("lookback_long", 30)

    daily_vol = (
        data.groupby("trade_date")["volume"]
            .sum()
            .sort_index()
    )

    ema_short = daily_vol.ewm(span=lookback_short, min_periods=1).mean()
    ema_long = daily_vol.ewm(span=lookback_long, min_periods=1).mean()
    mac_daily = (ema_short - ema_long) / ema_long.replace(0, np.nan)
    mac = data["trade_date"].map(mac_daily)
    mac_df = mac.to_frame(name="mac_volume")
    return mac_df


def avg_vol(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Daily rolling average volume over `lookback_days`."""
    lookback_days = kwargs.get("lookback_days", 30)

    daily_vol = (
        data.groupby("trade_date")["volume"]
            .sum()
            .sort_index()
    )

    rolling_avg = daily_vol.rolling(window=lookback_days, min_periods=1).mean()
    avg_vol = data["trade_date"].map(rolling_avg)
    avg_vol_df = avg_vol.to_frame(name="avg_volume")
    return avg_vol_df


def term_structure_slope(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the average slope of the term structure of implied volatility per day
    for options within a delta (moneyness) range.
    """

    delta_low = kwargs.get("delta_low", -0.45)
    delta_high = kwargs.get("delta_high", 0.45)
    otm_limit = kwargs.get("otm_limit", 0.05)
    near_expiry_days = kwargs.get("near_expiry_days", 14)
    far_expiry_days = kwargs.get("far_expiry_days", 30)

    subset = data[
        (delta_low <= data["delta"]) &
        (data["delta"] <= delta_high) &
        (data["delta"].abs() >= otm_limit)
    ].copy()

    def slope_for_day(df: pd.DataFrame) -> float:
        near = df[df["ttm"] <= near_expiry_days]
        far = df[df["ttm"] >= far_expiry_days]
        if near.empty or far.empty:
            return np.nan
        near_row = near.loc[near["ttm"].idxmax()]
        far_row = far.loc[far["ttm"].idxmin()]
        if near_row["ttm"] == far_row["ttm"]:
            return np.nan
        return (far_row["implied_volatility"] - near_row["implied_volatility"]) / (
            far_row["ttm"] - near_row["ttm"]
        )

    daily = (
        subset.groupby(["trade_date", "call_put"])
            .apply(slope_for_day)
            .groupby("trade_date")
            .mean()
    )

    avg_slope = data["trade_date"].map(daily)
    return avg_slope.to_frame(name="avg_slope")


def days_to_earnings(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Compute the number of business days to the next earnings date."""
    # Ensure clean date-only values
    trade_dates = pd.to_datetime(data['trade_date']).dt.normalize().values.astype('datetime64[D]')
    next_earnings = pd.to_datetime(data['next_earnings']).dt.normalize().values.astype('datetime64[D]')

    mask = ~pd.isna(data['next_earnings']).values
    if mask.any():
        bdays = np.busday_count(
            trade_dates[mask].astype('datetime64[D]'),
            next_earnings[mask].astype('datetime64[D]')
        )
        data.loc[mask, 'bdays_to_earnings'] = bdays
    else:
        data['bdays_to_earnings'] = np.nan
    return data['bdays_to_earnings']


def days_since_earnings(data: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Compute the number of business days since the last earnings date."""
    # Ensure clean date-only values
    trade_dates = pd.to_datetime(data['trade_date']).dt.normalize().values.astype('datetime64[D]')
    last_earnings = pd.to_datetime(data['last_earnings']).dt.normalize().values.astype('datetime64[D]')

    mask = ~pd.isna(data['last_earnings']).values
    if mask.any():
        bdays = np.busday_count(
            last_earnings[mask].astype('datetime64[D]'),
            trade_dates[mask].astype('datetime64[D]')
        )
        data.loc[mask, 'bdays_since_earnings'] = bdays
    else:
        data['bdays_since_earnings'] = np.nan
    return data['bdays_since_earnings']


@dataprep
def prepare_data(outputs: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Prepare data by filling missing values and sorting."""
    feature_names = kwargs.get('feature_names', [])

    theta_available = "theta_entry" in outputs.columns
    theta_ratio_col = "theta_log_ratio" if theta_available else None

    def combine_pnl(group: pd.DataFrame) -> pd.Series:
        entry_price_magnitude = group['mid_price_entry'].abs().sum()
        entry_position = (group['position'] * group['mid_price_entry']).sum()
        exit_position = (group['position'] * group['mid_price_exit']).sum()

        def _safe_log_ratio(numer: float, denom: float) -> float:
            """Return log(numer/denom) only when positive and finite."""
            try:
                ratio = numer / denom
            except Exception:
                return np.nan
            if ratio is None or not np.isfinite(ratio) or ratio <= 0:
                return np.nan
            return np.log(ratio)

        long_pnl = _safe_log_ratio(exit_position - entry_position + entry_price_magnitude, entry_price_magnitude)
        short_pnl = _safe_log_ratio(entry_price_magnitude - (exit_position - entry_position), entry_price_magnitude)
        series_dict = {
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
        }

        if theta_available:
            ratios = []
            if isinstance(group.index, pd.MultiIndex) and "call_put_idx" in group.index.names:
                pair_iter = group.groupby(level="call_put_idx", group_keys=False)
            else:
                pair_iter = [(None, group)]

            for _, pair in pair_iter:
                long_theta = pair.loc[pair['position'] > 0, 'theta_entry'].dropna()
                short_theta = pair.loc[pair['position'] < 0, 'theta_entry'].dropna()
                if long_theta.empty or short_theta.empty:
                    continue
                long_val = np.abs(long_theta.iloc[0])
                short_val = np.abs(short_theta.iloc[0])
                if long_val <= 0 or short_val <= 0:
                    continue
                ratios.append(np.log(long_val / short_val))

            series_dict[theta_ratio_col] = np.mean(ratios) if ratios else np.nan

        for feature_name in feature_names:
            if feature_name in group.columns:
                group[feature_name] = group[feature_name].fillna(0)
                series_dict[feature_name] = group[feature_name].mean()

        return pd.Series(series_dict)

    required_cols = ["mid_price_entry", "mid_price_exit", "position"]
    present_cols = [col for col in required_cols if col in outputs.columns]
    if present_cols:
        outputs = outputs.dropna(subset=present_cols)
    else:
        outputs = outputs.dropna()

    if outputs.empty:
        extra_cols = feature_names.copy()
        if theta_ratio_col:
            extra_cols.append(theta_ratio_col)
        empty_cols = ['short_pnl', 'long_pnl'] + extra_cols
        empty = pd.DataFrame(columns=empty_cols)
        empty.index.name = 'trade_date_idx'
        return empty

    aggregated = outputs.groupby('trade_date_idx', group_keys=False).apply(combine_pnl)
    aggregated.index.name = 'trade_date_idx'
    aggregated = aggregated.dropna()

    return aggregated


@dataprep
def compute_pnl(outputs: pd.DataFrame, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Prepare data by filling missing values and sorting."""
    import numpy as _np  # avoid shadowing module-level np in some contexts
    import pandas as _pd

    feature_names = kwargs.get('feature_names', [])
    initial_capital = kwargs.get('initial_capital', 1_000_000)
    capital_per_trade = kwargs.get('capital_per_trade', 0.05)
    size_leg = kwargs.get('size_leg', 'short')

    outputs = outputs.copy()
    outputs['equity'] = _np.nan
    outputs['combined_pnl'] = _np.nan

    # Initialize
    min_date = outputs.index.get_level_values('trade_date_idx').min()
    outputs.loc[outputs.index.get_level_values('trade_date_idx') == min_date, 'equity'] = initial_capital
    current_equity = initial_capital

    result_frame = _pd.DataFrame(columns=['trade_date_idx', 'combined_pnl', 'equity'] + feature_names)

    for date_idx, group in outputs.groupby('trade_date_idx'):
        # --- Determine entry value for the sizing leg ---
        if size_leg == 'short':
            leg = group.loc[group['position'] < 0]
        else:
            leg = group.loc[group['position'] > 0]

        if leg.empty:
            continue  # skip days with missing leg

        # Capital allocated to this day
        capital_today = capital_per_trade * current_equity

        # Entry value of sizing leg (should be 1 position)
        entry_value = (abs(leg['position']) * leg['mid_price_entry']).iloc[0]

        # Scale factor so total exposure = capital_today
        scale = capital_today / entry_value if entry_value != 0 else 0

        # Scale both long and short sides equally
        group['position'] *= scale

        # --- Compute daily PnL ---
        entry_value_total = (group['position'] * group['mid_price_entry']).sum()
        exit_value_total = (group['position'] * group['mid_price_exit']).sum()

        pnl = exit_value_total - entry_value_total
        ratio = (current_equity + pnl) / current_equity if current_equity != 0 else _np.nan
        combined_pnl = _np.log(ratio) if _np.isfinite(ratio) and ratio > 0 else _np.nan
        current_equity += pnl

        # Update outputs
        idx_mask = outputs.index.get_level_values('trade_date_idx') == date_idx
        outputs.loc[idx_mask, 'combined_pnl'] = combined_pnl
        outputs.loc[idx_mask, 'equity'] = current_equity

        # Collect summary row
        row = {'trade_date_idx': date_idx, 'combined_pnl': combined_pnl, 'equity': current_equity}
        for feature_name in feature_names:
            row[feature_name] = group[feature_name].mean() if feature_name in group.columns else _np.nan
        result_frame = _pd.concat([result_frame, _pd.DataFrame([row])], ignore_index=True)

    result_frame.set_index('trade_date_idx', inplace=True)
    return result_frame.dropna()
