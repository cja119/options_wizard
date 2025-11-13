"""
Data transformation functions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Tuple

from matplotlib import ticker
import pandas as pd
import numpy as np
import inspect

import pytz

if TYPE_CHECKING:
    from .manager import DataManager


class TransformWrapper:
    def __init__(self, ticks: list[str] | str, transform: Transformer):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.transform: Transformer = transform
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run transformation methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Transformer'

        for arg in args:
            method = getattr(self.transform, arg, None)
            if not callable(method):
                raise AttributeError(f"'{arg}' is not a valid method of Transformer")
            methods.append(method)

        manager = self.transform.manager
        if manager.load_lazy:
            for method in methods:
                manager.add_method(self.ticks, method, kwargs)
        else:
            for method in methods:
                manager.run_method(self.ticks, method, kwargs)

class Transformer:
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> TransformWrapper:
        return TransformWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        TransformWrapper(self.manager.universe.ticks, self)(*args, **kwargs)
        return None
    
    def add(self, name: str, func: Callable) -> None:
        """Dynamically add a new feature method."""
        ref_sig = inspect.signature(self.drop_stale_options)
        func_sig = inspect.signature(func)

        if ref_sig != func_sig:
            raise TypeError(
                f"Function '{func.__name__}' signature {func_sig} does not match expected {ref_sig}"
            )
        setattr(self, name, func.__get__(self))
        return None

    @staticmethod
    def get_underlying(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """
        Adds the underlying Yahoo Finance close price to each row in an options chain DataFrame.
        Does NOT modify the type or index of trade_date.
        """

        import pandas as pd
        import yfinance as yf

        try:
            ticker = kwargs.get("tick", "")
            if not ticker:
                raise ValueError("Missing required argument: 'tick' (underlying symbol)")

            # Convert trade_date just for range and merge purposes
            temp_dates = pd.to_datetime(data['trade_date'], errors='coerce')
            min_date, max_date = temp_dates.min(), temp_dates.max()

            if pd.isna(min_date) or pd.isna(max_date):
                raise ValueError("No valid trade_date values found in input data.")

            # --- Download underlying data ---
            close_prices = yf.download(
                ticker,
                start=min_date - pd.Timedelta(days=2),
                end=max_date + pd.Timedelta(days=2),
                progress=False,
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
            data["_merge_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
            
            # --- Merge underlying close prices ---
            merged = pd.merge(
                data,
                close_df,
                left_on="_merge_date",
                right_on="merge_date",
                how="left"
            ).drop(columns=["_merge_date", "merge_date"])
            return merged

        except Exception as e:
            raise RuntimeError(f"Failed to get underlying data for {kwargs.get('tick', '')}: {e}")

    @staticmethod
    def flag_chain_gaps(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """
        Flags gaps in option chains and returns the list of missing dates per contract.
        """

        import pytz
        import exchange_calendars as ec

        gap_threshold: int = kwargs.get("gap_threshold", 1)
        # Remove timezone for consistency
        data['norm_trade_date'] = pd.to_datetime(data['trade_date']).dt.tz_localize(None)

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

        # Merge back with original data
        result = data.merge(
            missing_df,
            on=['expiry_date', 'strike', 'call_put'],
            how='left'
        )
        result.drop(columns=['norm_trade_date'], inplace=True)

        return result

            
    @staticmethod
    def flag_stale_options(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Drops options that are stale (e.g., no volume or open interest)"""
        volume_threshold = kwargs.get("volume_threshold", 0)
        oi_threshold = kwargs.get("open_interest_threshold", 0)
        data = data[data.isna().sum(axis=1) == 0]
        if "volume" in data.columns:
            data['stale'] = data["volume"] <= volume_threshold
        if "open_interest" in data.columns:
            data['stale'] = data.get('stale', False) | (data["open_interest"] <= oi_threshold)
        if "bid_price" in data.columns:
            data['stale'] = data.get('stale', False) | (data["bid_price"] <= 0.01)
        if "ask_price" in data.columns:
            data['stale'] = data.get('stale', False) | (data["ask_price"] <= 0.01)
        return data

    @staticmethod
    def compute_rv(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
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
    
    @staticmethod
    def scale_by_splits(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Scales option prices for stock splits"""
        splits = kwargs.get("splits", {})
        if splits is None or len(splits) == 0:
            import yfinance as yf
            try:
                ticker = kwargs.get("tick", "")
                stock = yf.Ticker(ticker)
                splits = stock.splits.to_dict()
                splits = dict(sorted(splits.items()))
            except Exception as e:
                return data

        for date, ratio in splits.items():
            try:
                split_date = pd.to_datetime(date)
                mask = pd.to_datetime(data['trade_date']) < split_date
                cols_to_scale = ['strike', 'bid_price', 'ask_price', 'last_trade_price']
                cols_to_scale = [c for c in cols_to_scale if c in data.columns]
                data.loc[mask, cols_to_scale] /= ratio
                if 'underlying_close' in data.columns:
                    data.loc[mask, 'underlying_close'] /= ratio
            except Exception as e:
                print(f"Skipping split on {date}: {e}")
        return data
    
    @staticmethod
    def drop_contract(data, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Drops a contract type"""
        drop = kwargs.get('drop', None)
        if drop:
            data = data[data['call_put']!=drop.lower()]
        return data
    
    @staticmethod
    def train_test_split(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Splits data into training and testing sets based on a specified test size."""
        test_size = kwargs.get("test_size", 0.2)
        drop_out_of_sample = kwargs.get("drop_out_of_sample", False)
        drop_in_sample = kwargs.get("drop_in_sample", False)
        data = data.sort_values('trade_date')
        unique_dates = data['trade_date'].drop_duplicates().sort_values()
        split_index = int(len(unique_dates) * (1 - test_size))
        split_date = unique_dates.iloc[split_index]
        if drop_out_of_sample:
            data = data[data['trade_date'] < split_date]
        elif drop_in_sample:
            data = data[data['trade_date'] >= split_date]
        else:
            data['set'] = ['train' if date < split_date else 'test' for date in data['trade_date']]        
        return data

    @staticmethod
    def add_ttm(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Adds time to maturity (TTM) column to the data."""
        data['ttm'] = (pd.to_datetime(data['expiry_date']) - pd.to_datetime(data['trade_date'])).dt.days
        return data
    
    @staticmethod
    def filter_ttms(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Filters options based on minimum and maximum time to maturity (TTM)."""
        min_ttm = kwargs.get("min_ttm", 0)
        max_ttm = kwargs.get("max_ttm", float('inf'))
        if 'ttm' not in data.columns:
            data = Transformer.add_ttm(data, **kwargs)
        data = data[(data['ttm'] >= min_ttm) & (data['ttm'] <= max_ttm)]
        return data
    
    @staticmethod
    def implied_volatility(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Calculates the implied volatility from the midpoint of bis and ask implied vols"""
        data['implied_volatility'] = (data['bid_implied_volatility'] + data['ask_implied_volatility']) / 2
        return data
    
    @staticmethod
    def mid_price(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Calculates the mid price from the bid and ask prices."""
        data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2
        return data
    
    @staticmethod
    def safe_mean(series: pd.Series) -> float:
        """Compute mean, return NaN if all values are NaN."""
        if series.dropna().empty:
            return np.nan
        return series.mean()

    @staticmethod
    def pull_earnings_dates(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
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
        earnings_dates = pd.to_datetime(earnings_dates["Earnings Date"]).sort_values().to_numpy()

        # --- Prepare trade dates ---
        trade_dates = pd.to_datetime(data["trade_date"]).to_numpy()

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

    @staticmethod
    def safe_sum(series: pd.Series) -> float:
        """Compute sum, return NaN if all values are NaN."""
        if series.dropna().empty:
            return np.nan
        return series.sum()
    
    @staticmethod
    def to_datetime(data: pd.DataFrame, columns: List[str], **kwargs: dict[str, any]) -> pd.DataFrame:
        """Converts specified columns to datetime format."""
        tz = kwargs.get('tz', 'America/New_York')
        for col in columns:
            data[col] = pd.to_datetime(data[col]).dt.tz_localize(tz)
        return data
    
    @staticmethod
    def set_index(data: pd.DataFrame, index_cols: List[str], **kwargs: dict[str, any]) -> pd.DataFrame:
        """Sets the DataFrame index to the specified columns."""
        drop = kwargs.get('drop', False)  # default: keep index columns
        dedupe = kwargs.get('dedupe', False)
        if dedupe:
            data = data.drop_duplicates(subset=index_cols, keep='first')
        data = data.set_index(index_cols, drop=drop)
        data.index.names = [f"{i}_idx" for i in index_cols]
        return data
