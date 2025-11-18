"""
Data featuresfunctions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List

import pandas as pd
import numpy as np
import inspect

if TYPE_CHECKING:
    from .manager import DataManager

def dataprep(method: Callable) -> Callable:
    """Decorator to mark a feature method as a data preparation step."""
    method._is_dataprep = True
    return method

class FeaturesWrapper:
    def __init__(self, ticks: list[str] | str, features: Features):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.features: Features = features
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run features methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Features'

        for arg in args:
            method = getattr(self.features, arg, None)
            if not callable(method):
                raise AttributeError(f"'{arg}' is not a valid feature")
            methods.append(method)

        manager = self.features.manager
        if manager.load_lazy:
            for method in methods:
                manager.add_method(self.ticks, method, kwargs)
        else:
            for method in methods:
                manager.run_method(self.ticks, method, kwargs)

class Features:
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> FeaturesWrapper:
        return FeaturesWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        FeaturesWrapper(self.manager.universe.ticks, self)(*args, **kwargs)
        return None
    
    def add(self, name: str, func: Callable) -> None:
        """Dynamically add a new feature method."""
        ref_sig = inspect.signature(self.iv_rv_ratio)
        func_sig = inspect.signature(func)

        if ref_sig != func_sig:
            raise TypeError(
                f"Function '{func.__name__}' signature {func_sig} does not match expected {ref_sig}"
            )
        setattr(self, name, func.__get__(self))
        return None

    @staticmethod
    def iv_rv_ratio(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the ratio of implied volatility to realized volatility."""
        iv_rv_ratio = data['implied_volatility'] / data['realized_volatility']
        iv_rv_ratio.name = 'iv_rv_ratio'
        return iv_rv_ratio
    
    @staticmethod
    def mac_vol(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
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

    @staticmethod
    def avg_vol(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

    @staticmethod
    def term_structure_slope(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
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

        def slope_for_day(df):
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
    
    @staticmethod
    def days_to_earnings(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the number of days to the next earnings date."""
        # Ensure clean date-only values
        trade_dates = pd.to_datetime(data['trade_date']).dt.normalize().values.astype('datetime64[D]')
        next_earnings = pd.to_datetime(data['next_earnings']).dt.normalize().values.astype('datetime64[D]')
        mask = next_earnings.notna()
        data.loc[mask, 'bdays_to_earnings'] = np.busday_count(
                trade_dates[mask].values.astype('datetime64[D]'),
                next_earnings[mask].values.astype('datetime64[D]')
            )
        return data['bdays_to_earnings']

    @staticmethod
    def days_since_earnings(data: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Compute the number of days since the last earnings date."""
        # Ensure clean date-only values
        trade_dates = pd.to_datetime(data['trade_date']).dt.normalize().values.astype('datetime64[D]')
        last_earnings = pd.to_datetime(data['last_earnings']).dt.normalize().values.astype('datetime64[D]')

        mask = last_earnings.notna()
        data.loc[mask, 'bdays_since_earnings'] = np.busday_count(
                last_earnings[mask].values.astype('datetime64[D]'),
                trade_dates[mask].values.astype('datetime64[D]')
            )
        return data['bdays_since_earnings']

    @staticmethod
    @dataprep
    def prepare_data(outputs: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Prepare data by filling missing values and sorting."""
        feature_names = kwargs.get('feature_names', [])

        theta_available = "theta_entry" in outputs.columns
        theta_ratio_col = "theta_log_ratio" if theta_available else None

        def combine_pnl(group: pd.DataFrame) -> pd.Series:
            entry_price_magnitude = group['mid_price_entry'].abs().sum()
            entry_position = (group['position'] * group['mid_price_entry']).sum()
            exit_position = (group['position'] * group['mid_price_exit']).sum()
            
            long_pnl = np.log((exit_position - entry_position + entry_price_magnitude) / entry_price_magnitude)
            short_pnl = np.log((entry_price_magnitude - (exit_position - entry_position)) / entry_price_magnitude)
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

    @staticmethod
    @dataprep
    def compute_pnl(outputs: pd.DataFrame, **kwargs: dict[str, any]) -> pd.DataFrame:
        """Prepare data by filling missing values and sorting."""
        import numpy as np
        import pandas as pd

        feature_names = kwargs.get('feature_names', [])
        initial_capital = kwargs.get('initial_capital', 1_000_000)
        capital_per_trade = kwargs.get('capital_per_trade', 0.05)
        size_leg = kwargs.get('size_leg', 'short')

        outputs = outputs.copy()
        outputs['equity'] = np.nan
        outputs['combined_pnl'] = np.nan

        # Initialize
        min_date = outputs.index.get_level_values('trade_date_idx').min()
        outputs.loc[outputs.index.get_level_values('trade_date_idx') == min_date, 'equity'] = initial_capital
        current_equity = initial_capital

        result_frame = pd.DataFrame(columns=['trade_date_idx', 'combined_pnl', 'equity'] + feature_names)

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
            combined_pnl = np.log((current_equity + pnl) / current_equity)
            current_equity += pnl

            # Update outputs
            idx_mask = outputs.index.get_level_values('trade_date_idx') == date_idx
            outputs.loc[idx_mask, 'combined_pnl'] = combined_pnl
            outputs.loc[idx_mask, 'equity'] = current_equity

            # Collect summary row
            row = {'trade_date_idx': date_idx, 'combined_pnl': combined_pnl, 'equity': current_equity}
            for feature_name in feature_names:
                row[feature_name] = group[feature_name].mean() if feature_name in group.columns else np.nan
            result_frame = pd.concat([result_frame, pd.DataFrame([row])], ignore_index=True)

        result_frame.set_index('trade_date_idx', inplace=True)
        return result_frame.dropna()

   