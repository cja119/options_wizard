"""
Data featuresfunctions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List

import pandas as pd
import numpy as np
import inspect
from dataclasses import dataclass

if TYPE_CHECKING:
    from .manager import DataManager

import exchange_calendars as ec


@dataclass
class TradedContract:
    expiry_date_idx: pd.Timestamp       # <-- was expiry_date_idx
    call_put_idx: int
    strike_idx: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    scaled_position: float
    last_price: float


# ---- Equity calc: cash + MTM of all active contracts ----
def calculate_equity(data, current_cash, traded_contracts):
    # `data` not needed here; we use last_price already stored on each contract
    positions_value = sum(
        c.scaled_position * c.last_price
        for c in traded_contracts
    )
    return current_cash + positions_value


class BacktestWrapper:
    def __init__(self, ticks: list[str] | str, features: Backtest):
        self.ticks: list[str] = [ticks] if isinstance(ticks, str) else ticks
        self.features: Backtest = features
        return None

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        """Queue or run features methods on the specified ticks."""
        methods: list[Callable] = []
        kwargs['_source'] = 'Backtest'

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

class Backtest:
    def __init__(self, manager: DataManager):
        self.manager: DataManager = manager
        return None

    def __getitem__(self, ticks: list[str] | str) -> BacktestWrapper:
        return BacktestWrapper(ticks, self)

    def __call__(self, *args: str, **kwargs: dict[str, any]) -> None:
        BacktestWrapper(self.manager.universe.ticks, self)(*args, **kwargs)
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
    def fixed_hold_trade(trade_entries, data, **kwargs) -> pd.DataFrame:

        # ---- Params ----
        hold_period_days = kwargs.get('hold_period', 30)
        hold_period = pd.Timedelta(days=hold_period_days)

        keys = ['expiry_date_idx', 'call_put_idx', 'strike_idx']
        entry_cost_param = kwargs.get('entry_cost_size', 'short')  # 'short', 'long', 'diff'
        initial_capital = kwargs.get('initial_capital', 100000.0)
        capital_per_trade = kwargs.get('capital_per_trade', 0.05)
        all_returns = kwargs.get('all_returns', False)

        # ---- Filter data to only tradable contracts ----
        valid_contracts = (
            trade_entries.index
            .to_frame(index=False)      # <-- no MultiIndex kept
            [keys]
            .drop_duplicates()
        )
        filtered_data = data.merge(valid_contracts, on=keys, how='inner')

        current_capital = float(initial_capital)
        last_exit = None
        outputs = {}

        # Ensure sorted by trade_date_idx
        trade_entries = trade_entries.sort_values('trade_date_idx')

        # ---- Loop over entry days ----
        for trade_date_idx, entry_group in trade_entries.groupby('trade_date_idx'):
            # Skip if still in a previous trade
            if last_exit is not None and trade_date_idx < last_exit:
                continue
            

            entry_date = entry_group['trade_date'].iloc[0]
            exit_date = entry_date + hold_period

            # Long / short slices for cost calc
            shorts = entry_group[entry_group['position'] < 0]
            longs = entry_group[entry_group['position'] > 0]

            # Capital committed to this trade
            entry_capital = current_capital * capital_per_trade

            # ---- Determine scaling factor ----
            def _cost(df):
                return (df['mid_price'] * df['position'].abs()).sum()

            if entry_cost_param == 'short':
                denom = _cost(shorts)
            elif entry_cost_param == 'long':
                denom = _cost(longs)
            elif entry_cost_param == 'diff':
                denom = _cost(longs) - _cost(shorts)
            else:
                raise ValueError("entry_cost_size must be 'short', 'long', or 'diff'.")

            if denom == 0 or np.isnan(denom):
                # nothing to scale on this day; skip
                continue

            num_contracts = entry_capital / denom

            # ---- Build scaled positions per contract ----
            contracts_today = (
                entry_group
                .reset_index()
                .assign(scaled_position=lambda df: df["position"] * num_contracts)
                .loc[:, keys + ["scaled_position"]]
            )

            # ---- Pull price paths for those contracts from entry to exit ----
            today_data = filtered_data.merge(contracts_today, on=keys, how="inner")

            mask = (today_data['trade_date'] >= entry_date) & (today_data['trade_date'] <= exit_date)
            today_data = today_data.loc[mask]

            if today_data.empty:
                continue

            # Remaining capital is held as cash
            entry_cost = (
                (entry_group['mid_price'] * entry_group['position'])
                .sum() * num_contracts
            )
            cash = current_capital - entry_cost
            _outputs = {}
            # ---- Daily path: portfolio value + log-return ----
            for date, day_df in today_data.groupby('trade_date'):
                # value of all positions at this date
                positions_value = (day_df['scaled_position'] * day_df['mid_price']).sum()
                total_value = cash + positions_value

                # track previous total_value for correct daily log returns
                if 'prev_value' not in locals() or date == entry_date:
                    prev_value = current_capital

                log_ret = np.log(total_value / prev_value)
                
                if all_returns:
                    outputs[date] = log_ret
                else:
                    _outputs[date] = log_ret

                prev_value = total_value     # update for next day
                current_capital = total_value
                last_exit = date
            
            if not all_returns:
                outputs[trade_date_idx] = sum(_outputs.values())

        # ---- Result: time series of log returns ----
        returns = pd.DataFrame.from_dict(outputs, orient='index', columns=['log_return'])
        returns.sort_index(inplace=True)
        return returns

    @staticmethod
    def multi_day_entry(trade_entries, data, **kwargs) -> pd.DataFrame:

        # ---- Params ----
        hold_period_days = kwargs.get('hold_period', 30)
        hold_period = pd.Timedelta(days=hold_period_days)

        keys = ['expiry_date_idx', 'call_put_idx', 'strike_idx']
        entry_cost_param = kwargs.get('entry_cost_size', 'short')  # 'short', 'long', 'diff'
        initial_capital = float(kwargs.get('initial_capital', 100000.0))
        capital_per_trade = kwargs.get('capital_per_trade', 0.05)

        # ---- Filter data to only tradable contracts ----
        # trade_entries index is MultiIndex with these keys
        valid_contracts = (
            trade_entries.index
            .to_frame(index=False)[keys]
            .drop_duplicates()
        )
        filtered_data = data.merge(valid_contracts, on=keys, how='inner')

        # Ensure we have the contract IDs as columns as well
        trade_entries = trade_entries.reset_index()

        # ---- Normalise dates ----
        trade_entries = trade_entries.copy()
        filtered_data = filtered_data.copy()

        norm_trade_date = pd.to_datetime(trade_entries['trade_date'], errors='coerce')
        norm_trade_date_data = pd.to_datetime(filtered_data['trade_date'], errors='coerce')
        try:
            norm_trade_date = norm_trade_date.dt.tz_localize(None)
            norm_trade_date_data = norm_trade_date_data.dt.tz_localize(None)
        except TypeError:
            # already tz-naive
            pass
        trade_entries['norm_trade_date'] = norm_trade_date
        filtered_data['norm_trade_date'] = norm_trade_date_data

        # Sort by trade_date_idx
        trade_entries = trade_entries.sort_values('trade_date_idx')

        # Calendar sessions over the union of trade/price dates
        cal = ec.get_calendar("XNAS")
        sessions = cal.sessions_in_range(
            norm_trade_date.min(),
            norm_trade_date_data.max()
        )

        # ---- State ----
        active_contracts: list[TradedContract] = []
        current_cash = initial_capital
        equity_curve = []

        # For daily returns if needed later
        prev_equity = initial_capital

        # ---- Main daily loop over sessions ----
        for current_date in sessions:
            # Convert to pandas Timestamp without tz for comparisons
            if isinstance(current_date, pd.Timestamp):
                current_date_naive = current_date.tz_localize(None) if current_date.tzinfo else current_date
            else:
                current_date_naive = pd.Timestamp(current_date)

            # -------------------------
            # 1) Handle new entries today
            # -------------------------
            todays_entries = trade_entries.loc[
                trade_entries['norm_trade_date'] == current_date_naive
            ]
            if not todays_entries.empty:
                # group in case multiple trade_date_idx map to same calendar date
                for trade_date_idx, entry_group in todays_entries.groupby('trade_date_idx'):

                    entry_date = entry_group['trade_date'].iloc[0]
                    exit_date = entry_date + hold_period

                    # Long / short slices for cost calc
                    shorts = entry_group[entry_group['position'] < 0]
                    longs = entry_group[entry_group['position'] > 0]

                    # Capital committed to this trade
                    entry_capital = calculate_equity(None, current_cash, active_contracts) * capital_per_trade

                    # ---- Determine scaling factor ----
                    def _cost(df):
                        return (df['mid_price'] * df['position'].abs()).sum()

                    if entry_cost_param == 'short':
                        denom = _cost(shorts)
                    elif entry_cost_param == 'long':
                        denom = _cost(longs)
                    elif entry_cost_param == 'diff':
                        denom = _cost(longs) - _cost(shorts)
                    else:
                        raise ValueError("entry_cost_size must be 'short', 'long', or 'diff'.")

                    if denom == 0 or np.isnan(denom):
                        # nothing to scale on this day; skip
                        continue

                    num_contracts = entry_capital / denom

                    # ---- Open new positions ----
                    # Cash impact of opening trades: cash -= Σ(q * price)
                    trade_cost_open = 0.0

                    # Shorts
                    for _, row in shorts.iterrows():
                        scaled_pos = num_contracts * row['position']  # position already negative
                        trade_cost_open += row['position'] * row['mid_price'] * num_contracts

                        active_contracts.append(
                            TradedContract(
                                expiry_date_idx=row['expiry_date_idx'],
                                call_put_idx=row['call_put_idx'],
                                strike_idx=row['strike_idx'],
                                entry_date=entry_date,
                                exit_date=exit_date,
                                scaled_position=scaled_pos,
                                last_price=float(row['mid_price'])
                            )
                        )

                    # Longs
                    for _, row in longs.iterrows():
                        scaled_pos = num_contracts * row['position']  # positive for long
                        trade_cost_open += row['position'] * row['mid_price'] * num_contracts

                        active_contracts.append(
                            TradedContract(
                                expiry_date_idx=row['expiry_date_idx'],
                                call_put_idx=row['call_put_idx'],
                                strike_idx=row['strike_idx'],
                                entry_date=entry_date,
                                exit_date=exit_date,
                                scaled_position=scaled_pos,
                                last_price=float(row['mid_price'])
                            )
                        )

                    current_cash -= trade_cost_open  # Δcash from opening trades

            # -------------------------
            # 2) Update prices for all active contracts (MTM)
            # -------------------------
            todays_prices = filtered_data.loc[
                filtered_data['norm_trade_date'] == current_date_naive,
                keys + ['mid_price']
            ]

            if not todays_prices.empty:
                # make lookup by tuple of (expiry_date_idx, call_put_idx, strike_idx)
                todays_prices = todays_prices.set_index(keys)

                for c in active_contracts:
                    idx = (c.expiry_date_idx, c.call_put_idx, c.strike_idx)
                    if idx in todays_prices.index:
                        c.last_price = float(todays_prices.loc[idx, 'mid_price'])
                    # else: keep last_price (no quote today)

            # -------------------------
            # 3) Compute equity (before closing expiring positions)
            # -------------------------

            current_equity = calculate_equity(todays_prices, current_cash, active_contracts)
            equity_curve.append(
                {
                    "date": current_date_naive,
                    "equity": current_equity,
                    "daily_log_return": np.log(current_equity / prev_equity) if prev_equity > 0 else np.nan
                }
            )
            prev_equity = current_equity


            # -------------------------
            # 4) Close positions whose exit_date is today (move value to cash)
            # -------------------------
            still_active = []
            for c in active_contracts:
                if (c.exit_date.tz_localize(None) if c.exit_date.tzinfo else c.exit_date) <= current_date_naive:
                    # Closing at today's last_price:
                    # Δq = -position, cash -= Δq * price = -( -pos ) * price = pos * price
                    current_cash += c.scaled_position * c.last_price
                    # position removed; equity unchanged, just moves to cash
                else:
                    still_active.append(c)

            active_contracts = still_active

        # ---- Return equity curve ----
        equity_df = pd.DataFrame(equity_curve).set_index("date").sort_index()
        return equity_df
