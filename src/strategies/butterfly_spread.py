"""
Butterfly Spread Strategy
"""

import pandas as pd
import numpy as np

def butterfly_returns(data, **kwargs):
    """Entry point for finding returns from butterfly spread trades."""

    # First identify butterfly trades
    ttm = kwargs.get('ttm', 30)
    entry_data = data.copy()[data['ttm'] == ttm]
    trade_data = data.copy()[(data['ttm'] >= ttm - kwargs.get('hold_days', 15)) &
                            (data['ttm'] <= ttm)]
    
    trade_signals = identify_trades(entry_data, **kwargs)
    trade_data = fill_signals(trade_data, trade_signals)
    returns = get_butterfly_returns(trade_data, **kwargs) 
    return returns.dropna()


def fill_signals(data: pd.DataFrame, trade_signals: pd.DataFrame) -> pd.DataFrame:
    """Assign butterfly signals to original data rows."""
    data = data.copy()
    data['signal'] = np.nan

    # Use the **original index from trade_signals**
    for idx, row in trade_signals.iterrows():
        # Assign signal to all rows in data that match strike & expiry
        mask = (data['expiry_date'] == row['expiry_date']) & (data['strike'] == row['strike'])
        data.loc[mask, 'signal'] = row['signal']

    # Original index of data never changes
    return data

def _per_butterfly_returns(group: pd.DataFrame, collapse_trades: bool, short_ratio: float, margin_requirement: float) -> pd.Series:
    """Compute return for the full butterfly structure."""
    returns = pd.Series(np.nan, index=group.index, dtype=float)
    group = group.copy()
    group['trade_date'] = pd.to_datetime(group['trade_date'])
    group = group.sort_values('trade_date')

    if group['signal'].isna().all():
        return returns

    leg_data = []
    for strike, leg in group.dropna(subset=['signal']).groupby('strike'):
        leg_sorted = leg.sort_values('trade_date')
        entry_price = leg_sorted['mid_price'].iloc[0]
        exit_price = leg_sorted['mid_price'].iloc[-1]
        signal = leg_sorted['signal'].iloc[0]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        leg_data.append(
            {
                'strike': strike,
                'signal': signal,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
            }
        )

    if not leg_data:
        return returns

    pnl_total = 0.0
    long_cost = 0.0
    short_margin = 0.0
    has_long_leg = False
    has_short_leg = False
    positions: dict[float, float] = {}

    for leg in leg_data:
        base_signal = leg['signal']
        if base_signal == 0:
            continue

        qty = base_signal
        if base_signal < 0:
            qty = base_signal * short_ratio

        entry_price = leg['entry_price']
        exit_price = leg['exit_price']
        positions[leg['strike']] = qty

        if qty > 0:
            has_long_leg = True
            long_cost += entry_price * qty
        elif qty < 0:
            has_short_leg = True
            short_margin += abs(entry_price * qty) * margin_requirement
            pnl_total += short_margin

        pnl_total += (exit_price - entry_price) * qty

    if not has_long_leg or not has_short_leg:
        return returns

    initial_capital = long_cost + short_margin
    if initial_capital <= 0:
        return returns

    if collapse_trades:
        trade_return = np.log1p(pnl_total / initial_capital)
        entry_idx = (
            group.dropna(subset=['signal'])
            .sort_values('trade_date')
            .index[0]
        )
        returns.at[entry_idx] = trade_return
        return returns

    strike_mask = group['strike'].isin(positions)
    price_history = (
        group.loc[strike_mask, ['trade_date', 'strike', 'mid_price']]
        .sort_values('trade_date')
    )

    price_matrix = (
        price_history.pivot_table(
            index='trade_date',
            columns='strike',
            values='mid_price',
            aggfunc='first'
        )
        .sort_index()
        .ffill()
    )

    if price_matrix.shape[0] <= 1:
        return returns

    qty_series = pd.Series(positions)
    daily_pnl = price_matrix.diff().mul(qty_series, axis=1).sum(axis=1)
    daily_returns = np.log1p((daily_pnl)/ initial_capital)
    daily_returns.iloc[0] = np.nan

    for trade_date, value in daily_returns.dropna().items():
        idx_candidates = group.index[group['trade_date'] == trade_date]
        if len(idx_candidates):
            returns.at[idx_candidates[0]] = value

    return returns

def get_butterfly_returns(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute butterfly returns for a single group (trade_date, ttm, call_put)."""
    collapse_trades = kwargs.get('collapse_trades', False)
    short_ratio = kwargs.get('short_ratio', 2)
    margin_requirement = kwargs.get('margin_rate', 0.2)
    returns = data.groupby(['expiry_date'], group_keys=False).apply(
        _per_butterfly_returns,
        collapse_trades=collapse_trades,
        short_ratio=short_ratio,
        margin_requirement=margin_requirement
    )
    return returns

def _identify_trades_in_group(group: pd.DataFrame, lower_delta: float, upper_delta: float, short_ratio: float) -> pd.DataFrame:
    """Identify butterfly trades within a single group (trade_date)."""
    group = group.copy()
    group = group.sort_values('strike')

    lower_legs = group[group['delta'] <= lower_delta]
    upper_legs = group[group['delta'] >= upper_delta]
    middle_legs = group[(group['delta'] > lower_delta) & (group['delta'] < upper_delta)]

    # Create an output DataFrame with the same index as the group
    signals = signals = pd.DataFrame({
        'trade_date': group['trade_date'],
        'expiry_date': group['expiry_date'],
        'strike': group['strike'],
        'delta': group['delta'],
        'signal': np.nan
    }, index=group.index)

    # Identify butterfly legs
    if not lower_legs.empty and not upper_legs.empty and not middle_legs.empty:
        lower_idx = lower_legs['strike'].idxmax()
        upper_idx = upper_legs['strike'].idxmin()

        lower_delta = lower_legs.loc[lower_idx, 'delta']
        upper_delta = upper_legs.loc[upper_idx, 'delta']

        target_delta = (lower_delta + upper_delta) / short_ratio

        valid_middle_legs = middle_legs[
            (middle_legs['strike'] < lower_legs.loc[lower_idx, 'strike']) &
            (middle_legs['strike'] > upper_legs.loc[upper_idx, 'strike'])
        ]

        if not valid_middle_legs.empty:

            middle_idx = (valid_middle_legs['delta'] - target_delta).abs().idxmin()
            # Assign signals using the original DataFrame's index
            signals.loc[lower_idx, 'signal'] = 1
            signals.loc[middle_idx, 'signal'] = -1
            signals.loc[upper_idx, 'signal'] = 1

    return signals

def identify_trades(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Identify butterfly spread trades in the data."""
    delta_lower = kwargs.get('lower_delta', 0.05)
    delta_upper = kwargs.get('upper_delta', 0.95)

    signals = (
        data.groupby('trade_date', group_keys=False) 
        .apply(
            lambda group: _identify_trades_in_group(
                group,
                delta_lower,
                delta_upper,
                short_ratio=kwargs.get('short_ratio', 2)
            )
        )
    )

    # Drop rows where no signal was assigned
    signals = signals.dropna(subset=['signal'])

    return signals