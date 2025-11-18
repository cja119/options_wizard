"""
Data featuresfunctions
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import pandas as pd
import numpy as np
import inspect

from . import backtest_engine as backtest_engine

if TYPE_CHECKING:
    from .manager import DataManager

# Re-export dataclasses for backward compatibility
TradedContract = backtest_engine.TradedContract
Spread = backtest_engine.Spread
SubStrategy = backtest_engine.SubStrategy

def all_stocks(method: Callable) -> Callable:
    """Decorator to flag methods that should run once using all stocks."""
    target = method.__func__ if isinstance(method, staticmethod) else method
    setattr(target, "_all_stocks", True)
    return method

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

        # Ensure sorted by trade_date_idx
        trade_entries = trade_entries.sort_values('trade_date_idx')

        def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
            ts = pd.to_datetime(ts)
            return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts

        sequential_entries: list[tuple[int, pd.Timestamp]] = []
        selected_groups: list[pd.DataFrame] = []
        next_entry_allowed: pd.Timestamp | None = None

        for trade_date_idx, entry_group in trade_entries.groupby('trade_date_idx'):
            entry_date = _to_naive(entry_group['trade_date'].iloc[0])
            if next_entry_allowed is not None and entry_date < next_entry_allowed:
                continue
            sequential_entries.append((trade_date_idx, entry_date))
            selected_groups.append(entry_group)
            next_entry_allowed = entry_date + hold_period

        if not selected_groups:
            return pd.DataFrame(columns=['log_return'])

        filtered_entries = pd.concat(selected_groups).sort_values('trade_date_idx')

        equity_df = backtest_engine.main(
            filtered_entries,
            data,
            hold_period=hold_period_days,
            initial_capital=float(initial_capital),
            capital_per_trade=float(capital_per_trade),
            entry_cost_size=entry_cost_param,
            sizing_fn=kwargs.get('sizing_fn'),
            exit_fn=kwargs.get('exit_fn'),
            calendar=kwargs.get('calendar', 'XNAS'),
        )

        if equity_df.empty:
            return pd.DataFrame(columns=['log_return'])

        log_returns = equity_df['daily_log_return']

        if all_returns:
            return log_returns.to_frame(name='log_return')

        trade_returns: dict[int, float] = {}
        for trade_date_idx, entry_date in sequential_entries:
            exit_date = entry_date + hold_period
            mask = (log_returns.index >= entry_date) & (log_returns.index <= exit_date)
            if not mask.any():
                continue
            trade_returns[trade_date_idx] = log_returns.loc[mask].sum()

        return (
            pd.Series(trade_returns, name='log_return')
            .to_frame()
            .sort_index()
        )

    @staticmethod
    def multi_day_entry(trade_entries, data, **kwargs) -> pd.DataFrame:
        return backtest_engine.main(
            trade_entries,
            data,
            hold_period=kwargs.get('hold_period', 30),
            initial_capital=float(kwargs.get('initial_capital', 100000.0)),
            capital_per_trade=float(kwargs.get('capital_per_trade', 0.05)),
            entry_cost_size=kwargs.get('entry_cost_size', 'short'),
            sizing_fn=kwargs.get('sizing_fn'),
            exit_fn=kwargs.get('exit_fn'),
            calendar=kwargs.get('calendar', 'XNAS'),
        )

    @all_stocks
    @staticmethod
    def combined_multiday_entry(per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        # Backwards compatibility: defer to multi_stock implementation.
        return Backtest.multi_stock(per_stock_equity, **kwargs)

    @all_stocks
    @staticmethod
    def multi_stock(per_stock_equity: dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        if not isinstance(per_stock_equity, dict):
            raise TypeError("multi_stock expects a mapping of ticker -> equity DataFrame.")
        capital_fraction = float(kwargs.get('capital_per_trade', kwargs.get('capital_per_day', 0.05)))
        return backtest_engine.combine_multi_equity(
            per_stock_equity,
            initial_capital=float(kwargs.get('initial_capital', 100_000.0)),
            capital_per_trade=capital_fraction,
            capital_allocation=kwargs.get('capital_allocation'),
            entry_cost_size=kwargs.get('entry_cost_size', 'short'),
            top_per_date=kwargs.get('top_per_date'),
        )



