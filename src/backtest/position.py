"""
Position Management Module
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING, Tuple, Set

from data.trade import Cashflow, Snapshot
from backtest.trade import Trade

if TYPE_CHECKING:
    from data.date import DateObj
    from .trade import Trade


@dataclass
class BackTestConfig(ABC):
    starting_cash: float
    start_date: DateObj
    end_date: DateObj | None = None
    kwargs: Dict = field(default_factory=dict)


class PositionBase(ABC):
    def __init__(self, config: BackTestConfig) -> None:
        self._trades: list[Trade] = []
        self._sizes: dict[Trade, float] = {}
        self._start_date: DateObj = config.start_date
        self._end_date: DateObj | None = config.end_date
        self.starting_cash: float = config.starting_cash
        self._drop_trades: list[Trade] = []
        self._kwargs = config.kwargs
        self._active: list[Trade] = []
        self._entries: list[Trade] = []
        self._exits: list[Trade] = []
        self._entry_idx = 0
        self._exit_idx = 0
        self._schedule_ready = False

        self._snapshot = Snapshot(
            date=config.start_date,
            total_equity=0.0,
            total_cash=config.starting_cash,
            trade_equities={},
        )

    # ---- External Interface ---- #
    def add_trade(self, trade: "Trade" | List[Trade]) -> None:
        if isinstance(trade, Trade):
            self._trades.append(trade)
        else:
            self._trades.extend(trade)

    def __call__(self, date: DateObj) -> Snapshot:
        if not self._schedule_ready: self._finalize_schedule()
        
        entering, scheduled_exit = self._advance_schedule(date)
        self._sizes |= self._size_entry(entering)

        trades_to_close, exit_cf = self.exit_trigger(self._active, date, scheduled_exit)

        self._update_closes(self._active, trades_to_close)
        snapshot = self._update(self._active, date, exit_cf)
        
        return snapshot

    # ---- Abstract Methods ---- #
    @abstractmethod
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:
        pass

    @abstractmethod
    def exit_trigger(
        self, live_trades: List[Trade], current_date: DateObj, scheduled_exits: Set[Trade]
    ) -> Tuple[List[Trade], List["Cashflow"]]:
        pass

    # ---- Internal Methods ---- #
    def _advance_schedule(self, date):
        key = (date.year, date.month, date.day)
        entering, scheduled_exit = [], []
        while self._entry_idx < len(self._entries) and self._entries[self._entry_idx]._entry_key == key:
            t = self._entries[self._entry_idx]; self._active.append(t); entering.append(t); self._entry_idx += 1
        while self._exit_idx < len(self._exits) and self._exits[self._exit_idx]._exit_key <= key:
            t = self._exits[self._exit_idx]
            if t in self._active and not getattr(t, "_closed", False):
                scheduled_exit.append(t)
            self._exit_idx += 1
        return entering, scheduled_exit

    def _update_closes(
        self, live_trades: list[Trade], trades_to_close: Set[Trade]
    ) -> None:
        for trade in trades_to_close:
                if trade in live_trades:
                    live_trades.remove(trade)
        self._drop_trades = trades_to_close

    def _size_entry(self, trades: List[Trade]) -> Dict[Trade, float]:
        sizes = self.size_function(trades)
        for trade in trades:
            trade *= sizes[trade]
        return sizes

    def _trades_on(self, date: DateObj) -> list[Trade]:
        trades = [trade for trade in self._trades if trade.is_open_on(date)]
        return trades

    def trades_entering(self, date: DateObj) -> list[Trade]:
        trades = [trade for trade in self._trades if trade.is_entering_on(date)]
        return trades

    def _update(
        self,
        live_trades: list[Trade],
        date: DateObj,
        exit_cashflows: List[Cashflow] | None = None,
    ) -> Snapshot:

        total_cash = self._snapshot.total_cash
        if exit_cashflows is not None:
            for cashflow in exit_cashflows:
                if cashflow is None:
                    continue
                total_cash += cashflow.amount

        trade_equities = self._snapshot.trade_equities.copy()
        # Remove any trades already marked closed to avoid stale equity lingering
        for closed_trade in list(trade_equities.keys()):
            if getattr(closed_trade, "_closed", False):
                trade_equities.pop(closed_trade, None)
        # Drop trades that were closed on this tick so they don't linger in snapshots
        for closed_trade in self._drop_trades:
            trade_equities.pop(closed_trade, None)
        total_equity = 0.0

        for trade in live_trades:
            if trade in self._drop_trades:
                continue

            equity, cashflow = trade(date)

            if cashflow is not None:
                total_cash += cashflow.amount

            if equity is not None:
                trade_equities[trade] = equity
                total_equity += equity.value

        snapshot = Snapshot(
            date=date,
            total_equity=total_equity,
            total_cash=total_cash,
            trade_equities=trade_equities,
        )

        self._snapshot = snapshot
        return snapshot

    def _finalize_schedule(self):
        def key(d): return (d.year, d.month, d.day)

        # Drop trades that start before the backtest window to avoid blocking the schedule
        filtered_trades = []
        for t in self._trades:
            if t.entry_data.entry_date < self._start_date:
                continue
            if self._end_date is not None and t.entry_data.entry_date > self._end_date:
                continue
            filtered_trades.append(t)
        self._trades = filtered_trades

        for t in self._trades:
            t._entry_key = key(t.entry_data.entry_date)
            t._exit_key = key(t.entry_data.exit_date)

        self._entries = sorted(self._trades, key=lambda t: t._entry_key)
        self._exits = sorted(self._trades, key=lambda t: t._exit_key)
        self._entry_idx = 0
        self._exit_idx = 0
        self._schedule_ready = True
