"""
Position Management Module
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, TYPE_CHECKING, Tuple

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


class PositionBase(ABC):
    def __init__(self, config: BackTestConfig) -> None:
        self._trades: list[Trade] = []
        self._sizes: dict[Trade, float] = {}
        self._start_date: DateObj = config.start_date
        self._end_date: DateObj | None = config.end_date
        self.starting_cash: float = config.starting_cash
        self._drop_trades: list[Trade] = []

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
        # -- Update entries/exits --
        entering_trades = self.trades_entering(date)
        self._sizes |= self._size_entry(entering_trades)

        # -- Process exits --
        live_trades = self._trades_on(date)
        trades_to_close, exit_cashflows = self.exit_trigger(live_trades, date)
        self._update_closes(live_trades, trades_to_close)

        # -- Update equity --
        snapshot = self._update(live_trades, date, exit_cashflows)

        return snapshot

    # ---- Abstract Methods ---- #
    @abstractmethod
    def size_function(self, entering_trades: List[Trade]) -> Dict[Trade, float]:
        pass

    @abstractmethod
    def exit_trigger(
        self, live_trades: List[Trade], current_date: DateObj
    ) -> Tuple[List[Trade], List["Cashflow"]]:
        pass

    # ---- Internal Methods ---- #
    def _update_closes(
        self, live_trades: list[Trade], trades_to_close: list[Trade]
    ) -> None:
        for trade in trades_to_close:
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
                total_cash += cashflow.amount

        trade_equities = self._snapshot.trade_equities.copy()
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
