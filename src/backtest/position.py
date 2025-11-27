"""
Position Management Module
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
from .trade import DateObj, Trade, Equity, CashFlow

@dataclass(frozen=True)
class Snapshot:
    date: DateObj
    total_equity: float
    total_cash: float
    trade_equities: Dict[Trade, Equity]
        
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

        self._snapshot = Snapshot(
            date=config.start_date,
            total_equity=0.0,
            total_cash=config.starting_cash,
            trade_equities={}
        )
        
    # ---- External Interface ---- #
    def add_trade(self, trade: 'Trade' | List[Trade]) -> None:
        if isinstance(trade, Trade):
            self._trades.append(trade)
        else:
            self._trades.extend(trade)

    def __call__(self, date: DateObj) -> Snapshot:
        # First update entries/exits
        entering_trades = self.trades_entering(date)
        self._sizes |= self._size_entry(entering_trades)

        # Then process exits
        live_trades = self._trades_on(date)
        live_trades = self.exit_trigger(live_trades)

        # Finally update equity#
        snapshot = self._update(live_trades, date)

        return snapshot

    # ---- Abstract Methods ---- #
    @abstractmethod
    def size_function(self, trade: Trade) -> float:
        pass

    @abstractmethod
    def exit_trigger(self, trades: List[Trade]) -> List[Trade]:
        pass
    
    # ---- Internal Methods ---- #
    def _size_entry(self, trades: List[Trade]) -> Dict[Trade, float]:
        sizes = {}
        for trade in trades:
            sizes[trade] = self.size_function(trade)
            trade *= sizes[trade]
        return sizes

    def _trades_on(self, date: DateObj) -> list[Trade]:
        trades = [trade for trade in self._trades if trade.is_open_on(date)]
        return trades
    
    def trades_entering(self, date: DateObj) -> list[Trade]:
        trades = [trade for trade in self._trades if trade.is_entering_on(date)]
        return trades

    def _update(self, live_trades: list[Trade], date: DateObj) -> Snapshot:

        total_cash = self._snapshot.total_cash
        trade_equities = self._snapshot.trade_equities.copy()
        total_equity = 0.0

        for trade in live_trades:
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
            
    