"""
Class definitions for backtest/trade model objects
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .date import DateObj
    from backtest.trade import Trade
    from .contract import BaseUnderlying


class AccountingConvention(Enum):
    CASH = "cash"
    MTM = "mark_to_market"

class PositionType(Enum):
    LONG = "long"
    SHORT = "short"

class TransactionCostModel(Enum):
    NONE = "none"
    SPREAD = "spread"

@dataclass 
class PriceSeries:
    prices: dict[DateObj, BaseUnderlying] = field(default_factory=dict)

    def add(self, price_quote: BaseUnderlying) -> None:
        self.prices[price_quote.date] = price_quote
        
@dataclass
class Equity:
    date: DateObj
    value: float
    accounting_convention: AccountingConvention 
    parent_trade: 'Trade' | None = None

@dataclass
class Cashflow:
    date: DateObj
    amount: float
    accounting_convention: AccountingConvention 
    parent_trade: 'Trade' | None = None

@dataclass 
class BaseTradeFeatures(ABC):
    pass

@dataclass
class EntryData:
    entry_date: DateObj
    position_type: PositionType
    price_series: PriceSeries
    exit_date: DateObj
    position_size: float = 1.0
    features: BaseTradeFeatures | None = None

@dataclass(frozen=True)
class Snapshot:
    date: DateObj
    total_equity: float
    total_cash: float
    trade_equities: Dict[Trade, Equity]

@dataclass
class BackTestResult:
    snapshots: List[Snapshot]
    returns: List[float]
    sharpe: float
    max_drawdown: float
    volatility: float
    total_return: float
    cagr: float

    