"""
Class definitions for backtest/trade model objects
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING
from enum import Enum

from data.date import DateObj, Serializable
from data.contract import BaseUnderlying
from .base import Serializable

if TYPE_CHECKING:
    from .date import DateObj
    from backtest.trade import Trade
    from .contract import BaseUnderlying


class AccountingConvention(str, Enum):
    CASH = "cash"
    MTM = "mark_to_market"

class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"

class TransactionCostModel(str, Enum):
    NONE = "none"
    SPREAD = "spread"

@dataclass 
class PriceSeries(Serializable):
    prices: dict[str, BaseUnderlying] | None = field(default_factory=dict)

    def add(self, price_quote: BaseUnderlying) -> None:
        if price_quote is None:
            return
        if price_quote.date is None:
            return
        key = price_quote.date.to_iso()
        self.prices[key] = price_quote   

    def get(self, date: DateObj) -> BaseUnderlying | None:
        return self.prices.get(date.to_iso(), None)
    
    def __post_init__(self):
        if self.prices is None:
            self.prices = {}

        self.prices = {
            k: v for k, v in self.prices.items()
            if v is not None
        }

    @classmethod
    def from_dict(cls, d):
        prices_raw = d.get("prices", {})
        prices = {k: BaseUnderlying.from_dict(v) for k, v in prices_raw.items()}
        return cls(prices=prices)
        
@dataclass
class Equity(Serializable):
    date: DateObj
    value: float
    accounting_convention: AccountingConvention 
    parent_trade: 'Trade' | None = None
    
@dataclass
class Cashflow(Serializable):
    date: DateObj
    amount: float
    accounting_convention: AccountingConvention 
    parent_trade: 'Trade' | None = None

@dataclass 
class BaseTradeFeatures(ABC):
    pass

@dataclass
class EntryData(Serializable):
    entry_date: DateObj
    position_type: PositionType
    price_series: PriceSeries
    exit_date: DateObj
    position_size: float = 1.0
    features: BaseTradeFeatures | None = None


@dataclass(frozen=True)
class Snapshot(Serializable):
    date: DateObj
    total_equity: float
    total_cash: float
    trade_equities: Dict[Trade, Equity]

@dataclass
class BackTestResult(Serializable):
    snapshots: List[Snapshot]
    returns: List[float]
    sharpe: float
    max_drawdown: float
    volatility: float
    total_return: float
    cagr: float
