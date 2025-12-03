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
    tick: str
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

        self.prices = {k: v for k, v in self.prices.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "PriceSeries":
        from .contract import Option, Spot, Future, OptionType

        prices_raw = d.get("prices") or {}
        tick = d.get("tick", "")
        out = {}
        for k, v in prices_raw.items():
            if v is None:
                continue
            utag = v.get("underlying_type") or v.get("underlying_type".encode(), None)
            if isinstance(utag, (bytes, bytearray)):
                try:
                    import pickle
                    utag = pickle.loads(utag)
                except Exception:
                    utag = None
            utag = str(utag).lower() if utag is not None else None
            if utag and "Option".lower() in utag:
                out[k] = Option(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=DateObj.from_iso(v["date"]) if isinstance(v["date"], str) else v["date"],
                    tick=v.get("tick", tick),
                    option_type=OptionType(v["option_type"]),
                    strike=v["strike"],
                    expiry=DateObj.from_iso(v["expiry"]) if isinstance(v["expiry"], str) else v["expiry"],
                    iv=v.get("iv"),
                    underlying=_decode_underlying(v.get("underlying"), tick),
                    rfr=v.get("rfr"),
                    delta=v.get("delta"),
                    gamma=v.get("gamma"),
                    vega=v.get("vega"),
                    theta=v.get("theta"),
                    rho=v.get("rho"),
                )
            elif utag and "Spot".lower() in utag:
                out[k] = Spot(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=DateObj.from_iso(v["date"]) if isinstance(v["date"], str) else v["date"],
                    tick=v.get("tick", tick),
                )
            elif utag and "Future".lower() in utag:
                out[k] = Future(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=DateObj.from_iso(v["date"]) if isinstance(v["date"], str) else v["date"],
                    tick=v.get("tick", tick),
                )
            else:
                # fallback minimal Spot
                out[k] = Spot(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=DateObj.from_iso(v["date"]) if isinstance(v["date"], str) else v["date"],
                    tick=v.get("tick", tick),
                )
        return cls(prices=out, tick=tick)


@dataclass
class Equity(Serializable):
    date: DateObj
    value: float
    accounting_convention: AccountingConvention
    parent_trade: "Trade" | None = None


@dataclass
class Cashflow(Serializable):
    date: DateObj
    amount: float
    accounting_convention: AccountingConvention
    parent_trade: "Trade" | None = None


@dataclass
class BaseTradeFeatures(ABC):
    pass


@dataclass
class EntryData(Serializable):
    entry_date: DateObj
    position_type: PositionType
    price_series: PriceSeries
    exit_date: DateObj
    tick: str
    position_size: float = 1.0
    features: BaseTradeFeatures | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "EntryData":
        ed = d.get("entry_date")
        xd = d.get("exit_date")
        return cls(
            entry_date=DateObj.from_iso(ed) if isinstance(ed, str) else ed,
            position_type=PositionType(d["position_type"]),
            price_series=PriceSeries.from_dict(d["price_series"]),
            exit_date=DateObj.from_iso(xd) if isinstance(xd, str) else xd,
            tick=d["tick"],
            position_size=d.get("position_size", 1.0),
            features=None if d.get("features") is None else d["features"],  # or decode if needed
        )


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
    dates: List[DateObj]

def _decode_underlying(u, tick):
    from .contract import Spot, Future, BaseUnderlying
    if u is None:
        return None
    # if already decoded, return
    if isinstance(u, BaseUnderlying):
        return u
    utag = str(u.get("underlying_type", "")).lower()
    date = u.get("date")
    date = DateObj.from_iso(date) if isinstance(date, str) else date
    if "Spot".lower() in utag:
        return Spot(bid=u["bid"], ask=u["ask"], volume=u.get("volume", 0.0), date=date, tick=u.get("tick", tick))
    if "Future".lower() in utag:
        return Future(bid=u["bid"], ask=u["ask"], volume=u.get("volume", 0.0), date=date, tick=u.get("tick", tick))
    return None