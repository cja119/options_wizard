"""
Class definitions for backtest/trade model objects
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, fields, MISSING
from typing import List, Dict, TYPE_CHECKING, Any, Tuple, Type
from enum import Enum
import logging
import pickle

import numpy as np

from options_wizard.data.date import DateObj, Serializable
from options_wizard.data.contract import BaseUnderlying
from .base import Serializable


if TYPE_CHECKING:
    from .date import DateObj
    from options_wizard.backtest.trade import Trade
    from .contract import BaseUnderlying

class PositionType(int, Enum):
    LONG = 1
    SHORT = -1

class AccountingConvention(str, Enum):
    CASH = "cash"
    MTM = "mark_to_market"


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

        self.log_debug(tick=self.tick,  prefix="Price series debug: ")

    @classmethod
    def from_dict(cls, d: dict) -> "PriceSeries":
        from .contract import Option, Spot, Future, IntraDayPerf, OptionType

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

                other_data = v.get("other")
                if other_data is not None:
                    other_data = Serializable._decode(other_data, Dict[str, Any])

                out[k] = Option(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=(
                        DateObj.from_iso(v["date"])
                        if isinstance(v["date"], str)
                        else v["date"]
                    ),
                    tick=v.get("tick", tick),
                    option_type=OptionType(v["option_type"]),
                    strike=v["strike"],
                    expiry=(
                        DateObj.from_iso(v["expiry"])
                        if isinstance(v["expiry"], str)
                        else v["expiry"]
                    ),
                    iv=v.get("iv"),
                    underlying=_decode_underlying(v.get("underlying"), tick),
                    rfr=v.get("rfr"),
                    delta=v.get("delta"),
                    gamma=v.get("gamma"),
                    vega=v.get("vega"),
                    theta=v.get("theta"),
                    rho=v.get("rho"),
                    other=other_data,
                )
            elif utag and "Spot".lower() in utag:
                out[k] = Spot(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=(
                        DateObj.from_iso(v["date"])
                        if isinstance(v["date"], str)
                        else v["date"]
                    ),
                    tick=v.get("tick", tick),
                )
            elif utag and "Future".lower() in utag:
                out[k] = Future(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=(
                        DateObj.from_iso(v["date"])
                        if isinstance(v["date"], str)
                        else v["date"]
                    ),
                    tick=v.get("tick", tick),
                    expiry=(
                        DateObj.from_iso(v["expiry"])
                        if isinstance(v["expiry"], str)
                        else v["expiry"]
                    ),
                    settlement_price=v.get("settlement_price"),
                    contract_id=v.get("contract_id"),
                    contract_multiplier=v.get("contract_multiplier"),
                    open_interest=v.get("open_interest"),
                    
                )
            elif utag and ("intradayperf" in utag or "intraday_perf" in utag):
                out[k] = IntraDayPerf(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=(
                        DateObj.from_iso(v["date"])
                        if isinstance(v["date"], str)
                        else v["date"]
                    ),
                    tick=v.get("tick", tick),
                )
            else:
                # fallback minimal Spot
                out[k] = Spot(
                    bid=v["bid"],
                    ask=v["ask"],
                    volume=v.get("volume", 0.0),
                    date=(
                        DateObj.from_iso(v["date"])
                        if isinstance(v["date"], str)
                        else v["date"]
                    ),
                    tick=v.get("tick", tick),
                )
        return cls(prices=out, tick=tick)


@dataclass
class Equity(Serializable):
    date: DateObj
    value: float
    accounting_convention: AccountingConvention
    parent_trade: "Trade" | None = None

    def __post_init__(self):
        self.log_debug(tick=self.parent_trade._tick,  prefix="Equity debug: ")


@dataclass
class Cashflow(Serializable):
    date: DateObj
    amount: float
    accounting_convention: AccountingConvention
    parent_trade: "Trade" | None = None

    def __post_init__(self):
        self.log_debug(tick=self.parent_trade._tick,  prefix="Cashflow debug: ")

class CarryRankingFeature(str, Enum):
    RAW_CARRY = "raw_carry"
    SMOOTHED_CARRY = "smoothed_carry"
    RAW_RELATIVE_CARRY = "raw_relative_carry"
    SMOOTHED_RELATIVE_CARRY = "smoothed_relative_carry"

@dataclass
class BaseTradeFeatures(Serializable, ABC):

    def __post_init__(self):
        self.log_debug(tick="",  prefix="Cashflow debug: ")

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTradeFeatures":
        if cls is BaseTradeFeatures:
            return cls._deserialize(d)
        return super().from_dict(d)

    @classmethod
    def from_serialized_dict(cls, d: dict) -> "BaseTradeFeatures":
        return cls._deserialize(d)

    @staticmethod
    def _deserialize(d: dict) -> "BaseTradeFeatures":
        if isinstance(d, BaseTradeFeatures):
            return d
        inferred = infer_trade_features_type(d)
        if inferred is BaseTradeFeatures:
            return Serializable.from_dict.__func__(BaseTradeFeatures, d)
        return inferred.from_dict(d)


@dataclass
class SpreadFeatures(BaseTradeFeatures):
    other_contracts: Tuple[str]
    raw_carry: float
    smoothed_carry: float
    raw_relative_carry: float
    smoothed_relative_carry: float
    notional_exposure: float
    volatility: float


def _normalize_feature_tag_value(tag: Any) -> Any:
    if isinstance(tag, (bytes, bytearray)):
        try:
            tag = pickle.loads(tag)
        except Exception:
            tag = None
    if isinstance(tag, Enum):
        tag = tag.value
    return tag


def _normalize_feature_tag_text(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _iter_feature_subclasses(base_cls: Type["BaseTradeFeatures"]):
    for sub in base_cls.__subclasses__():
        yield sub
        yield from _iter_feature_subclasses(sub)


def infer_trade_features_type(d: dict) -> Type["BaseTradeFeatures"]:
    if not isinstance(d, dict):
        return BaseTradeFeatures

    tag = d.get("features_type") or d.get("feature_type") or d.get("type")
    tag = _normalize_feature_tag_value(tag)

    if isinstance(tag, type) and issubclass(tag, BaseTradeFeatures):
        return tag

    if isinstance(tag, str):
        tag_norm = _normalize_feature_tag_text(tag)
        for sub in _iter_feature_subclasses(BaseTradeFeatures):
            name_norm = _normalize_feature_tag_text(sub.__name__)
            if tag_norm == name_norm:
                return sub
            if name_norm.endswith("features") and tag_norm == name_norm[: -len("features")]:
                return sub
            alias = getattr(sub, "FEATURES_TAG", None)
            if isinstance(alias, str) and tag_norm == _normalize_feature_tag_text(alias):
                return sub
            aliases = getattr(sub, "FEATURES_TAGS", None)
            if isinstance(aliases, (list, tuple, set)):
                if any(
                    tag_norm == _normalize_feature_tag_text(str(a)) for a in aliases
                ):
                    return sub

    keys = set(d.keys())
    base_fields = {f.name for f in fields(BaseTradeFeatures)}
    best = None
    best_required = -1
    best_overlap = -1

    for sub in _iter_feature_subclasses(BaseTradeFeatures):
        sub_fields = [f for f in fields(sub) if f.name not in base_fields]
        if not sub_fields:
            continue
        required = {
            f.name
            for f in sub_fields
            if f.default is MISSING and f.default_factory is MISSING
        }
        all_names = {f.name for f in sub_fields}
        overlap = len(all_names & keys)
        if required and not required <= keys:
            continue
        if required:
            if (len(required), overlap) > (best_required, best_overlap):
                best = sub
                best_required, best_overlap = len(required), overlap
        else:
            if overlap > best_overlap:
                best = sub
                best_required, best_overlap = 0, overlap

    if best is not None and best_overlap > 0:
        return best
    return BaseTradeFeatures

@dataclass
class EntryData(Serializable):
    entry_date: DateObj
    position_type: PositionType
    price_series: PriceSeries
    exit_date: DateObj
    tick: str
    position_size: float = 1.0
    features: BaseTradeFeatures | None = None
    features_key: str | int | None = None

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
            features=(
                None if d.get("features") is None else BaseTradeFeatures.from_dict(d["features"])
            ),  # or decode if needed
            features_key=d.get("features_key"),
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
    
    def __post_init__(self):
        logging.info(
            f"[BACKTEST] Backtest completed: "
            f"Sharpe={self.sharpe:.3f}, "
            f"CAGR={self.cagr*100:.2f}%, "
            f"Total Return={self.total_return*100:.2f}%, "
            f"Max DD={self.max_drawdown*100:.2f}%, "
            f"Volatility={self.volatility*100:.2f}%, "
            f"Trades={len(self.snapshots)}"
        )
    

def _decode_underlying(u, tick):
    from .contract import Spot, Future, IntraDayPerf, BaseUnderlying

    if u is None:
        return None
    # if already decoded, return
    if isinstance(u, BaseUnderlying):
        return u
    utag = str(u.get("underlying_type", "")).lower()
    date = u.get("date")
    date = DateObj.from_iso(date) if isinstance(date, str) else date
    if "Spot".lower() in utag:
        return Spot(
            bid=u["bid"],
            ask=u["ask"],
            volume=u.get("volume", 0.0),
            date=date,
            tick=u.get("tick", tick),
        )
    if "Future".lower() in utag:
        return Future(
            bid=u["bid"],
            ask=u["ask"],
            volume=u.get("volume", 0.0),
            date=date,
            tick=u.get("tick", tick),
        )
    if "intradayperf" in utag or "intraday_perf" in utag:
        return IntraDayPerf(
            bid=u["bid"],
            ask=u["ask"],
            volume=u.get("volume", 0.0),
            date=date,
            tick=u.get("tick", tick),
        )
    return None
