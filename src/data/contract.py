from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC
from enum import Enum
import pickle
import sys
from typing import Callable, Optional, Type, List

from .base import Serializable
from .date import DateObj


@dataclass
class OptionsTradeSpec:
    call_put: OptionType
    lm_fn: Callable = field(default=lambda x: True)
    ttm: Callable = field(default=lambda x: True)
    abs_delta: Callable = field(default=lambda x: True)
    entry_cond: Callable | List[Callable] = field(default=lambda x: True)
    entry_col: Optional[str] | List[Optional[str]] = field(default=None)
    exit_cond: Callable | List[Callable] = field(default=lambda x: False)
    exit_col: Optional[str] | List[Optional[str]] = field(default=None)
    volume_min: int = 0
    open_interest_min: int = 0
    entry_min: str = "perc_spread"
    max_hold_period: int = 30
    position: float = 1.0


@dataclass
class CarryTradeSpec:
    tenor_targets: List[int]    
    exposure_targets: List[float]
    roll_target: int
    metric: str  # "FRONT_CARRY" or "SPREAD_CARRY"
    spread_override_bps: Optional[float] = field(default=None)

    def __post_init__(self):
        if len(self.tenor_targets) != len(self.exposure_targets):
            raise ValueError("Tenor targets and exposure targets must be the same length")
        if sum(self.exposure_targets) != 0.0:
            raise ValueError("Exposure targets must sum to zero")
        if self.metric not in ["FRONT_RELATIVE", "INTERNAL"]:
            raise ValueError("Metric must be either FRONT_RELATIVE or INTERNAL")


@dataclass
class BaseUnderlying(Serializable, ABC):
    # --- Required Fields --- #
    bid: float
    ask: float
    volume: float
    date: "DateObj"
    tick: str

    @classmethod
    def from_serialized_dict(cls, d: dict):
        """Dispatch to the correct subclass using the embedded underlying_type tag."""
        return infer_underlying_type(d).from_dict(d)


class OptionType(str, Enum):
    CALL = "c"
    PUT = "p"


class UnderlyingType(str, Enum):
    OPTION = "Option"
    FUTURE = "Future"
    SPOT = "Spot"


@dataclass
class Spot(BaseUnderlying):
    underlying_type: UnderlyingType = UnderlyingType.SPOT
    pass


@dataclass
class Future(BaseUnderlying):
    # --- Required Fields --- #
    expiry: DateObj
    contract_id: str

    # -- Optional Fields --- #
    underlying_type: UnderlyingType = UnderlyingType.FUTURE
    settlement_price: Optional[float] = field(default=None)
    contract_multiplier: Optional[float] = field(default=None)
    open_interest: Optional[float] = field(default=None)
    pass


@dataclass
class Option(BaseUnderlying):
    # --- Required Fields --- #
    option_type: OptionType
    strike: float
    expiry: DateObj

    # --- Optional Fields --- #
    iv: Optional[float] = field(default=None)
    underlying: Optional[BaseUnderlying] = field(default=None)
    rfr: Optional[float] = field(default=None)
    delta: Optional[float] = field(default=None)
    gamma: Optional[float] = field(default=None)
    vega: Optional[float] = field(default=None)
    theta: Optional[float] = field(default=None)
    rho: Optional[float] = field(default=None)
    other: Optional[dict] = field(default=None)

    # --- Internal Fields --- #
    underlying_type: UnderlyingType = UnderlyingType.OPTION

    def __hash__(self) -> int:
        return hash((self.option_type, self.strike, self.expiry, self.date))


def infer_underlying_type(d: dict) -> Type["BaseUnderlying"]:
    """
    Heuristically choose the concrete BaseUnderlying subclass for a serialized dict.
    Prefer the explicit 'underlying_type' tag when present (Enum, string, or pickled bytes),
    otherwise fall back to structural hints.
    """
    if not isinstance(d, dict):
        return BaseUnderlying

    tag = d.get("underlying_type")

    # unwrap common serialized forms of the tag
    if isinstance(tag, (bytes, bytearray)):
        try:
            tag = pickle.loads(tag)
        except Exception:
            tag = None

    if isinstance(tag, str):
        try:
            tag = UnderlyingType(tag)
        except ValueError:
            tag = None

    if isinstance(tag, UnderlyingType):
        # dynamic: map the enum's value to the class object in this module
        cls_name = tag.value
        cls = getattr(sys.modules[__name__], cls_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseUnderlying):
            return cls

    # structural fallback
    if {"option_type", "strike", "expiry"} <= d.keys():
        return Option
    return BaseUnderlying
