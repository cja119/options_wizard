
from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC
from enum import Enum
from typing import Optional

from .base import Serializable
from .date import DateObj

@dataclass
class BaseUnderlying(Serializable, ABC):
    bid: float
    ask: float
    volume: float
    date: 'DateObj'
    tick: str

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class Spot(BaseUnderlying):
    pass

@dataclass
class Future(BaseUnderlying):
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
    rho:   Optional[float] = field(default=None)


    def __hash__(self) -> int:
        return hash((self.option_type, self.strike, self.expiry, self.date))
    

    