
from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .date import DateObj

@dataclass
class BaseUnderlying(ABC):
    bid: float
    ask: float
    volume: float
    date: 'DateObj'

class OptionType(Enum):
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
    underlying: Optional['Spot' | 'Future'] = field(default=None)   
    rfr: Optional[float] = field(default=None)       
    delta: Optional[float] = field(default=None)
    gamma: Optional[float] = field(default=None)
    vega: Optional[float] = field(default=None)
    theta: Optional[float] = field(default=None)
    rho:   Optional[float] = field(default=None)


    def __hash__(self) -> int:
        return hash((self.option_type, self.strike, self.expiry, self.date))


    