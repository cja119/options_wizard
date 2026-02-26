"""
Universe definition for options wizard
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
import os
import structlog
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest.trade import DateObj

logger = structlog.get_logger(__name__)


@dataclass
class Stock:
    ticker: str
    tradable: List[Tuple[DateObj, DateObj]]

class UniverseBase(ABC):

    def __init__(self, ticks: list[str] | None = None):
        self.ticks: list[str] | None = ticks
        self.lower_date: DateObj | None = None
        self.upper_date: DateObj | None = None
        return None

    def set_dates(self, lower: DateObj, upper: DateObj):
        self.lower_date = lower
        self.upper_date = upper
        return None
    
    @abstractmethod
    def load_constituents(self, **kwargs) -> None:
        return None

    @abstractmethod
    def check_ticks(self, **kwargs):
        return None