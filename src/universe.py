"""
Universe definition for options wizard
"""

from __future__ import annotations

import pandas as pd


class Universe:

    def __init__(self, ticks: list[str]):
        self.ticks: list[str] = ticks
        self.lower_date: pd.Timestamp | None = None
        self.upper_date: pd.Timestamp | None = None
        return None

    def set_dates(self, lower: pd.Timestamp, upper: pd.Timestamp):
        self.lower_date = lower
        self.upper_date = upper
        return None
