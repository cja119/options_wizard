from functools import total_ordering
from dataclasses import dataclass
from data.base import Serializable
import polars as pl

SECONDS_IN_DAY = 86400


@total_ordering
@dataclass(frozen=True)
class DateObj(Serializable):
    year: int
    month: int
    day: int

    # --- Conversion Methods --- #
    def to_datetime(self):
        from datetime import datetime
        return datetime(self.year, self.month, self.day)
    
    def to_pl(self) -> pl.Date:
        return pl.date(self.year, self.month, self.day)

    def to_iso(self) -> str:
        return self.__str__()
    
    def to_timestamp(self):
        return self.to_datetime().timestamp()

    # treat as a string
    def to_dict(self):
        return self.to_iso()

    # --- Static Methods --- #
    @staticmethod
    def from_datetime(dt):
        return DateObj(dt.year, dt.month, dt.day)

    @staticmethod
    def from_timestamp(ts):
        from datetime import datetime

        dt = datetime.fromtimestamp(ts)
        return DateObj(dt.year, dt.month, dt.day)
    
    @staticmethod
    def from_pl(date: pl.Date) -> "DateObj":
        return DateObj(year=date.year, month=date.month, day=date.day)
    
    @staticmethod
    def from_iso(date_str: str) -> "DateObj":
        year, month, day = map(int, date_str.split("-"))
        return DateObj(year, month, day)
    
    @classmethod
    def from_dict(cls, value):
        return cls.from_iso(value)

    # --- Operator Overloads --- #
    def __eq__(self, other):
        if not isinstance(other, DateObj):
            return NotImplemented
        return (self.year, self.month, self.day) == (other.year, other.month, other.day)

    def __lt__(self, other):
        if not isinstance(other, DateObj):
            return NotImplemented
        return (self.year, self.month, self.day) < (other.year, other.month, other.day)

    def __sub__(self, other):
        return (self.to_timestamp() - other.to_timestamp()) // SECONDS_IN_DAY

    def __add__(self, other: int):
        from datetime import timedelta

        dt = self.to_datetime() + timedelta(days=other)
        return DateObj.from_datetime(dt)

    def __str__(self):
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    def __repr__(self):
        return f"DateObj(year={self.year}, month={self.month}, day={self.day})"


