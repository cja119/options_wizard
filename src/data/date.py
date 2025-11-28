
from dataclasses import dataclass
from functools import total_ordering

SECONDS_IN_DAY = 86400

@total_ordering
@dataclass(frozen=True)
class DateObj:
    year: int
    month: int
    day: int

    # --- Conversion Methods --- #
    def to_datetime(self):
        from datetime import datetime
        return datetime(self.year, self.month, self.day)
    
    def to_timestamp(self):
        return self.to_datetime().timestamp()
    
    # --- Static Methods --- #
    @staticmethod
    def from_datetime(dt):
        return DateObj(dt.year, dt.month, dt.day)
    
    @staticmethod
    def from_timestamp(ts):
        from datetime import datetime
        dt = datetime.fromtimestamp(ts)
        return DateObj(dt.year, dt.month, dt.day)
    
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


