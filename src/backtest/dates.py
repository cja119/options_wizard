"""
Date range utility, based on the em module
"""

from enum import Enum
import pandas as pd
from data.date import DateObj
import exchange_calendars as ec


class Exchange(str, Enum):
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    LSE = "lse"
    CME = "cme"


def market_dates(lower: DateObj, upper: DateObj, exchange: Exchange) -> list[DateObj]:
    """
    Return all market-open dates (inclusive) between lower and upper for the given exchange.
    """
    # Map your enum → exchange_calendars name
    EXCHANGE_MAP = {
        Exchange.NYSE: "XNYS",
        Exchange.NASDAQ: "XNAS",
        Exchange.LSE: "XLON",
        Exchange.CME: "CMES",
    }

    cal_name = EXCHANGE_MAP[exchange]
    cal = ec.get_calendar(cal_name)
    start = pd.Timestamp(lower.to_iso())
    end = pd.Timestamp(upper.to_iso())

    sessions = cal.sessions_in_range(start, end)
    return [DateObj(year=ts.year, month=ts.month, day=ts.day) for ts in sessions]
