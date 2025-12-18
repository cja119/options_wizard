"""
Date range utility, based on the em module
"""

from enum import Enum
import pandas as pd
from data.date import DateObj
from datetime import date
import exchange_calendars as ec


class Exchange(str, Enum):
    NYSE = "nyse"
    NYMEX = "nymex"
    NASDAQ = "nasdaq"
    LSE = "lse"
    CME = "cme"
    ICE = "ice"

_EXCHANGE_ALIASES = {
    # ---- NYSE ----
    "nyse": Exchange.NYSE,
    "new york stock exchange": Exchange.NYSE,
    "xnys": Exchange.NYSE,

    # ---- NASDAQ ----
    "nasdaq": Exchange.NASDAQ,
    "nasdaq stock market": Exchange.NASDAQ,
    "xnas": Exchange.NASDAQ,

    # ---- NYMEX / Energy ----
    "nym": Exchange.NYMEX,
    "nymex": Exchange.NYMEX,
    "new york mercantile exchange": Exchange.NYMEX,

    # ---- CME (parent / Globex) ----
    "cme": Exchange.CME,
    "cme group": Exchange.CME,
    "cme globex": Exchange.CME,
    "xcle": Exchange.CME,   # CME Clearing
    "xcme": Exchange.CME,   # CME MIC

    # ---- LSE ----
    "lse": Exchange.LSE,
    "london stock exchange": Exchange.LSE,
    "xlon": Exchange.LSE,

    # ---- ICE ----
    "ice": Exchange.ICE,
    "nyb": Exchange.ICE,  # New York Board of Trade
}


def market_dates(lower: DateObj | date, upper: DateObj | date, exchange: Exchange) -> list[DateObj]:
    """
    Return all market-open dates (inclusive) between lower and upper for the given exchange.
    """
    # Map your enum → exchange_calendars name
    EXCHANGE_MAP = {
        Exchange.NYSE: "XNYS",
        Exchange.NASDAQ: "XNAS",
        Exchange.LSE: "XLON",
        Exchange.CME: "CMES",
        Exchange.NYMEX: "CMES",
        Exchange.ICE: "ICEUS",
    }

    cal_name = EXCHANGE_MAP[_EXCHANGE_ALIASES[exchange.lower()]]
    cal = ec.get_calendar(cal_name)
    start = pd.Timestamp(lower.to_iso()) if isinstance(lower, DateObj) else pd.Timestamp(lower)
    end = pd.Timestamp(upper.to_iso()) if isinstance(upper, DateObj) else pd.Timestamp(upper)

    sessions = cal.sessions_in_range(start, end)
    if isinstance(lower, DateObj) and isinstance(upper, DateObj):
        return [DateObj(year=ts.year, month=ts.month, day=ts.day) for ts in sessions]
    else:
        return list(sessions)