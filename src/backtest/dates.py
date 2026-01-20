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
    CBOT = "cbot"
    COMEX = "comex"
    DCE = "dce"
    LME = "lme"
    EDX = "edx"


_EXCHANGE_ALIASES = {
    # ---- NYSE ----
    "nyse": Exchange.NYSE,
    "new york stock exchange": Exchange.NYSE,
    "xnys": Exchange.NYSE,
    # ---- NASDAQ ----
    "nasdaq": Exchange.NASDAQ,
    "nasdaq stock market": Exchange.NASDAQ,
    "xnas": Exchange.NASDAQ,
    # ---- LSE ----
    "lse": Exchange.LSE,
    "london stock exchange": Exchange.LSE,
    "xlon": Exchange.LSE,
    # ---- CME / Globex ----
    "cme": Exchange.CME,
    "cme group": Exchange.CME,
    "cme globex": Exchange.CME,
    "xcme": Exchange.CME,
    "xcle": Exchange.CME,  # CME Clearing (clearing MIC)
    # ---- NYMEX ----
    "nym": Exchange.NYMEX,
    "nymex": Exchange.NYMEX,
    "new york mercantile exchange": Exchange.NYMEX,
    # ---- ICE ----
    "ice": Exchange.ICE,
    "iceus": Exchange.ICE,
    "nyb": Exchange.ICE,  # NYBOT (ICE)
    # ---- Additions you asked for ----
    # CBOT (you wrote "cbt")
    "cbt": Exchange.CBOT,
    "cbot": Exchange.CBOT,
    "xcbt": Exchange.CBOT,
    # COMEX (you wrote "cmx"; duplicated "cmx" ignored by dict anyway)
    "cmx": Exchange.COMEX,
    "comex": Exchange.COMEX,
    # DCE / LME / EDX
    "dce": Exchange.DCE,
    "xdce": Exchange.DCE,
    "lme": Exchange.LME,
    "xlme": Exchange.LME,
    "edx": Exchange.EDX,
    "xedx": Exchange.EDX,
}


def market_dates(
    lower: DateObj | date, upper: DateObj | date, exchange: Exchange
) -> list[DateObj]:
    """
    Return all market-open dates (inclusive) between lower and upper for the given exchange.
    """
    # Map your enum → exchange_calendars name
    EXCHANGE_MAP = {
        Exchange.NYSE: "XNYS",
        Exchange.NASDAQ: "XNAS",
        Exchange.LSE: "XLON",
        # Futures/derivatives
        Exchange.CME: "CMES",
        Exchange.NYMEX: "NYMEX",
        Exchange.CBOT: "CBOT",
        Exchange.COMEX: "COMEX",
        Exchange.ICE: "ICEUS",
        # Approximations for missing calendars in exchange_calendars:
        # DCE (mainland China) ~ Shanghai Stock Exchange holidays
        Exchange.DCE: "XSHG",
        # LME / EDX (London markets) ~ LSE holidays
        Exchange.LME: "XLON",
        Exchange.EDX: "XLON",
    }

    cal_name = EXCHANGE_MAP[_EXCHANGE_ALIASES[exchange.lower()]]
    cal = ec.get_calendar(cal_name)
    start = (
        pd.Timestamp(lower.to_iso())
        if isinstance(lower, DateObj)
        else pd.Timestamp(lower)
    )
    end = (
        pd.Timestamp(upper.to_iso())
        if isinstance(upper, DateObj)
        else pd.Timestamp(upper)
    )

    sessions = cal.sessions_in_range(start, end)
    if isinstance(lower, DateObj) and isinstance(upper, DateObj):
        return [DateObj(year=ts.year, month=ts.month, day=ts.day) for ts in sessions]
    else:
        return list(sessions)


def market_dates_union(
    lower: DateObj | date,
    upper: DateObj | date,
    exchanges: list[Exchange | str],
) -> list[DateObj] | list[pd.Timestamp]:
    """
    Return the union of market-open dates across multiple exchanges.
    """
    dates: set = set()
    for exchange in exchanges:
        dates.update(market_dates(lower, upper, exchange=exchange))
    return sorted(dates)
