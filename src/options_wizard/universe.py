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

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .backtest.trade import DateObj

logger = structlog.get_logger(__name__)


@dataclass
class Stock:
    ticker: str
    tradable: List[Tuple[DateObj, DateObj]]


class Universe:

    def __init__(self, ticks: list[str] | None = None):
        self.ticks: list[str] | None = ticks
        self.lower_date: DateObj | None = None
        self.upper_date: DateObj | None = None
        return None

    def set_dates(self, lower: DateObj, upper: DateObj):
        self.lower_date = lower
        self.upper_date = upper
        return None

    def top_constituents(self, n_const: int) -> None:

        load_dotenv()

        na_vals = ["NaN", "nan", "#N/A", "#N/A N/A", "#N/A Invalid Security"]

        index_df = pd.read_csv(os.getenv("INDEX_CONSTITUENTS_PATH"), na_values=na_vals)
        mktcap_df = pd.read_csv(os.getenv("MKT_CAP_PATH"), na_values=na_vals)

        index_df = index_df.drop(columns=index_df.columns[0])
        mktcap_df = mktcap_df.drop(columns=mktcap_df.columns[0])

        index_df = index_df.rename(columns={index_df.columns[0]: "date"})
        mktcap_df = mktcap_df.rename(columns={mktcap_df.columns[0]: "date"})

        index_df["date"] = pd.to_datetime(index_df["date"], dayfirst=True)
        mktcap_df["date"] = pd.to_datetime(mktcap_df["date"], dayfirst=True)

        index_df = index_df.set_index("date")
        mktcap_df = mktcap_df.set_index("date")

        index_long = index_df.stack().reset_index()
        index_long = index_long.rename(columns={"level_1": "col_idx", 0: "ticker"})

        mktcap_long = mktcap_df.stack().reset_index()
        mktcap_long = mktcap_long.rename(columns={"level_1": "col_idx", 0: "mktcap"})

        merged = pd.merge(index_long, mktcap_long, on=["date", "col_idx"], how="outer")
        panel = merged.pivot(index="date", columns="ticker", values="mktcap")

        if self.lower_date is not None and self.upper_date is not None:
            panel = panel.loc[self.lower_date : self.upper_date]

        panel_clean = panel.loc[:, ~panel.columns.str.match(r"^\d")]
        top_n_per_date = panel_clean.apply(
            lambda row: row.nlargest(n_const).index.tolist(), axis=1
        )
        self.top_per_date = top_n_per_date
        self.ticks = list({ticker for sublist in top_n_per_date for ticker in sublist})

        return None

    def check_ticks(self):
        load_dotenv()
        tick_path = os.getenv("TICK_PATH", "").split(os.pathsep)[0]

        p = Path(tick_path)
        if p.is_dir():  # make sure it exists
            available_ticks = [
                f.name.replace(".parquet", "") for f in p.iterdir() if f.is_file()
            ]

        for stock in self.ticks.copy():
            if stock not in available_ticks:
                logger.warning(
                    "Removing ticker from universe - data not available",
                    tick=stock,
                )
                self.ticks.remove(stock)
