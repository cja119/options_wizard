"""
Backtest Coordinator Module
"""

from __future__ import annotations

from typing import Deque, Dict, TYPE_CHECKING
from collections import deque
from dataclasses import dataclass, field
from abc import ABC
import logging


from numpy import log as np_log
from numpy import fromiter as np_fromiter
from numpy import sqrt as np_sqrt

from options_wizard.data.trade import BackTestResult
from options_wizard.position.base import PositionBase
from . import diagnostics as bt_diag

if TYPE_CHECKING:
    from ..data.date import DateObj

@dataclass
class BackTestConfig(ABC):
    starting_cash: float
    start_date: DateObj
    end_date: DateObj | None = None
    kwargs: Dict = field(default_factory=dict)


class BackTestCoordinator:

    def __init__(
        self, position: PositionBase, dates: Deque[DateObj]
    ) -> None:
        self._position = position
        self._dates = dates
        self._snapshots = deque()
        self._returns = deque()
        self._max_drawdown = 0.0
        pass

    # --- External Interface --- #
    def run(self) -> BackTestResult:
        bt_diag.reset(clear_log=True)
        self._run_backtest()
        result = self._process_results()
        bt_diag.finalize()
        return result

    def compress(self, result: BackTestResult) -> None:

        pass

    # --- Internal Methods --- #

    def _run_backtest(self) -> None:

        max_equity = self._position.starting_cash
        current_equity = self._position.starting_cash
        max_drawdown = 0.0

        for date in self._dates:
            snapshot = self._position(date)

            logging.debug(
                f"Backtest snapshot on {date.to_iso()}: "
                f"equity={snapshot.total_equity:.2f}, "
                f"cash={snapshot.total_cash:.2f}",
                extra={"tick_name": "BACKTEST"}
            )

            position = snapshot.total_equity + snapshot.total_cash
            self._snapshots.append(snapshot)

            if position > max_equity:
                max_equity = position

            drawdown = (max_equity - position) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            self._returns.append(np_log(position / current_equity))
            current_equity = position
        self._max_drawdown = max_drawdown
        pass


    def _process_results(self) -> BackTestResult:

        returns_array = np_fromiter(self._returns, dtype=float)
        std = returns_array.std()
        volatility = std * np_sqrt(252)
        final_pos = self._snapshots[-1].total_equity + self._snapshots[-1].total_cash
        sharpe_ratio = (
            (returns_array.mean() / std) * np_sqrt(252) if volatility > 0 else 0.0
        )
        total_return = final_pos / self._position.starting_cash - 1
        cagr = (final_pos / self._position.starting_cash) ** (
            252 / len(self._snapshots)
        ) - 1
        dates = [self._snapshots[i].date for i in range(len(self._snapshots))]

        self._clean_snapshots()

        return BackTestResult(
            snapshots=list(self._snapshots),
            returns=returns_array.tolist(),
            sharpe=sharpe_ratio,
            max_drawdown=self._max_drawdown,
            volatility=volatility,
            total_return=total_return,
            cagr=cagr,
            dates=dates,
        )

    def _clean_snapshots(self) -> None:
        for snapshot in self._snapshots:
            update_equity = {}
            for trade, equity in snapshot.trade_equities.items():
                equity.underlying_trade = None
                update_equity[trade] = equity

    @staticmethod
    def dk(d: DateObj):
        return (d.year, d.month, d.day)
