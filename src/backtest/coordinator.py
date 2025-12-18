"""
Backtest Coordinator Module
"""

from __future__ import annotations

from typing import Deque, TYPE_CHECKING
from collections import deque
from .position.base import PositionBase

from tqdm import tqdm
from numpy import log as np_log
from numpy import fromiter as np_fromiter
from numpy import sqrt as np_sqrt

from data.trade import BackTestResult

if TYPE_CHECKING:
    from ..data.date import DateObj


class BackTestCoordinator:

    def __init__(
        self, position: PositionBase, dates: Deque[DateObj], debug=False
    ) -> None:
        self._position = position
        self._dates = dates
        self._snapshots = deque()
        self._returns = deque()
        self._max_drawdown = 0.0
        self._debug = debug
        pass

    # --- External Interface --- #
    def run(self) -> BackTestResult:
        self._run_backtest()
        return self._process_results()
    
    def compress(self, result: BackTestResult) -> None:

        pass

    # --- Internal Methods --- #

    def _run_backtest(self) -> None:

        max_equity = self._position.starting_cash
        current_equity = self._position.starting_cash
        max_drawdown = 0.0

        for date in tqdm(self._dates):
            snapshot = self._position(date)
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

        if self._debug:
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
