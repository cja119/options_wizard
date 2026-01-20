"""
Docstring for backtest.position.futures.carry
"""

from data.date import DateObj
from ..base import PositionBase, Cashflow
from backtest.trade import Trade
from typing import Tuple, override, List

VOLATILITY_TARGET = 0.2
VOLATILITY_FLOOR = 0.02
MAX_NOTIONAL_FRAC = 1.0

class FuturesCarry(PositionBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._live_spreads = set()
        self._live_vol = 0.0

    def _spread_vol(self, trade: Trade) -> float:
        vol_floor = self._kwargs.get("vol_floor", VOLATILITY_FLOOR)
        vol_floor = 0.0 if vol_floor is None else float(vol_floor)
        vol = getattr(trade.features, "volatility", 0.0) if trade.features else 0.0
        return max(vol, vol_floor) if vol_floor > 0 else vol

    def _spread_notional(self, trade: Trade) -> float:
        notional = (
            getattr(trade.features, "notional_exposure", 0.0) if trade.features else 0.0
        )
        return notional or 0.0

    def _portfolio_vol(self, trade: Trade, size: float, equity: float) -> float:
        if size <= 0.0 or equity <= 0.0:
            return 0.0
        spread_vol = self._spread_vol(trade)
        notional = self._spread_notional(trade)
        if spread_vol <= 0.0 or notional <= 0.0:
            return 0.0
        return (spread_vol * notional * size) / equity

    @override
    def size_function(self, entering_trades):
        
        # Calculating current equity
        te = self._snapshot.total_equity + self._snapshot.total_cash
        
        entering_spreads = {}
        sizes = {}
        total_notional = 0.0
        
        # Iterating through entering trades and group by spread
        for trade in entering_trades:
            spread_key = trade.features.other_contracts
            # Skip if we are already live on this spread
            if spread_key in self._live_spreads:
                continue
            
            # Grouping trades by their spread features
            if spread_key not in entering_spreads:
                total_notional += trade.features.notional_exposure
                entering_spreads[spread_key] = []
            
            entering_spreads[spread_key].append(trade)

        # If not entering spreads, return zero sizes
        if not entering_spreads:
            return {trade: 0.0 for trade in entering_trades}
        
        # We want to account for any scheduled volatility changes today as trades
        # last a month, we won't hit our vol budget otherwise.
        scheduled_exits = self._pre_entry_exits
        exiting_vol = self._exiting_vol(scheduled_exits)
        projected_vol = max(self._live_vol - exiting_vol, 0.0)

        # Per spread volatility budget, we assume uncorrelated returns. Need to 
        # think about how we make this more realistic as many commodites are correlated.
        vol_target = self._kwargs.get("vol_target", VOLATILITY_TARGET)
        vol_target = float(VOLATILITY_TARGET if vol_target is None else vol_target)
        vol_headroom = max(vol_target - projected_vol, 0.0)
        vol_budget = vol_headroom / len(entering_spreads)
        max_notional_frac = self._kwargs.get("max_notional_frac", MAX_NOTIONAL_FRAC)

        # Now size each spread based on its notional and vol, each entering trade gets
        # the same share of volatility budget.
        for trades in entering_spreads.values():
            
            # Sizing based on volatility target share
            spread_vol = self._spread_vol(trades[0])
            notional = self._spread_notional(trades[0])
            if spread_vol <= 0.0 or notional <= 0.0 or te <= 0.0:
                size_multip = 0.0
            else:
                size_multip = (vol_budget * te) / (spread_vol * notional)
            if max_notional_frac is not None and notional > 0.0 and te > 0.0:
                max_size = (max_notional_frac * te) / notional
                if size_multip > max_size:
                    size_multip = max_size

            for trade in trades:
                sizes[trade] = size_multip
                
        for trade in entering_trades:
            if trade not in sizes:
                sizes[trade] = 0.0

        entering_vol = 0.0
        for spread_key, trades in entering_spreads.items():
            if not trades:
                continue
            trade = trades[0]
            entering_vol += self._portfolio_vol(trade, sizes.get(trade, 0.0), te)
            if sizes.get(trade, 0.0) != 0.0:
                self._live_spreads.add(spread_key)
        self._live_vol += entering_vol
        return sizes

    def _exiting_vol(self, exiting_trades: List[Trade]) -> float:
        total_vol = 0.0
        exiting_spreads = set()
        te = self._snapshot.total_equity + self._snapshot.total_cash
        for trade in exiting_trades:
            if trade.features.other_contracts in exiting_spreads:
                continue
            exiting_spreads.add(trade.features.other_contracts) 
            total_vol += self._portfolio_vol(trade, self._sizes.get(trade, 0.0), te)
        return total_vol
    
    
    @override
    def exit_trigger(
        self,
        live_trades: List[Trade],
        current_date: DateObj,
        scheduled_exits: List[Trade],
    ) -> Tuple[List[Trade], List[Cashflow]]:
        
        trades_to_close = set()
        exit_cashflows = []

        for trade in scheduled_exits:
            if trade.features.other_contracts in self._live_spreads:
                self._live_spreads.remove(trade.features.other_contracts)

            if trade not in trades_to_close:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
                trades_to_close.add(trade)

        if trades_to_close:
            self._live_vol -= self._exiting_vol(list(trades_to_close))
        return trades_to_close, exit_cashflows
