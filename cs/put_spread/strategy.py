import options_wizard as ow
from typing import List, Dict, override

class FixedHold1MnNotional(ow.PositionBase):

    @override
    def size_function(self, entering_trades: List[ow.Trade]) -> Dict[ow.Trade, float]:
        sizes = {}
        notional = 2 * 1_000_000 / len(entering_trades) if entering_trades else 0
        for trade in entering_trades:
            underlying_price = trade.entry_data.price_series.get(trade.entry_data.entry_date)
            sizes[trade] = notional / underlying_price.bid
        return sizes
    
    @override
    def exit_trigger(self, live_trades: List[ow.Trade], current_date: ow.DateObj) -> Tuple[List[ow.Trade], List[ow.Cashflow]]:
        exit_cashflows = []
        for trade in live_trades.copy():
            if trade.days_open(current_date) >= 30:
                _, cashflow = trade.close(current_date)
                exit_cashflows.append(cashflow)
        return live_trades, exit_cashflows