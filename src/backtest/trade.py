"""
Trade class definitions
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from data.trade import (
    TransactionCostModel,
    AccountingConvention,
    Equity,
    Cashflow
)

if TYPE_CHECKING:
    from ..data.date import DateObj
    from ..data.trade import (
        PositionType,
        EntryData,
        BaseUnderlying,
        BaseTradeFeatures
    )
    

class Trade:
    def __init__(
            self,
            entry_data: EntryData,
            transaction_cost_model: TransactionCostModel = TransactionCostModel.NONE,
            accounting_type: AccountingConvention = AccountingConvention.CASH,
            ) -> None:
        self.entry_data = entry_data
        self.accounting_type = accounting_type
        self.transaction_cost_model = transaction_cost_model
        self._opened = False
        self._closed = False

    # ---- External Interface ---- #
    def __call__(self, date: DateObj) -> Tuple[Equity | None, Cashflow | None]:

        if not self._live_condition(date):
            return None, None
        
        cashflow = None

        if self._open_condition(date):
            cashflow: Cashflow = self._initialize_trade()
        elif self._close_condition(date):
            cashflow: Cashflow = self._close_trade()

        if self.accounting_type == AccountingConvention.MTM:
            equity: Equity = self._mtm_equity(date)
        
        elif self.accounting_type == AccountingConvention.CASH:
            equity: Equity = self._cash_equity(date)

        return equity, cashflow
    
    def features(self) -> 'BaseTradeFeatures' | None:
        return self.entry_data.features
    
    def is_open_on(self, date: DateObj) -> bool:
        return self._open_condition(date) or self._live_condition(date) or self._close_condition(date)
    
    def close(self, date: DateObj) -> Tuple[Equity, Cashflow]:
        self.entry_data.exit_date = date
        equity, cashflow = self(date)
        return equity, cashflow
    
    def is_entering_on(self, date: DateObj) -> bool:
        return self._open_condition(date)

    # ---- Internal Methods ---- #
    def _calculate_tcs(self, price: BaseUnderlying) -> float:
        if self.transaction_cost_model == TransactionCostModel.NONE:
            return 0.0
        if self.transaction_cost_model == TransactionCostModel.SPREAD:
            spread = price.ask - price.bid
            return spread / 2
        else:
            return 0.0

    def _mtm_equity(self, date: DateObj) -> Equity:
        price: BaseUnderlying = self.entry_data.price_series.prices[date]
        equity = Equity(
            date=date,
            value=self._price_exposure(price) - self._entry_exposure,
            accounting_convention = AccountingConvention.MTM,
            parent_trade=self
        )
        return equity
    
    @property
    def _pos_sign(self) -> int:
        return 1 if self.entry_data.position_type == PositionType.LONG else -1

    def _cash_equity(self, date: DateObj) -> Equity:
        price: BaseUnderlying = self.entry_data.price_series.prices[date]
        ps: float = self.entry_data.position_size
        value: float = (price.bid + price.ask) / 2  * self._pos_sign * ps

        equity = Equity(
            date=date,
            value=value,
            accounting_convention = AccountingConvention.CASH,  
            parent_trade=self
        )

        return equity

    def _open_condition(self, date: DateObj) -> bool:
        return date == self.entry_data.entry_date and not self._opened and not self._closed
    
    def _close_condition(self, date: DateObj) -> bool:
        return date == self.entry_data.exit_date and not self._closed and self._opened
    
    def _live_condition(self, date: DateObj) -> bool:
        return self.entry_data.entry_date <= date <= self.entry_data.exit_date and not self._closed 

    def _close_trade(self) -> Cashflow:
        price: BaseUnderlying = self.entry_data.price_series.prices[self.entry_data.exit_date]
        tcs: float = self._calculate_tcs(price)

        if self.accounting_type == AccountingConvention.CASH:
            ps: float = self.entry_data.position_size 
            cashflow_amount = price.bid * ps - tcs if self._pos_sign > 0 else - price.ask * ps - tcs 
        elif self.accounting_type == AccountingConvention.MTM:
            cashflow_amount = self._price_exposure(price) - self._entry_exposure

        cashflow = Cashflow(
            date=self.entry_data.exit_date,
            amount=cashflow_amount,
            accounting_convention=self.accounting_type,
            parent_trade=self
        )

        self._closed = True
        self._opened = False

        return cashflow
    
    def __rmul__(self, other: float) -> 'Trade':
        if self._opened:
            raise RuntimeError("Cannot resize an already-open trade")
        self.entry_data.position_size *= other
        return self
    
    def __imul__(self, other: float) -> 'Trade':
        if self._opened:
            raise RuntimeError("Cannot resize an already-open trade")
        self.entry_data.position_size *= other
        return self
    
    def _cash_position(self, price: BaseUnderlying) -> float:
        tcs: float = self._calculate_tcs(price)
        return - price.ask - tcs if (self._pos_sign) > 0 else price.bid - tcs

    def _price_exposure(self, price: BaseUnderlying) -> float:
        ps: float = self.entry_data.position_size
        return (price.bid + price.ask) / 2 * self._pos_sign * ps
    
    def _initialize_trade(self) -> Cashflow:
       
        price: BaseUnderlying = self.entry_data.price_series.prices[self.entry_data.entry_date]

        if self.accounting_type == AccountingConvention.CASH:
            cashflow_amount =  self._cash_position(price)
            
        elif self.accounting_type == AccountingConvention.MTM:
            self._entry_exposure = self._price_exposure(price)
            cashflow_amount = self._calculate_tcs(price)
    
        cashflow = Cashflow(
            date=self.entry_data.entry_date,
            amount=cashflow_amount,
            accounting_convention=self.accounting_type,
            parent_trade=self
        )

        self._opened = True
        self._closed = False

        return cashflow


        
