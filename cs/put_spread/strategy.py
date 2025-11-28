def ratio_spread(data, **kwargs) -> pd.DataFrame:
    lower_ttm = kwargs.get('lower_ttm', 90)
    upper_ttm = kwargs.get('upper_ttm', 150)
    delta_atm = kwargs.get('delta_atm', 0.45)
    delta_otm = kwargs.get('delta_otm', 0.15)
    otm_ratio = kwargs.get('otm_ratio', 2)
    hold_period = kwargs.get('hold_period', 30)
    call_put = kwargs.get('call_put', 'p')

    data['entered'] = False
    data['position'] = 0

    def pick_daily_contracts(day_chain, **kwargs) -> pd.DataFrame:
        fitler = (
            (day_chain['delta'].abs() <= delta_atm) &
            day_chain['ttm'].between(lower_ttm, upper_ttm) &
            (day_chain['call_put'] == call_put) &
            (day_chain['days_until_last_trade'] > hold_period)
        )
        atm_candidates = day_chain.loc[fitler]
        
        if atm_candidates.empty:
            return day_chain
        
        nearest_atm = atm_candidates.iloc[
            (atm_candidates['delta'].abs() - delta_atm).abs().argsort()[:1]
        ]
        
        otm_candidates = day_chain.loc[fitler & (day_chain['delta'].abs() > delta_otm)]
        
        if otm_candidates.empty:
            return day_chain
    
        nearest_otm = otm_candidates.iloc[(otm_candidates['delta'].abs() - delta_otm).abs().argsort()[:1]]

        if not nearest_atm.empty and not nearest_otm.empty:
            day_chain.loc[nearest_atm.index, 'entered'] = True
            day_chain.loc[nearest_otm.index, 'entered'] = True
            day_chain.loc[nearest_atm.index, 'position'] = -1
            day_chain.loc[nearest_otm.index, 'position'] = otm_ratio

        return day_chain

    grouped = data.groupby('trade_date')
    processed = grouped.apply(pick_daily_contracts)
    result = (
        processed[processed['entered']]
        .drop(columns=['entered'])
        .reset_index(level='trade_date', drop=True)
    )
    # Add these two lines before assigning back into data:
    processed_aligned = processed.droplevel(0).reindex(data.index)

    data['entered'] = processed_aligned['entered']
    data['position'] = processed_aligned['position']

    return result
