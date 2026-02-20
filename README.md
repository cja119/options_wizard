# Options-Wizard 

This package provides the functionality to backtest and evaluate trade strategys in Python. This was originally built for options chains, so prioritises efficient computation through the evaluation of strategies serially (using polars); NAV tracking is kept accurate through in the cross section (ie time-stepping throughout the whole universe). This is designed to integrate into existing databases and is compatible with SQL. Have a look at the examples, or read through the docs to find out more. 

## To do
- Intraday evaluation
- Case studies (using yf data?)
- Docs
- Build some out the box regression models for use too.
- Make universe a customizable base such that users can define their own instances for specific applications.
- Add greek calculations to options definitions so can be used for sizing (thinking notional exposure targeting)

## Disclaimer
This software is provided as is and shouldn't be used as the basis of any investment decisions and is not providing financial advice. I am not liable for any losses incurred through the use of this product.
