# Options-Wizard 🧙

This package provides the functionality to backtest and evaluate trade strategys in Python. This was originally built for options chains, so prioritises efficient computation through the evaluation of strategies serially (using polars).

## How it Works 🚀

1. Data processing is defined via a generalised function pipeline, which can iterate through different securities. This builds data objects that contain all necessary information to evaluate a trade for that security (typically a time indexed set of security objects).

2. During the backtest, these cached security objects are loaded and processed by a generalisable position sizing and risk management class.

3. Cash, position and margin tracking is performed accross all dates, cross exchange for as many securities and strategies that are loaded. This allows portfolio level backtsests to be performed alongside strategy level.

The benefit of this approach is it separates trade identification and building from backtesting, this makes it much quicker to develop and implement new strategies. Crucially, backtesting is done cross sectionally which gives much better NAV, cash  and margin tracking.

## To do 🔎
- Intraday evaluation
- Case studies (using yf data?)
- Docs
- Build some out the box regression models for use too.
- Make universe a customizable base such that users can define their own instances for specific applications.
- Add greek calculations to options definitions so can be used for sizing (thinking notional exposure targeting)
- Come up with a better package name(?)

## Disclaimer ⚠️
This software is provided as is and shouldn't be used as the basis of any investment decisions and does not offer any investment or financial advice. I am not liable for any losses incurred through the use of this product.
