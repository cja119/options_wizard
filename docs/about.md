## How it Works 🚀

1. Data processing is defined via a generalised function pipeline, which can iterate through different securities. This builds data objects that contain all necessary information to evaluate a trade for that security (typically a time indexed set of security objects).

2. During the backtest, these cached security objects are loaded and processed by a generalisable position sizing and risk management class.

3. Cash, position and margin tracking is performed accross all dates, cross exchange for as many securities and strategies that are loaded. This allows portfolio level backtsests to be performed alongside strategy level.

The benefit of this approach is it separates trade identification and building from backtesting, this makes it much quicker to develop and implement new strategies. Crucially, backtesting is done cross sectionally which gives much better NAV, cash  and margin tracking.
