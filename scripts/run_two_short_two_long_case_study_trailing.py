"""
Backtest a 4-leg put structure (2 shorts, 2 longs) with delta trailing stops and visualise performance.

Setup:
- 1 short put at ~35 delta
- 1 short put at ~50 delta
- 2 long puts at ~20 delta
- 3 long puts at ~15 delta
- Minimise entry by time-to-maturity inside a 120-180d window, hold/roll every 120d
- Transaction costs: 50% spread capture
- Trailing stops (per contract): gap 0.05 abs(delta); shorts stop if abs(delta) rises by gap from best (lowest) abs delta; longs stop if abs(delta) falls by gap from best (highest) abs delta; any stop closes all legs.

Outputs:
- Stacked plot: strategy payoff (equity), underlying curve, drawdown
- Annualised simple return (no compounding), Sharpe ratio, and max drawdown
- Entry-day IV / skew plot: short entry IV and long-short skew (near/far OTM)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from operator import eq, le
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import options_wizard as ow
import pandas as pd
import yfinance as yf
from backtest.position.fixed_hold_delta_trail import FixedHoldNotionalDeltaTrail

# -----------------------------------------------------------
#                    CONFIG
# -----------------------------------------------------------

START_DATE = ow.DateObj(2008, 1, 1)
END_DATE = ow.DateObj(2020, 12, 31)
STOCK = "NDQ"
DELTA_TOL = 0.05
PROTECTED_NOTIONAL = 1_000_000
SPREAD_CAPTURE_OVERRIDE = 0.5  # capture half the spread
EXPIRY_RANGE = (120, 180)
HOLD_PERIOD = 120
UNDERLYING_TICKER = "^IXIC"
PLOT_PATH = Path("tmp/two_short_two_long_case_study_trailing.png")
IV_SKEW_PLOT_PATH = Path("tmp/two_short_two_long_case_study_trailing_iv_skew.png")
TRADING_DAYS = 252
TRAILING_GAP = 0.05

PALETTE = {
    "equity": "#1f77b4",
    "underlying": "#111827",
    "drawdown": "#d9534f",
    "combined": "#f59e0b",
    "short_iv": "#1b9e77",
    "skew_near": "#d62728",
    "skew_far": "#9467bd",
}


# -----------------------------------------------------------
#                    SPEC HELPERS
# -----------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    delta: float
    position: float  # negative for short, positive for long


LEGS: List[Leg] = [
    Leg(delta=0.35, position=-1.0),  # short ~35 delta
    Leg(delta=0.50, position=-1.0),  # short ~50 delta
    Leg(delta=0.20, position=2.0),  # two longs ~20 delta
    Leg(delta=0.15, position=3.0),  # three longs ~15 delta
]


def _delta_filter(target: float):
    return lambda d, tgt=target: (abs(d) >= tgt - DELTA_TOL) and (
        abs(d) <= tgt + DELTA_TOL
    )


def _ttm_filter(t: float) -> bool:
    return (t >= EXPIRY_RANGE[0]) and (t <= EXPIRY_RANGE[1])


def build_specs() -> List[ow.OptionsTradeSpec]:
    specs = []
    for leg in LEGS:
        specs.append(
            ow.OptionsTradeSpec(
                call_put=ow.OptionType.PUT,
                ttm=_ttm_filter,
                lm_fn=lambda k: True,
                abs_delta=_delta_filter(leg.delta),
                entry_min="ttm",
                max_hold_period=HOLD_PERIOD,
                position=leg.position,
            )
        )
    return specs


# -----------------------------------------------------------
#                    CORE BACKTEST
# -----------------------------------------------------------


def _load_or_build_strat(specs: List[ow.OptionsTradeSpec], suffix: str) -> ow.StratType:
    try:
        return ow.StratType.load(STOCK, save_type=ow.SaveType.PICKLE, suffix=suffix)
    except Exception:
        pass

    universe = ow.Universe([STOCK])
    pipeline = ow.Pipeline(
        universe=universe,
        save_type=ow.SaveType.PICKLE,
        saves=[ow.SaveFrames.STRAT],
    )

    kwargs: Dict = {
        "max_date": END_DATE.to_pl(),
        "keep_col": ["call_put", "ttm", "n_missing"],
        "keep_oper": [eq, le, le],
        "keep_val": ["p", EXPIRY_RANGE[1], 0],
        "specs": specs,
        "hold_period": HOLD_PERIOD,
        "protected_notional": PROTECTED_NOTIONAL,
        "suffix": suffix,
    }

    ow.add_idx_spread_methods_opt(pipeline, kwargs)
    pipeline.run()
    return ow.StratType.load(STOCK, save_type=ow.SaveType.PICKLE, suffix=suffix)


def run_structure(
    suffix: str = "two_short_two_long_case_trailing", return_trades: bool = False
) -> ow.BackTestResult | tuple[ow.BackTestResult, List[ow.Trade]]:
    specs = build_specs()
    strat = _load_or_build_strat(specs, suffix)
    ptf = partial(
        ow.Trade,
        transaction_cost_model=ow.TransactionCostModel.SPREAD,
        accounting_type=ow.AccountingConvention.CASH,
    )
    trades = strat.reconstruct(ptf)

    for trade in trades or []:
        try:
            trade._spread_capture = SPREAD_CAPTURE_OVERRIDE
        except Exception:
            pass

    cfg = ow.BackTestConfig(
        starting_cash=PROTECTED_NOTIONAL,
        start_date=START_DATE,
        end_date=END_DATE,
        kwargs={
            "hold_period": HOLD_PERIOD,
            "protected_notional": PROTECTED_NOTIONAL,
            "trailing_gap": TRAILING_GAP,
        },
    )
    position = FixedHoldNotionalDeltaTrail(cfg)
    position.add_trade(trades)

    dates = ow.market_dates(START_DATE, END_DATE, exchange=ow.Exchange.NASDAQ)

    result = ow.BackTestCoordinator(
        position=position,
        dates=dates,
        debug=False,
    ).run()

    if return_trades:
        return result, trades
    return result


# -----------------------------------------------------------
#                    METRICS & PLOTS
# -----------------------------------------------------------


def annualized_simple(result: ow.BackTestResult) -> float:
    if not result.dates:
        return float("nan")
    years = max(1e-9, len(result.dates) / TRADING_DAYS)
    return float(result.total_return / years)


def _constant_notional_underlying(close: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=float)
    returns = np.ones_like(close, dtype=float)
    returns[1:] = close[1:] / close[:-1]
    total = np.empty_like(close, dtype=float)
    total[0] = PROTECTED_NOTIONAL
    if close.size > 1:
        total[1:] = PROTECTED_NOTIONAL + np.cumsum(
            PROTECTED_NOTIONAL * (returns[1:] - 1.0)
        )
    return total


def load_underlying_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    hist = yf.Ticker(UNDERLYING_TICKER).history(start=start, end=end)
    if hist.empty:
        return pd.Series(dtype=float)
    series = _constant_notional_underlying(hist["Close"].to_numpy(dtype=float))
    idx = pd.to_datetime(hist.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    return pd.Series(series, index=idx)


def _entry_option(trade: ow.Trade):
    entry_date = trade.entry_data.entry_date
    return trade.entry_data.price_series.prices.get(entry_date.to_iso())


def _entry_iv_delta_strike(trade: ow.Trade):
    opt = _entry_option(trade)
    if not isinstance(opt, ow.Option):
        return None
    iv = getattr(opt, "iv", None)
    if iv is None:
        return None
    delta = getattr(opt, "delta", None)
    strike = getattr(opt, "strike", None)
    return float(iv), delta, strike


def _trades_by_entry(trades: List[ow.Trade]) -> Dict[pd.Timestamp, List[ow.Trade]]:
    entries: Dict[pd.Timestamp, List[ow.Trade]] = {}
    for trade in trades or []:
        ed = trade.entry_data
        if ed is None or ed.entry_date is None:
            continue
        ts = pd.to_datetime(ed.entry_date.to_datetime()).normalize()
        entries.setdefault(ts, []).append(trade)
    return entries


def _short_entry_iv_series(trades: List[ow.Trade]) -> pd.Series:
    by_date = _trades_by_entry(trades)
    values: Dict[pd.Timestamp, float] = {}
    for ts, tlist in by_date.items():
        ivs: List[float] = []
        for t in tlist:
            if t.entry_data.position_type != ow.PositionType.SHORT:
                continue
            entry_meta = _entry_iv_delta_strike(t)
            if entry_meta:
                ivs.append(entry_meta[0])
        if ivs:
            values[ts] = float(np.mean(ivs))
    return pd.Series(values, dtype=float).sort_index()


def _pick_near_far_long(long_meta: List[tuple[float, float | None, float | None]]):
    if not long_meta:
        return None, None
    with_delta = [m for m in long_meta if m[1] is not None]
    if with_delta:
        far = min(with_delta, key=lambda m: abs(m[1]))
        near = max(with_delta, key=lambda m: abs(m[1]))
    else:
        with_strike = [m for m in long_meta if m[2] is not None]
        if with_strike:
            far = min(with_strike, key=lambda m: m[2])
            near = max(with_strike, key=lambda m: m[2])
        else:
            far = near = long_meta[0]
    return far, near


def _entry_skew_series(trades: List[ow.Trade]) -> tuple[pd.Series, pd.Series]:
    by_date = _trades_by_entry(trades)
    far_vals: Dict[pd.Timestamp, float] = {}
    near_vals: Dict[pd.Timestamp, float] = {}
    for ts, tlist in by_date.items():
        short_ivs: List[float] = []
        for t in tlist:
            if t.entry_data.position_type != ow.PositionType.SHORT:
                continue
            entry_meta = _entry_iv_delta_strike(t)
            if entry_meta:
                short_ivs.append(entry_meta[0])
        if not short_ivs:
            continue
        short_iv = float(np.mean(short_ivs))
        long_meta = [
            _entry_iv_delta_strike(t)
            for t in tlist
            if t.entry_data.position_type == ow.PositionType.LONG
        ]
        long_meta = [m for m in long_meta if m is not None]
        if not long_meta:
            continue
        far, near = _pick_near_far_long(long_meta)
        if far:
            far_vals[ts] = float(far[0] - short_iv)
        if near:
            near_vals[ts] = float(near[0] - short_iv)
    far_series = pd.Series(far_vals, dtype=float).sort_index()
    near_series = pd.Series(near_vals, dtype=float).sort_index()
    return far_series, near_series


def _equity_series(result: ow.BackTestResult) -> pd.Series:
    dates = pd.to_datetime([s.date.to_datetime() for s in result.snapshots])
    equity = pd.Series(
        [s.total_equity + s.total_cash for s in result.snapshots], index=dates
    )
    return equity.sort_index()


def _weekly_equity(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    equity = equity.sort_index()
    daily_returns = equity.pct_change().dropna()
    weekly_returns = (1.0 + daily_returns).resample("W-FRI").prod()
    if weekly_returns.empty:
        return pd.Series([equity.iloc[0]], index=[equity.index[0]])
    weekly_equity = equity.iloc[0] * weekly_returns.cumprod()
    weekly_equity = pd.concat(
        [
            pd.Series([equity.iloc[0]], index=[equity.index[0]]),
            weekly_equity,
        ]
    )
    weekly_equity = weekly_equity[~weekly_equity.index.duplicated(keep="first")]
    return weekly_equity.sort_index().ffill()


def _drawdown_from_equity(equity: pd.Series) -> pd.Series:
    if equity.empty or equity.iloc[0] == 0:
        return pd.Series(dtype=float)
    equity_rel = equity / equity.iloc[0]
    return (equity_rel.cummax() - equity_rel) / equity_rel.cummax()


def plot_entry_iv_and_skew(
    trades: List[ow.Trade], out_path: Path = IV_SKEW_PLOT_PATH
) -> Path | None:
    short_iv = _short_entry_iv_series(trades)
    far_skew, near_skew = _entry_skew_series(trades)

    if short_iv.empty and far_skew.empty and near_skew.empty:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    if not short_iv.empty:
        axes[0].plot(
            short_iv.index,
            short_iv,
            color=PALETTE["short_iv"],
            linewidth=1.8,
            label="Short entry IV",
        )
        axes[0].scatter(short_iv.index, short_iv, color=PALETTE["short_iv"], s=12)
    axes[0].set_ylabel("Short Entry IV")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.7)
    if not near_skew.empty:
        axes[1].plot(
            near_skew.index,
            near_skew,
            color=PALETTE["skew_near"],
            linewidth=1.6,
            label="IV_long (near OTM) - IV_short",
        )
    if not far_skew.empty:
        axes[1].plot(
            far_skew.index,
            far_skew,
            color=PALETTE["skew_far"],
            linewidth=1.6,
            linestyle="--",
            label="IV_long (furthest OTM) - IV_short",
        )
    axes[1].set_ylabel("IV Skew (long - short)")
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_strategy(result: ow.BackTestResult, out_path: Path = PLOT_PATH) -> Path | None:
    if not result.snapshots:
        return None

    equity = _equity_series(result)
    if equity.empty or equity.iloc[0] == 0:
        return None

    equity_rel = equity / equity.iloc[0]
    equity_line = equity_rel - 1.0
    drawdown = _drawdown_from_equity(_weekly_equity(equity))

    dates = equity.index
    underlying = load_underlying_series(dates.min(), dates.max())
    underlying = underlying.reindex(equity.index, method="ffill")
    underlying_norm = (
        underlying / underlying.iloc[0]
        if not underlying.empty
        else pd.Series(dtype=float)
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.0]},
    )

    axes[0].plot(
        equity.index,
        equity_line,
        color=PALETTE["equity"],
        linewidth=2.0,
        label="Strategy payoff",
    )
    axes[0].axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("Equity (normalised, start=0)")
    axes[0].set_title(
        "2 Shorts / 2 Longs | hold 120d | ttm 120-180d | trailing delta stop",
        weight="bold",
    )
    axes[0].legend(frameon=False)

    if not underlying_norm.empty:
        axes[1].plot(
            underlying_norm.index,
            underlying_norm,
            color=PALETTE["underlying"],
            linewidth=1.4,
            label="Underlying",
        )
    axes[1].set_ylabel("Underlying (normalised)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].fill_between(
        drawdown.index, drawdown, color=PALETTE["drawdown"], alpha=0.25
    )
    axes[2].plot(
        drawdown.index,
        drawdown,
        color=PALETTE["drawdown"],
        linewidth=1.4,
        label="Drawdown",
    )
    axes[2].set_ylabel("Drawdown (weekly)")
    axes[2].set_xlabel("Date")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_structure() -> Dict[str, float]:
    result, trades = run_structure(return_trades=True)
    equity = _equity_series(result)
    weekly_drawdown = _drawdown_from_equity(_weekly_equity(equity))
    metrics = {
        "annual_simple_return": annualized_simple(result),
        "sharpe": float(result.sharpe),
        "max_drawdown": (
            float(weekly_drawdown.max()) if not weekly_drawdown.empty else float("nan")
        ),
    }
    plot_strategy(result, PLOT_PATH)
    plot_entry_iv_and_skew(trades, IV_SKEW_PLOT_PATH)
    return metrics


def main() -> None:
    metrics = evaluate_structure()
    print(
        f"2 shorts / 2 longs (trailing stop): annual_simple_return={metrics['annual_simple_return']:.4f} | "
        f"sharpe={metrics['sharpe']:.3f} | max_drawdown={metrics['max_drawdown']:.4f}"
    )
    print(f"Equity/drawdown plot written to {PLOT_PATH}")
    print(f"Entry IV + skew plot written to {IV_SKEW_PLOT_PATH}")


if __name__ == "__main__":
    main()
