"""
Run a single short/long put structure across a stock basket and export equity series.

Args:
  --short_delta: abs delta for the short put
  --long_delta: abs delta for the long put
  --otm_ratio: number of long contracts per short
  --short_delta_lim: upper abs-delta limit for the short
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from operator import eq, le
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import options_wizard as ow
import pandas as pd
import yfinance as yf

from backtest.position.options.fixed_hold_delta_exit import (
    StockFixedHoldNotionalDeltaExitShortLimitRoll,
)

# -----------------------------------------------------------
#                    CONFIG
# -----------------------------------------------------------

START_DATE = ow.DateObj(2008, 1, 1)
END_DATE = ow.DateObj(2020, 12, 31)
DELTA_TOL = 0.02
PROTECTED_NOTIONAL = 1_000_000
SPREAD_CAPTURE_OVERRIDE = 0.5
EXPIRY_RANGE = (120, 180)
HOLD_PERIOD = 120
TRADING_DAYS = 252
SHORT_DELTA_LIM = 0.55

# Pre-set basket for the spread universe (edit as needed).
SPREAD_TICKS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "TSLA",
    "CSCO",
]
N_CONST = max(200, len(SPREAD_TICKS))

PLOT_ROOT = Path("tmp")
IV_SKEW_ROOT = Path("tmp")
EQUITY_CSV_ROOT = Path("tmp")

PALETTE = {
    "equity": "#1f77b4",
    "underlying": "#111827",
    "drawdown": "#d9534f",
    "combined": "#f59e0b",
    "delta": "#f59e0b",
    "short_iv": "#1b9e77",
    "skew_near": "#d62728",
    "skew_far": "#9467bd",
}


# -----------------------------------------------------------
#                    SPEC HELPERS
# -----------------------------------------------------------


def _delta_filter(target: float):
    return lambda d, tgt=target: (abs(d) >= tgt - DELTA_TOL) and (
        abs(d) <= tgt + DELTA_TOL
    )


def _ttm_filter(t: float) -> bool:
    return (t >= EXPIRY_RANGE[0]) and (t <= EXPIRY_RANGE[1])


def build_specs(
    short_delta: float, long_delta: float, otm_ratio: int
) -> List[ow.OptionsTradeSpec]:
    specs = []
    specs.append(
        ow.OptionsTradeSpec(
            call_put=ow.OptionType.PUT,
            ttm=_ttm_filter,
            lm_fn=lambda k: True,
            abs_delta=_delta_filter(short_delta),
            entry_min="ttm",
            max_hold_period=HOLD_PERIOD,
            position=-1.0,
        )
    )
    specs.append(
        ow.OptionsTradeSpec(
            call_put=ow.OptionType.PUT,
            ttm=_ttm_filter,
            lm_fn=lambda k: True,
            abs_delta=_delta_filter(long_delta),
            entry_min="ttm",
            max_hold_period=HOLD_PERIOD,
            position=float(otm_ratio),
        )
    )
    return specs


def _resolve_spread_ticks() -> List[str]:
    ticks = [tick.strip().upper() for tick in SPREAD_TICKS if tick.strip()]
    if not ticks:
        return []
    try:
        universe = ow.Universe(list(ticks))
        universe.check_ticks()
        return universe.ticks or ticks
    except Exception:
        return ticks


# -----------------------------------------------------------
#                    CORE BACKTEST
# -----------------------------------------------------------


def _fmt_delta(delta: float) -> str:
    return f"{int(round(delta * 10_000)):04d}"


def _suffix_from_params(short_delta: float, long_delta: float, otm_ratio: int) -> str:
    return (
        "multi_stock_case_study"
        f"_sd{_fmt_delta(short_delta)}"
        f"_ld{_fmt_delta(long_delta)}"
        f"_r{otm_ratio}"
    )


def _csv_path_from_params(
    short_delta: float, long_delta: float, otm_ratio: int
) -> Path:
    suffix = _suffix_from_params(short_delta, long_delta, otm_ratio)
    return EQUITY_CSV_ROOT / f"{suffix}_equity.csv"


def _plot_path_from_params(
    short_delta: float, long_delta: float, otm_ratio: int
) -> Path:
    suffix = _suffix_from_params(short_delta, long_delta, otm_ratio)
    return PLOT_ROOT / f"{suffix}.png"


def _iv_skew_path_from_params(
    short_delta: float, long_delta: float, otm_ratio: int
) -> Path:
    suffix = _suffix_from_params(short_delta, long_delta, otm_ratio)
    return IV_SKEW_ROOT / f"{suffix}_iv_skew.png"


def _strat_path(tick: str, suffix: str) -> Path:
    return Path("tmp") / f"strategy_{tick}_{suffix}.pkl"


def _load_or_build_strats(
    specs: List[ow.OptionsTradeSpec], suffix: str, ticks: List[str]
) -> Dict[str, ow.StratType]:
    if not ticks:
        return {}

    def _load_existing() -> Dict[str, ow.StratType]:
        strats: Dict[str, ow.StratType] = {}
        for tick in ticks:
            if not _strat_path(tick, suffix).exists():
                continue
            try:
                strats[tick] = ow.StratType.load(
                    tick, save_type=ow.SaveType.PICKLE, suffix=suffix
                )
            except Exception:
                continue
        return strats

    if all(_strat_path(tick, suffix).exists() for tick in ticks):
        return _load_existing()

    universe = ow.Universe(ticks)
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
        "n_const": N_CONST,
    }

    ow.add_put_spread_methods(pipeline, kwargs)
    pipeline.run()

    return _load_existing()


def load_trades(
    short_delta: float, long_delta: float, otm_ratio: int, suffix: str
) -> List[ow.Trade]:
    ticks = _resolve_spread_ticks()
    if not ticks:
        return []

    specs = build_specs(short_delta, long_delta, otm_ratio)
    strats = _load_or_build_strats(specs, suffix, ticks)
    ptf = partial(
        ow.Trade,
        transaction_cost_model=ow.TransactionCostModel.SPREAD,
        accounting_type=ow.AccountingConvention.CASH,
    )
    trades: List[ow.Trade] = []
    for tick in ticks:
        strat = strats.get(tick)
        if strat is None or strat.isempty():
            continue
        trades.extend(strat.reconstruct(ptf))
    for trade in trades or []:
        try:
            trade._spread_capture = SPREAD_CAPTURE_OVERRIDE
        except Exception:
            pass
    return trades


def run_structure(
    short_delta: float,
    long_delta: float,
    otm_ratio: int,
    suffix: str,
    return_trades: bool = False,
    short_delta_lim: float = SHORT_DELTA_LIM,
) -> ow.BackTestResult | tuple[ow.BackTestResult, List[ow.Trade]]:
    trades = load_trades(short_delta, long_delta, otm_ratio, suffix)
    cfg = ow.BackTestConfig(
        starting_cash=PROTECTED_NOTIONAL,
        start_date=START_DATE,
        end_date=END_DATE,
        kwargs={
            "hold_period": HOLD_PERIOD,
            "protected_notional": PROTECTED_NOTIONAL,
            "short_abs_delta_lower": None,
            "short_abs_delta_upper": None,
            "short_abs_delta_limit": short_delta_lim,
            "long_abs_delta_lower": None,
            "long_abs_delta_upper": None,
        },
    )
    position = StockFixedHoldNotionalDeltaExitShortLimitRoll(cfg)
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


def _constant_notional_underlying(close: np.ndarray, notional: float) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=float)
    returns = np.ones_like(close, dtype=float)
    returns[1:] = close[1:] / close[:-1]
    total = np.empty_like(close, dtype=float)
    total[0] = notional
    if close.size > 1:
        total[1:] = notional + np.cumsum(notional * (returns[1:] - 1.0))
    return total


def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _is_rate_limited(err: Exception) -> bool:
    resp = getattr(err, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    return "429" in str(err)


def _with_backoff(fn, attempts: int = 5, base_delay: float = 1.0):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limited(exc) and i < attempts:
                time.sleep(base_delay * i)
                continue
            raise


def _load_close_series(tick: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    hist = _with_backoff(
        lambda: yf.Ticker(tick).history(
            start=start, end=end, auto_adjust=False, interval="1d"
        )
    )
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    idx = _normalize_index(hist.index)
    return pd.Series(hist["Close"].to_numpy(dtype=float), index=idx)


def load_underlying_series(
    start: pd.Timestamp, end: pd.Timestamp, ticks: List[str] | None = None
) -> pd.Series:
    tick_list = ticks or _resolve_spread_ticks()
    tick_list = [tick for tick in tick_list if tick]
    if not tick_list:
        return pd.Series(dtype=float)

    closes: Dict[str, pd.Series] = {}
    for tick in tick_list:
        try:
            series = _load_close_series(tick, start, end)
        except Exception:
            continue
        if not series.empty:
            closes[tick] = series

    if not closes:
        return pd.Series(dtype=float)

    notional_per_tick = PROTECTED_NOTIONAL / len(closes)
    series_list = []
    for series in closes.values():
        total = _constant_notional_underlying(
            series.to_numpy(dtype=float), notional_per_tick
        )
        series_list.append(pd.Series(total, index=series.index))

    frame = pd.concat(series_list, axis=1).sort_index().ffill()
    combined = frame.sum(axis=1, min_count=1).dropna()
    return combined


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


def _price_on_date(trade: ow.Trade, date: ow.DateObj):
    price_series = trade.entry_data.price_series
    px = price_series.get(date)
    if px is not None:
        return px
    try:
        nearest = trade._nearest_date_below(date)
    except Exception:
        return None
    return price_series.get(nearest)


def _portfolio_delta_series(result: ow.BackTestResult) -> pd.Series:
    if not result.snapshots:
        return pd.Series(dtype=float)
    values: Dict[pd.Timestamp, float] = {}
    for snapshot in result.snapshots:
        total_delta = 0.0
        total_contracts = 0.0
        for trade in snapshot.trade_equities.keys():
            entry = trade.entry_data
            if entry is None:
                continue
            px = _price_on_date(trade, snapshot.date)
            delta = getattr(px, "delta", None) if px is not None else None
            if delta is None:
                continue
            pos_sign = 1.0 if entry.position_type == ow.PositionType.LONG else -1.0
            position_size = float(entry.position_size)
            held_contracts = position_size * pos_sign
            total_delta += held_contracts * float(delta)
            total_contracts += abs(held_contracts)
        ts = pd.to_datetime(snapshot.date.to_datetime()).normalize()
        if total_contracts > 0:
            avg_delta = total_delta / total_contracts
            values[ts] = float(np.clip(avg_delta, -1.0, 1.0))
        else:
            values[ts] = 0.0
    return pd.Series(values, dtype=float).sort_index()


def equity_underlying_frame(result: ow.BackTestResult) -> pd.DataFrame:
    equity = _equity_series(result)
    if equity.empty:
        return pd.DataFrame(columns=["underlying_equity", "strategy_payoff"])
    dates = equity.index
    underlying = load_underlying_series(dates.min(), dates.max())
    underlying = underlying.reindex(dates, method="ffill")
    frame = pd.DataFrame(
        {"underlying_equity": underlying, "strategy_payoff": equity}, index=dates
    )
    frame.index.name = "date"
    return frame.sort_index()


def export_equity_csv(
    result: ow.BackTestResult, out_path: Path
) -> Path | None:
    frame = equity_underlying_frame(result)
    if frame.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path)
    return out_path


def plot_entry_iv_and_skew(
    trades: List[ow.Trade], out_path: Path
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


def plot_strategy(result: ow.BackTestResult, out_path: Path) -> Path | None:
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
    portfolio_delta = _portfolio_delta_series(result)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.0, 1.0]},
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
        "2 Shorts / 2 Longs | hold 120d | ttm 120-180d", weight="bold"
    )
    axes[0].legend(frameon=False)

    if not underlying_norm.empty:
        axes[1].plot(
            underlying_norm.index,
            underlying_norm,
            color=PALETTE["underlying"],
            linewidth=1.4,
            label="Underlying basket",
        )
    axes[1].set_ylabel("Underlying (normalised)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    if not portfolio_delta.empty:
        axes[2].plot(
            portfolio_delta.index,
            portfolio_delta,
            color=PALETTE["delta"],
            linewidth=1.4,
            label="Portfolio delta (avg per contract)",
        )
    axes[2].axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[2].set_ylabel("Delta (avg per contract)")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.3)

    axes[3].fill_between(
        drawdown.index, drawdown, color=PALETTE["drawdown"], alpha=0.25
    )
    axes[3].plot(
        drawdown.index,
        drawdown,
        color=PALETTE["drawdown"],
        linewidth=1.4,
        label="Drawdown",
    )
    axes[3].set_ylabel("Drawdown (weekly)")
    axes[3].set_xlabel("Date")
    axes[3].set_ylim(bottom=0)
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------------------------------------
#                    MAIN
# -----------------------------------------------------------


def evaluate_structure(
    short_delta: float, long_delta: float, otm_ratio: int, short_delta_lim: float
) -> Dict[str, float] | tuple[Dict[str, float], Path | None]:
    suffix = _suffix_from_params(short_delta, long_delta, otm_ratio)
    csv_path = _csv_path_from_params(short_delta, long_delta, otm_ratio)
    plot_path = _plot_path_from_params(short_delta, long_delta, otm_ratio)
    iv_skew_path = _iv_skew_path_from_params(short_delta, long_delta, otm_ratio)
    result, trades = run_structure(
        short_delta,
        long_delta,
        otm_ratio,
        suffix,
        short_delta_lim=short_delta_lim,
        return_trades=True,
    )
    equity = _equity_series(result)
    weekly_drawdown = _drawdown_from_equity(_weekly_equity(equity))
    metrics = {
        "annual_simple_return": annualized_simple(result),
        "sharpe": float(result.sharpe),
        "max_drawdown": (
            float(weekly_drawdown.max()) if not weekly_drawdown.empty else float("nan")
        ),
    }

    plot_strategy(result, plot_path)
    plot_entry_iv_and_skew(trades, iv_skew_path)
    csv_path = export_equity_csv(result, csv_path)
    return metrics, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single short/long put structure on a stock basket."
    )
    parser.add_argument("--short_delta", type=float, default=0.35)
    parser.add_argument("--long_delta", type=float, default=0.20)
    parser.add_argument("--otm_ratio", type=int, default=2)
    parser.add_argument("--short_delta_lim", type=float, default=SHORT_DELTA_LIM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, csv_path = evaluate_structure(
        short_delta=args.short_delta,
        long_delta=args.long_delta,
        otm_ratio=args.otm_ratio,
        short_delta_lim=args.short_delta_lim,
    )
    print(
        f"Short/long: short_delta={args.short_delta:.3f} | long_delta={args.long_delta:.3f} | "
        f"otm_ratio={args.otm_ratio} | short_delta_lim={args.short_delta_lim:.3f} | "
        f"annual_simple_return={metrics['annual_simple_return']:.4f} | "
        f"sharpe={metrics['sharpe']:.3f} | max_drawdown={metrics['max_drawdown']:.4f}"
    )
    print(
        "Equity/drawdown plot written to "
        f"{_plot_path_from_params(args.short_delta, args.long_delta, args.otm_ratio)}"
    )
    print(
        "Entry IV + skew plot written to "
        f"{_iv_skew_path_from_params(args.short_delta, args.long_delta, args.otm_ratio)}"
    )
    if csv_path:
        print(f"Equity series CSV written to {csv_path}")


if __name__ == "__main__":
    main()
