"""
Monte Carlo sweep over delta-exit thresholds for the multi-stock spread case study.

Ranges:
- short_delta: 0.35 to 0.50
- long_delta: 0.05 to 0.20
- ratio: 1 or 2
- short_delta_lim: 0.50 to 0.85

Args:
  --n_sim: number of combinations to sample
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
from operator import eq, le
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import options_wizard as ow
import pandas as pd

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
SHORT_DELTAS = np.round(np.arange(0.35, 0.501, 0.05), 3)
LONG_DELTAS = np.round(np.arange(0.05, 0.201, 0.05), 3)
RATIOS = [1, 2]
SHORT_DELTA_LIMS = np.round(np.arange(0.5, 0.851, 0.05), 3)

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

MAX_COMBOS = 5000

PLOT_ROOT = Path("tmp")
EQUITY_PLOT_PATH = PLOT_ROOT / "multi_stock_spread_exit_mc_equity.png"
HOLD_HIST_PATH = PLOT_ROOT / "multi_stock_spread_exit_mc_hold_hist.png"


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


def _suffix_from_params(short_delta: float, long_delta: float, otm_ratio: int) -> str:
    def fmt_delta(d: float) -> str:
        return f"{int(round(d * 10_000)):04d}"

    return (
        "multi_stock_spread_mc"
        f"_sd{fmt_delta(short_delta)}"
        f"_ld{fmt_delta(long_delta)}"
        f"_r{otm_ratio}"
    )


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
    trades_template: List[ow.Trade],
    short_abs_delta_limit: float,
) -> Tuple[ow.BackTestResult, List[ow.Trade]]:
    trades = copy.deepcopy(trades_template)
    cfg = ow.BackTestConfig(
        starting_cash=PROTECTED_NOTIONAL,
        start_date=START_DATE,
        end_date=END_DATE,
        kwargs={
            "hold_period": HOLD_PERIOD,
            "protected_notional": PROTECTED_NOTIONAL,
            "short_abs_delta_lower": None,
            "short_abs_delta_upper": None,
            "short_abs_delta_limit": short_abs_delta_limit,
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
    return result, trades


# -----------------------------------------------------------
#                    ANALYSIS HELPERS
# -----------------------------------------------------------


def _equity_series(result: ow.BackTestResult) -> pd.Series:
    dates = pd.to_datetime([s.date.to_datetime() for s in result.snapshots])
    equity = pd.Series(
        [s.total_equity + s.total_cash for s in result.snapshots], index=dates
    )
    return equity.sort_index()


def _avg_hold_days(trades: List[ow.Trade]) -> float:
    durations: List[float] = []
    for trade in trades or []:
        if not (getattr(trade, "_closed", False) or getattr(trade, "_opened", False)):
            continue
        entry = trade.entry_data.entry_date
        exit_date = trade.entry_data.exit_date
        if entry is None or exit_date is None:
            continue
        durations.append(float(exit_date - entry + 1))
    if not durations:
        return float("nan")
    return float(np.mean(durations))


def _build_combos() -> List[Tuple[float, float, int, float]]:
    combos: List[Tuple[float, float, int, float]] = []
    for sd in SHORT_DELTAS:
        for ld in LONG_DELTAS:
            for ratio in RATIOS:
                for lim in SHORT_DELTA_LIMS:
                    if lim <= sd:
                        continue
                    combos.append((float(sd), float(ld), int(ratio), float(lim)))
                    if len(combos) >= MAX_COMBOS:
                        return combos
    return combos


# -----------------------------------------------------------
#                    MAIN
# -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo sweep of delta-exit rules for multi-stock spreads."
    )
    parser.add_argument("--n_sim", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = _build_combos()
    if not combos:
        print("No valid combinations available for the requested ranges.")
        return
    rng = np.random.default_rng(42)
    n_sim = max(1, int(args.n_sim))
    if n_sim >= len(combos):
        sampled = combos
    else:
        sampled_idx = rng.choice(len(combos), size=n_sim, replace=False)
        sampled = [combos[i] for i in sampled_idx]

    equity_series_list: List[pd.Series] = []
    equity_labels: List[str] = []
    avg_hold_list: List[float] = []

    for short_delta, long_delta, ratio, short_lim in sampled:
        suffix = _suffix_from_params(short_delta, long_delta, ratio)
        trades_template = load_trades(
            short_delta=short_delta,
            long_delta=long_delta,
            otm_ratio=ratio,
            suffix=suffix,
        )
        if not trades_template:
            continue
        result, trades = run_structure(
            trades_template=trades_template,
            short_abs_delta_limit=short_lim,
        )
        equity = _equity_series(result)
        if not equity.empty:
            equity_series_list.append(equity / equity.iloc[0] - 1.0)
            equity_labels.append(
                f"sd={short_delta:.2f} ld={long_delta:.2f} r={ratio} lim={short_lim:.2f}"
            )
        avg_hold_list.append(_avg_hold_days(trades))

    if equity_series_list:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        for series, label in zip(equity_series_list, equity_labels):
            ax.plot(series.index, series.values, alpha=0.35, linewidth=1.0, label=label)
        ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title("Multi-Stock Spread Monte Carlo: Equity Curves", weight="bold")
        ax.set_ylabel("Equity (normalised, start=0)")
        ax.set_xlabel("Date")
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
        EQUITY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(EQUITY_PLOT_PATH, bbox_inches="tight")
        plt.close(fig)

    avg_hold_clean = [v for v in avg_hold_list if np.isfinite(v)]
    if avg_hold_clean:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            avg_hold_clean,
            bins=12,
            color="#1f77b4",
            alpha=0.75,
            edgecolor="#111827",
            linewidth=0.8,
        )
        ax.set_title("Average Hold Duration per Trade", weight="bold")
        ax.set_xlabel("Average hold (days)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        HOLD_HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(HOLD_HIST_PATH, bbox_inches="tight")
        plt.close(fig)

    print(
        f"Monte Carlo complete: {len(equity_series_list)} cases | "
        f"equity plot: {EQUITY_PLOT_PATH} | hold histogram: {HOLD_HIST_PATH}"
    )


if __name__ == "__main__":
    main()
