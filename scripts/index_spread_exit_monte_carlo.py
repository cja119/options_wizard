"""
Monte Carlo sweep over short delta limits for the index spread case study.

Ranges:
- short_delta_lim: 0.45 to 0.75 (step 0.05)

Args:
  --short_delta: abs delta for the short put
  --long_delta: abs delta for the long put
  --otm_ratio: number of long contracts per short
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
from operator import eq, le
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import options_wizard as ow
import pandas as pd

from backtest.position.options.fixed_hold_delta_exit import (
    FixedHoldNotionalDeltaExitShortLimitRoll,
)


# -----------------------------------------------------------
#                    CONFIG
# -----------------------------------------------------------

START_DATE = ow.DateObj(2008, 1, 1)
END_DATE = ow.DateObj(2020, 12, 31)
STOCK = "NDQ"
DELTA_TOL = 0.02
PROTECTED_NOTIONAL = 1_000_000
SPREAD_CAPTURE_OVERRIDE = 0.5
EXPIRY_RANGE = (120, 180)
HOLD_PERIOD = 120
TRADING_DAYS = 252
SHORT_DELTA_LIMS = np.round(np.arange(0.45, 0.751, 0.05), 3)

PLOT_ROOT = Path("tmp")
EQUITY_PLOT_PATH = PLOT_ROOT / "index_spread_exit_mc_equity.png"
HOLD_HIST_PATH = PLOT_ROOT / "index_spread_exit_mc_hold_hist.png"


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


# -----------------------------------------------------------
#                    CORE BACKTEST
# -----------------------------------------------------------


def _suffix_from_params(short_delta: float, long_delta: float, otm_ratio: int) -> str:
    def fmt_delta(d: float) -> str:
        return f"{int(round(d * 10_000)):04d}"

    return (
        "index_spread_mc"
        f"_sd{fmt_delta(short_delta)}"
        f"_ld{fmt_delta(long_delta)}"
        f"_r{otm_ratio}"
    )


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


def load_trades(
    short_delta: float, long_delta: float, otm_ratio: int, suffix: str
) -> List[ow.Trade]:
    specs = build_specs(short_delta, long_delta, otm_ratio)
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
    position = FixedHoldNotionalDeltaExitShortLimitRoll(cfg)
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


# -----------------------------------------------------------
#                    MAIN
# -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo sweep of delta-exit rules for index spreads."
    )
    parser.add_argument("--short_delta", type=float, default=0.35)
    parser.add_argument("--long_delta", type=float, default=0.20)
    parser.add_argument("--otm_ratio", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = _suffix_from_params(args.short_delta, args.long_delta, args.otm_ratio)
    trades_template = load_trades(
        short_delta=args.short_delta,
        long_delta=args.long_delta,
        otm_ratio=args.otm_ratio,
        suffix=suffix,
    )
    if not trades_template:
        print("No trades available for the requested parameters.")
        return

    equity_series_list: List[pd.Series] = []
    equity_labels: List[str] = []
    avg_hold_list: List[float] = []

    for short_lim in SHORT_DELTA_LIMS:
        if short_lim <= abs(args.short_delta):
            continue
        result, trades = run_structure(
            trades_template=trades_template,
            short_abs_delta_limit=short_lim,
        )
        equity = _equity_series(result)
        if not equity.empty:
            equity_series_list.append(equity / equity.iloc[0] - 1.0)
            equity_labels.append(
                f"sd={args.short_delta:.2f} ld={args.long_delta:.2f} "
                f"r={args.otm_ratio} lim={short_lim:.2f}"
            )
        avg_hold_list.append(_avg_hold_days(trades))

    if equity_series_list:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        for series, label in zip(equity_series_list, equity_labels):
            ax.plot(series.index, series.values, alpha=0.35, linewidth=1.0, label=label)
        ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title("Index Spread Monte Carlo: Equity Curves", weight="bold")
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
