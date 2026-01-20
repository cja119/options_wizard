"""
Compare annualized simple performance for rolling a 1x2 put spread at two tenors.

Assumptions:
- 1 short put at ~45 delta vs 2 long puts at ~15 delta.
- Entry contracts prioritize the shortest time-to-maturity inside the allowed window
    (via entry_min="ttm") to bias toward holding through expiry. Hold periods are set to
    the upper end of each window so the backtest can carry the trade to maturity.
- Results use MTM accounting with spread-based transaction costs and no compounding; we
    report simple annualized return (total_return divided by years in sample).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from operator import eq, le
from pathlib import Path
from typing import Dict, Tuple

import options_wizard as ow
import matplotlib.pyplot as plt
import pandas as pd

START_DATE = ow.DateObj(2008, 1, 1)
END_DATE = ow.DateObj(2020, 12, 31)
STOCK = "NDQ"
DELTA_TOL = 0.02
PROTECTED_NOTIONAL = 1_000_000
SPREAD_CAPTURE_OVERRIDE = 0.0

SHORT_DELTA = 0.45
LONG_DELTA = 0.15
COVID_UP_START = pd.Timestamp("2020-03-23")
COVID_UP_END = pd.Timestamp("2020-09-02")


@dataclass(frozen=True)
class RollSpec:
    name: str
    expiry_range: Tuple[int, int]
    hold_period: int

    @property
    def label(self) -> str:
        return f"{self.name} | exp {self.expiry_range[0]}-{self.expiry_range[1]} | hold {self.hold_period}"


ROLL_SPECS = [
    RollSpec(name="roll30d_exp30-60", expiry_range=(30, 60), hold_period=30),
    RollSpec(name="roll180d_exp180-360", expiry_range=(180, 360), hold_period=180),
]


def build_specs(spec: RollSpec) -> list[ow.OptionsTradeSpec]:
    low, high = spec.expiry_range

    def _ttm_fn(t, lo=low, hi=high):
        return (t >= lo) and (t <= hi)

    def _short_delta_fn(d, target=SHORT_DELTA):
        return (abs(d) >= target - DELTA_TOL) and (abs(d) <= target + DELTA_TOL)

    def _long_delta_fn(d, target=LONG_DELTA):
        return (abs(d) >= target - DELTA_TOL) and (abs(d) <= target + DELTA_TOL)

    short_spec = ow.OptionsTradeSpec(
        call_put=ow.OptionType.PUT,
        ttm=_ttm_fn,
        lm_fn=lambda k: True,
        abs_delta=_short_delta_fn,
        entry_min="ttm",  # prefer nearest-expiring contract inside window
        max_hold_period=spec.hold_period,
        position=-1.0,
    )

    long_spec = ow.OptionsTradeSpec(
        call_put=ow.OptionType.PUT,
        ttm=_ttm_fn,
        lm_fn=lambda k: True,
        abs_delta=_long_delta_fn,
        entry_min="ttm",
        max_hold_period=spec.hold_period,
        position=2.0,
    )

    return [short_spec, long_spec]


def run_backtest(spec: RollSpec) -> ow.BackTestResult:
    specs = build_specs(spec)
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
        "keep_val": ["p", spec.expiry_range[1], 0],
        "specs": specs,
        "hold_period": spec.hold_period,
        "protected_notional": PROTECTED_NOTIONAL,
        "suffix": spec.name,
    }

    ow.add_idx_spread_methods_opt(pipeline, kwargs)
    pipeline.run()

    strat = ow.StratType.load(STOCK, save_type=ow.SaveType.PICKLE, suffix=spec.name)
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
        kwargs={"hold_period": spec.hold_period},
    )
    position = ow.FixedHoldNotional(cfg)
    position.add_trade(trades)

    dates = ow.market_dates(START_DATE, END_DATE, exchange=ow.Exchange.NASDAQ)

    return ow.BackTestCoordinator(
        position=position,
        dates=dates,
        debug=False,
    ).run()


def annualized_simple(result: ow.BackTestResult) -> float:
    years = max(1e-9, len(result.dates) / 252)
    return result.total_return / years


def window_simple_return(
    result: ow.BackTestResult, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    if not result.snapshots:
        return float("nan")
    dates = pd.to_datetime([s.date.to_datetime() for s in result.snapshots])
    equity = pd.Series(
        [s.total_equity + s.total_cash for s in result.snapshots], index=dates
    )
    window = equity.loc[(equity.index >= start) & (equity.index <= end)]
    if window.empty or window.iloc[0] == 0:
        return float("nan")
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def plot_equity_curves(
    results: Dict[str, ow.BackTestResult], out_path: Path | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, res in results.items():
        dates = [d.to_datetime() for d in res.dates]
        equity = [s.total_equity + s.total_cash for s in res.snapshots]
        ax.plot(dates, equity, label=name)

    ax.set_title("1x2 Put Spread Equity (hold to maturity)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved equity curves to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    results: Dict[str, ow.BackTestResult] = {}
    annualized: Dict[str, float] = {}
    covid_up: Dict[str, float] = {}

    for spec in ROLL_SPECS:
        res = run_backtest(spec)
        results[spec.name] = res
        annualized[spec.name] = annualized_simple(res)
        covid_up[spec.name] = window_simple_return(res, COVID_UP_START, COVID_UP_END)
        print(
            f"{spec.label}: total_return={res.total_return:.4f} | "
            f"simple_annualized={annualized[spec.name]:.4f} | "
            f"covid_up={covid_up[spec.name]:.4f}"
        )

    diff = annualized[ROLL_SPECS[0].name] - annualized[ROLL_SPECS[1].name]
    print(f"Annualized simple return difference (30d - 180d): {diff:.4f}")

    covid_diff = covid_up[ROLL_SPECS[0].name] - covid_up[ROLL_SPECS[1].name]
    print(f"COVID upside return difference (30d - 180d): {covid_diff:.4f}")

    plot_equity_curves(results, Path("tmp/compare_put_spread_rolls.png"))


if __name__ == "__main__":
    main()
