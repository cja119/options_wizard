"""
Run a grid of put-spread index parameter combinations serially.

Each combination:
 - builds and runs the put-spread pipeline
 - reconstructs trades and backtests with FixedHoldNotional
 - saves a unique pickle with params + BackTestResult
Progress is shown in the terminal via tqdm.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
from itertools import product
from operator import eq, le
from pathlib import Path
from typing import Dict, Tuple

import dill
import options_wizard as ow
from tqdm import tqdm

# -----------------------------------------------------------
#                    CONFIG
# -----------------------------------------------------------

START_DATE = ow.DateObj(2008, 1, 1)
END_DATE = ow.DateObj(2020, 12, 31)
STOCK = "NDQ"
DELTA_TOL = 0.02
PROTECTED_NOTIONAL = 1_000_000
SPREAD_CAPTURE_OVERRIDE = 0.5

SAVE_ROOT = Path("tmp/put_spread_sweeps_f50")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
#                    DATA CLASSES
# -----------------------------------------------------------


@dataclass(frozen=True)
class ComboSpec:
    ratio: Tuple[int, int]
    expiry_range: Tuple[int, int]
    short_delta: float
    long_delta: float
    hold_period: int
    spread_capture_override: float = SPREAD_CAPTURE_OVERRIDE

    @property
    def suffix(self) -> str:
        def fmt_delta(d: float) -> str:
            return f"{int(round(d * 10_000)):04d}"

        return (
            f"s50_r{self.ratio[0]}x{self.ratio[1]}"
            f"_exp{self.expiry_range[0]}-{self.expiry_range[1]}"
            f"_sd{fmt_delta(self.short_delta)}"
            f"_ld{fmt_delta(self.long_delta)}"
            f"_hp{self.hold_period}"
        )

    @property
    def label(self) -> str:
        return (
            f"{self.ratio[0]}x{self.ratio[1]} | "
            f"exp {self.expiry_range[0]}-{self.expiry_range[1]} | "
            f"sd {self.short_delta:.3f} | "
            f"ld {self.long_delta:.3f} | "
            f"hold {self.hold_period}"
        )

    def as_dict(self) -> Dict:
        return asdict(self)


# -----------------------------------------------------------
#                    CORE HELPERS
# -----------------------------------------------------------


def build_specs(combo: ComboSpec) -> list[ow.OptionsTradeSpec]:
    """Construct ATM/OTM specs for a combo."""
    lo, hi = combo.expiry_range

    def _ttm_fn(t, low=lo, high=hi):
        return (t >= low) and (t <= high)

    def _short_delta_fn(d, target=combo.short_delta):
        return (abs(d) >= target - DELTA_TOL) and (abs(d) <= target + DELTA_TOL)

    def _long_delta_fn(d, target=combo.long_delta):
        return (abs(d) >= target - DELTA_TOL) and (abs(d) <= target + DELTA_TOL)

    atm_spec = ow.OptionsTradeSpec(
        call_put=ow.OptionType.PUT,
        ttm=_ttm_fn,
        lm_fn=lambda k: True,
        abs_delta=_short_delta_fn,
        entry_min="perc_spread",
        max_hold_period=combo.hold_period,
        position=-float(combo.ratio[0]),
    )

    otm_spec = ow.OptionsTradeSpec(
        call_put=ow.OptionType.PUT,
        ttm=_ttm_fn,
        lm_fn=lambda k: True,
        abs_delta=_long_delta_fn,
        entry_min="perc_spread",
        max_hold_period=combo.hold_period,
        position=float(combo.ratio[1]),
    )

    return [atm_spec, otm_spec]


def run_single_combo(combo: ComboSpec) -> tuple[bool, str, str]:
    """Run pipeline + backtest for one combo."""
    suffix = combo.suffix

    try:
        specs = build_specs(combo)
        universe = ow.Universe([STOCK])
        pipeline = ow.Pipeline(
            universe=universe,
            save_type=ow.SaveType.PICKLE,
            saves=[ow.SaveFrames.STRAT],
        )

        kwargs = {
            "max_date": END_DATE.to_pl(),
            "keep_col": ["call_put", "ttm", "n_missing"],
            "keep_oper": [eq, le, le],
            "keep_val": ["p", 150, 0],
            "specs": specs,
            "hold_period": combo.hold_period,
            "protected_notional": PROTECTED_NOTIONAL,
            "suffix": suffix,
        }

        ow.add_idx_spread_methods_opt(pipeline, kwargs)
        pipeline.run()

        strat = ow.StratType.load(STOCK, save_type=ow.SaveType.PICKLE, suffix=suffix)
        ptf = partial(
            ow.Trade,
            transaction_cost_model=ow.TransactionCostModel.SPREAD,
            accounting_type=ow.AccountingConvention.CASH,
        )
        trades = strat.reconstruct(ptf)

        # Explicitly set spread capture on each trade (request: override = 1.0)
        for trade in trades or []:
            try:
                trade._spread_capture = combo.spread_capture_override
            except Exception:
                pass

        cfg = ow.BackTestConfig(
            starting_cash=PROTECTED_NOTIONAL,
            start_date=START_DATE,
            end_date=END_DATE,
            kwargs={"hold_period": combo.hold_period},
        )
        position = ow.FixedHoldNotional(cfg)
        position.add_trade(trades)

        dates = ow.market_dates(START_DATE, END_DATE, exchange=ow.Exchange.NASDAQ)

        result = ow.BackTestCoordinator(
            position=position,
            dates=dates,
            debug=False,
        ).run()

        out_path = SAVE_ROOT / f"{suffix}.pkl"
        payload = {"params": combo.as_dict(), "suffix": suffix, "result": result}
        with out_path.open("wb") as f:
            dill.dump(payload, f)

        return True, suffix, str(out_path)

    except Exception as exc:  # noqa: BLE001
        return False, suffix, str(exc)


def generate_combos() -> list[ComboSpec]:
    ratios = [(1, 2), (1, 3)]
    expiry_ranges = [(30, 60), (60, 90), (90, 120)]
    short_deltas = [0.35, 0.4, 0.45]
    long_deltas = [0.075, 0.1, 0.15, 0.2]
    holds = [30, 60]

    combos = []
    for hold in holds:
        allowed_expiries = expiry_ranges if hold == 30 else [(60, 90), (90, 120)]
        for ratio, exp, sd, ld in product(
            ratios, allowed_expiries, short_deltas, long_deltas
        ):
            combos.append(
                ComboSpec(
                    ratio=ratio,
                    expiry_range=exp,
                    short_delta=sd,
                    long_delta=ld,
                    hold_period=hold,
                )
            )
    return combos


def main() -> None:
    combos = generate_combos()
    print(f"Running {len(combos)} combinations serially...")

    successes = 0
    failures: list[tuple[str, str]] = []
    for combo in tqdm(combos, desc="Put spread combos"):
        ok, suffix, detail = run_single_combo(combo)
        if ok:
            successes += 1
        else:
            failures.append((suffix, detail))

    print(f"Completed {successes}/{len(combos)} combos.")
    if failures:
        print("Failures:")
        for suffix, err in failures:
            print(f" - {suffix}: {err}")


if __name__ == "__main__":
    main()
