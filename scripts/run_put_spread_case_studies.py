"""
Run targeted 1x2 put-spread case studies focused on holding to maturity.

Combinations:
- short deltas: 0.35, 0.40, 0.45
- long deltas: 0.075, 0.10, 0.15, 0.20
- cases:
    * hold 90d, expiry 90-120d
    * hold 120d, expiry 120-180d
    * hold 180d, expiry 180-360d

Contract selection minimizes time-to-maturity (ttm) within the allowed window
to bias toward holding contracts to expiry (instead of minimizing bid-ask spread).
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

SAVE_ROOT = Path("tmp/put_spread_case_studies_f50")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
#                    DATA CLASSES
# -----------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    expiry_range: Tuple[int, int]
    hold_period: int

    @property
    def label(self) -> str:
        return (
            f"exp{self.expiry_range[0]}-{self.expiry_range[1]}_hold{self.hold_period}"
        )


@dataclass(frozen=True)
class ComboSpec:
    short_delta: float
    long_delta: float
    case: CaseSpec

    @property
    def suffix(self) -> str:
        def fmt_delta(d: float) -> str:
            return f"{int(round(d * 10_000)):04d}"

        return (
            f"s50_r1x2_{self.case.label}"
            f"_sd{fmt_delta(self.short_delta)}"
            f"_ld{fmt_delta(self.long_delta)}"
        )

    @property
    def label(self) -> str:
        return (
            f"1x2 | {self.case.label} | "
            f"sd {self.short_delta:.3f} | ld {self.long_delta:.3f}"
        )

    def as_dict(self) -> Dict:
        return asdict(self)


# -----------------------------------------------------------
#                    CORE HELPERS
# -----------------------------------------------------------


def build_specs(combo: ComboSpec) -> list[ow.OptionsTradeSpec]:
    """Construct short/long legs targeting low TTM within the window."""
    lo, hi = combo.case.expiry_range

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
        entry_min="ttm",  # pick lowest maturity available in window
        max_hold_period=combo.case.hold_period,
        position=-1.0,
    )

    otm_spec = ow.OptionsTradeSpec(
        call_put=ow.OptionType.PUT,
        ttm=_ttm_fn,
        lm_fn=lambda k: True,
        abs_delta=_long_delta_fn,
        entry_min="ttm",
        max_hold_period=combo.case.hold_period,
        position=2.0,
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
            "keep_val": ["p", combo.case.expiry_range[1], 0],
            "specs": specs,
            "hold_period": combo.case.hold_period,
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

        for trade in trades or []:
            try:
                trade._spread_capture = SPREAD_CAPTURE_OVERRIDE
            except Exception:
                pass

        cfg = ow.BackTestConfig(
            starting_cash=PROTECTED_NOTIONAL,
            start_date=START_DATE,
            end_date=END_DATE,
            kwargs={"hold_period": combo.case.hold_period},
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
    short_deltas = [0.35, 0.4, 0.45]
    long_deltas = [0.075, 0.1, 0.15, 0.2]
    cases = [
        CaseSpec(expiry_range=(90, 120), hold_period=90),
        CaseSpec(expiry_range=(120, 180), hold_period=120),
        CaseSpec(expiry_range=(180, 360), hold_period=180),
    ]

    combos = []
    for sd, ld, case in product(short_deltas, long_deltas, cases):
        combos.append(ComboSpec(short_delta=sd, long_delta=ld, case=case))
    return combos


def main() -> None:
    combos = generate_combos()
    print(f"Running {len(combos)} case-study combinations serially...")

    successes = 0
    failures: list[tuple[str, str]] = []
    for combo in tqdm(combos, desc="Put spread case studies"):
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
