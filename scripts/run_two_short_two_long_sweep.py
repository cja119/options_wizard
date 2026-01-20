"""
Run a grid of 4-leg put structures (2 shorts, 2 longs) serially.

Rules:
 - Shorts: two distinct deltas, ratio fixed at 1 each
 - Longs: two distinct deltas, ratios vary; total longs > total shorts and at least
   one long ratio is 2
 - Entry minimises time-to-maturity inside 120-180d; hold 120d
 - Saves a unique pickle with params + BackTestResult
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
from itertools import product
from operator import eq, le
from pathlib import Path
from typing import Dict, Tuple
import random
import os
from concurrent import futures

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

EXPIRY_RANGE = (120, 180)
HOLD_PERIOD = 120

SHORT1_DELTAS = [0.35, 0.4, 0.45]
SHORT2_DELTAS = [0.45, 0.5, 0.55]
LONG1_DELTAS = [0.075, 0.1, 0.15]
LONG2_DELTAS = [0.15, 0.2, 0.25]
LONG1_RATIOS = [1, 2, 3]
LONG2_RATIOS = [1, 2]

SAVE_ROOT = Path("tmp/two_short_two_long_sweeps")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
#                    DATA CLASSES
# -----------------------------------------------------------


@dataclass(frozen=True)
class MultiLegSpec:
    expiry_range: Tuple[int, int]
    hold_period: int
    short1_delta: float
    short2_delta: float
    long1_delta: float
    long2_delta: float
    long1_ratio: int
    long2_ratio: int
    spread_capture_override: float = SPREAD_CAPTURE_OVERRIDE

    @property
    def suffix(self) -> str:
        def fmt_delta(d: float) -> str:
            return f"{int(round(d * 10_000)):04d}"

        return (
            f"2s2l_exp{self.expiry_range[0]}-{self.expiry_range[1]}"
            f"_sd{fmt_delta(self.short1_delta)}-{fmt_delta(self.short2_delta)}"
            f"_ld{fmt_delta(self.long1_delta)}-{fmt_delta(self.long2_delta)}"
            f"_lr{self.long1_ratio}x{self.long2_ratio}"
            f"_hp{self.hold_period}"
        )

    @property
    def label(self) -> str:
        return (
            f"2s2l exp {self.expiry_range[0]}-{self.expiry_range[1]} | "
            f"sd1 {self.short1_delta:.3f} sd2 {self.short2_delta:.3f} | "
            f"ld1 {self.long1_delta:.3f} ld2 {self.long2_delta:.3f} | "
            f"lr {self.long1_ratio}x{self.long2_ratio} | "
            f"hold {self.hold_period}"
        )

    def as_dict(self) -> Dict:
        return asdict(self)


# -----------------------------------------------------------
#                    CORE HELPERS
# -----------------------------------------------------------


def _delta_filter(target: float):
    return lambda d, tgt=target: (abs(d) >= tgt - DELTA_TOL) and (
        abs(d) <= tgt + DELTA_TOL
    )


def build_specs(combo: MultiLegSpec) -> list[ow.OptionsTradeSpec]:
    """Construct specs for two shorts + two longs with TTM minimisation."""
    lo, hi = combo.expiry_range

    def _ttm_fn(t, low=lo, high=hi):
        return (t >= low) and (t <= high)

    return [
        ow.OptionsTradeSpec(
            call_put=ow.OptionType.PUT,
            ttm=_ttm_fn,
            lm_fn=lambda k: True,
            abs_delta=_delta_filter(combo.short1_delta),
            entry_min="ttm",
            max_hold_period=combo.hold_period,
            position=-1.0,
        ),
        ow.OptionsTradeSpec(
            call_put=ow.OptionType.PUT,
            ttm=_ttm_fn,
            lm_fn=lambda k: True,
            abs_delta=_delta_filter(combo.short2_delta),
            entry_min="ttm",
            max_hold_period=combo.hold_period,
            position=-1.0,
        ),
        ow.OptionsTradeSpec(
            call_put=ow.OptionType.PUT,
            ttm=_ttm_fn,
            lm_fn=lambda k: True,
            abs_delta=_delta_filter(combo.long1_delta),
            entry_min="ttm",
            max_hold_period=combo.hold_period,
            position=float(combo.long1_ratio),
        ),
        ow.OptionsTradeSpec(
            call_put=ow.OptionType.PUT,
            ttm=_ttm_fn,
            lm_fn=lambda k: True,
            abs_delta=_delta_filter(combo.long2_delta),
            entry_min="ttm",
            max_hold_period=combo.hold_period,
            position=float(combo.long2_ratio),
        ),
    ]


def run_single_combo(combo: MultiLegSpec) -> tuple[bool, str, str]:
    """Run pipeline then backtest for one combo (sequential fallback)."""
    ok, suffix, detail, future = run_pipeline_and_enqueue_backtest(combo, None)
    if not ok:
        return False, suffix, detail
    result_future = future.result()
    if result_future[0]:
        return result_future[0], result_future[1], result_future[2]
    return result_future[0], result_future[1], result_future[2]


def _run_backtest(combo: MultiLegSpec, trades, suffix: str) -> tuple[bool, str, str]:
    try:
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


def run_pipeline_and_enqueue_backtest(
    combo: MultiLegSpec, executor: futures.Executor | None
):
    """Run pipeline serially and optionally schedule backtest asynchronously."""
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
            "keep_val": ["p", combo.expiry_range[1], 0],
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

        for trade in trades or []:
            try:
                trade._spread_capture = combo.spread_capture_override
            except Exception:
                pass

        if executor is None:
            # Synchronous path for compatibility
            res = _run_backtest(combo, trades, suffix)
            fut: futures.Future = futures.Future()
            fut.set_result(res)
            return res[0], suffix, res[2], fut

        future = executor.submit(_run_backtest, combo, trades, suffix)
        return True, suffix, "", future

    except Exception as exc:  # noqa: BLE001
        return False, suffix, str(exc), None


def generate_combos() -> list[MultiLegSpec]:
    combos: list[MultiLegSpec] = []
    for l1_delta, l2_delta, l1_ratio, l2_ratio, s1_delta, s2_delta in product(
        LONG1_DELTAS,
        LONG2_DELTAS,
        LONG1_RATIOS,
        LONG2_RATIOS,
        SHORT1_DELTAS,
        SHORT2_DELTAS,
    ):
        long_total = l1_ratio + l2_ratio
        long_has_two = (l1_ratio == 2) or (l2_ratio == 2)
        if not long_has_two or long_total <= 2:
            continue
        combos.append(
            MultiLegSpec(
                expiry_range=EXPIRY_RANGE,
                hold_period=HOLD_PERIOD,
                short1_delta=s1_delta,
                short2_delta=s2_delta,
                long1_delta=l1_delta,
                long2_delta=l2_delta,
                long1_ratio=l1_ratio,
                long2_ratio=l2_ratio,
            )
        )
    return combos


def main() -> None:
    combos = generate_combos()
    # Randomly drop ~50% to reduce runtime; deterministic seed for repeatability
    rnd = random.Random(42)
    target = max(1, len(combos) // 2)
    combos = rnd.sample(combos, k=target)

    print(f"Running {len(combos)} combinations serially (two shorts / two longs)...")

    successes = 0
    failures: list[tuple[str, str]] = []
    pending: list[futures.Future] = []

    max_workers = max(1, min(4, (os.cpu_count() or 2)))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for combo in tqdm(combos, desc="Two-short / two-long combos"):
            ok, suffix, detail, fut = run_pipeline_and_enqueue_backtest(combo, executor)
            if not ok:
                failures.append((suffix, detail))
                continue
            if fut:
                pending.append(fut)

        for fut in futures.as_completed(pending):
            try:
                ok, suffix, detail = fut.result()
            except Exception as exc:  # noqa: BLE001
                ok, suffix, detail = False, "unknown", str(exc)
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
