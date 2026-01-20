"""
Load sweep pickle files, rank strategies, and plot top performers.

Rankings:
 1) Highest Sharpe ratio
 2) Lowest max drawdown
 3) Largest 1-month return (approx. 21 trading days)

Three plots are written to tmp/put_spread_sweeps/plots overlaying
the top 5 combos for each ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import dill
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SAVE_ROOT = Path("tmp/put_spread_sweeps_f50")
PLOT_DIR = SAVE_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW = 21  # ~1 trading month


@dataclass
class LoadedResult:
    suffix: str
    params: Dict[str, Any]
    sharpe: float
    max_drawdown: float
    max_1m_return: float
    equity: np.ndarray
    dates: np.ndarray

    def label(self) -> str:
        ratio = self.params.get("ratio", [])
        exp = self.params.get("expiry_range", [])
        hold = self.params.get("hold_period")
        sd = self.params.get("short_delta")
        ld = self.params.get("long_delta")
        return (
            f"{ratio[0]}x{ratio[1]} exp {exp[0]}-{exp[1]} "
            f"sd {sd:.3f} ld {ld:.3f} hold {hold}"
        )


def max_one_month_return(log_returns: np.ndarray) -> float:
    if log_returns.size < ROLLING_WINDOW:
        return float("nan")
    cumsum = np.concatenate(([0.0], np.cumsum(log_returns)))
    window_sum = cumsum[ROLLING_WINDOW:] - cumsum[:-ROLLING_WINDOW]
    return float(np.exp(window_sum).max() - 1.0)


def load_result(path: Path) -> LoadedResult | None:
    try:
        with path.open("rb") as f:
            payload = dill.load(f)
    except Exception:
        return None

    result = payload.get("result")
    if result is None or not getattr(result, "snapshots", None):
        return None
    params = payload.get("params", {})
    suffix = payload.get("suffix", path.stem)

    equity = np.array(
        [snap.total_equity + snap.total_cash for snap in result.snapshots],
        dtype=float,
    )
    dates = np.array([snap.date.to_datetime() for snap in result.snapshots])
    log_returns = np.array(result.returns, dtype=float)

    return LoadedResult(
        suffix=suffix,
        params=params,
        sharpe=float(result.sharpe),
        max_drawdown=float(result.max_drawdown),
        max_1m_return=max_one_month_return(log_returns),
        equity=equity,
        dates=dates,
    )


def plot_top(records: List[LoadedResult], key: str, title: str, filename: str) -> None:
    plt.figure(figsize=(12, 6))
    for rec in records[:5]:
        curve = rec.equity
        if curve.size == 0:
            continue
        norm_curve = curve / curve[0]
        plt.plot(rec.dates, norm_curve, label=rec.label())
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.close()


def print_top(records: List[LoadedResult], key: str, reverse: bool, title: str) -> None:
    print(f"\n{title}")
    sorted_recs = sorted(records, key=lambda r: getattr(r, key), reverse=reverse)[:5]
    for rec in sorted_recs:
        val = getattr(rec, key)
        print(
            f"{rec.suffix}: {key}={val:.4f} "
            f"| sharpe={rec.sharpe:.3f} "
            f"| max_dd={rec.max_drawdown:.4f} "
            f"| max_1m={rec.max_1m_return:.4f}"
        )


def main() -> None:
    pkls = sorted(SAVE_ROOT.glob("*.pkl"))
    if not pkls:
        print(f"No pickle files found in {SAVE_ROOT}")
        return

    results: List[LoadedResult] = []
    for path in tqdm(pkls, desc="Loading pickles"):
        rec = load_result(path)
        if rec is not None:
            results.append(rec)

    if not results:
        print("No results could be loaded.")
        return

    by_sharpe = sorted(results, key=lambda r: r.sharpe, reverse=True)
    by_drawdown = sorted(results, key=lambda r: r.max_drawdown)
    by_one_month = sorted(results, key=lambda r: r.max_1m_return, reverse=True)
    plot_top(by_sharpe, "sharpe", "Top 5 by Sharpe", "top_by_sharpe.png")
    plot_top(
        by_drawdown, "max_drawdown", "Top 5 by Lowest Max DD", "top_by_drawdown.png"
    )
    plot_top(
        by_one_month, "max_1m_return", "Top 5 by 1M Return", "top_by_1m_return.png"
    )

    print_top(results, "sharpe", True, "Top 5 Sharpe")
    print_top(results, "max_drawdown", False, "Top 5 Lowest Max Drawdown")
    print_top(results, "max_1m_return", True, "Top 5 Largest 1M Return")

    print(f"\nPlots written to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
