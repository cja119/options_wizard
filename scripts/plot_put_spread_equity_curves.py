"""
Plot all put-spread sweep equity curves on a single chart.

Loads every pickle in tmp/put_spread_sweeps/, extracts the equity path
from BackTestResult.snapshots, and overlays the curves with a shared
legend beneath the plot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple, List

import dill
import matplotlib.pyplot as plt

# Ensure project modules used by dill are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def load_equity_curve(path: Path) -> Tuple[str, List, List]:
    """Return (label, dates, equity_values) for a sweep pickle."""
    payload = dill.load(path.open("rb"))
    label = payload.get("suffix", path.stem) if isinstance(payload, dict) else path.stem
    result = payload.get("result", payload) if isinstance(payload, dict) else payload

    snapshots = getattr(result, "snapshots", None) or []
    dates = [s.date.to_datetime() for s in snapshots]
    equities = [s.total_equity + s.total_cash for s in snapshots]
    return label, dates, equities


def plot_equity_curves(paths: Iterable[Path]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 9))

    handles = []
    labels = []
    for p in paths:
        label, dates, eq = load_equity_curve(p)
        if not dates:
            continue
        (line,) = ax.plot(dates, eq, alpha=0.5, linewidth=1.1)
        handles.append(line)
        labels.append(label)

    ax.set_title("Put-Spread Sweep Equity Curves", weight="bold", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.2)

    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=8,
            frameon=False,
            fontsize=7,
        )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def main() -> None:
    root = PROJECT_ROOT / "tmp" / "put_spread_sweeps"
    pkls = sorted(root.glob("*.pkl"))
    if not pkls:
        print(f"No .pkl files found in {root}")
        return
    plot_equity_curves(pkls)


if __name__ == "__main__":
    main()
