"""
Analyze results from the case-study sweep in tmp/put_spread_case_studies.

Rankings:
 1) Highest Sharpe ratio
 2) Lowest max drawdown
 3) Largest 1-month return (approx. 21 trading days)
 4) Lowest combined volatility when blended with the underlying at constant notional
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

SAVE_ROOT = Path("tmp/put_spread_case_studies_f50")
PLOT_DIR = SAVE_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

UNDERLYING_TICKER = "^IXIC"
PROTECTED_NOTIONAL = 1_000_000
TRADING_DAYS = 252
ROLLING_WINDOW = 21  # ~1 trading month
COVID_UP_START = pd.Timestamp("2020-03-23")
COVID_UP_END = pd.Timestamp("2020-09-02")


@dataclass
class LoadedResult:
    suffix: str
    params: Dict[str, Any]
    sharpe: float
    max_drawdown: float
    max_1m_return: float
    covid_upside_return: float
    equity: np.ndarray
    dates: np.ndarray

    def label(self) -> str:
        exp = self.params.get("case", {}).get("expiry_range", [])
        hold = self.params.get("case", {}).get("hold_period")
        sd = self.params.get("short_delta")
        ld = self.params.get("long_delta")
        return f"exp {exp[0]}-{exp[1]} sd {sd:.3f} ld {ld:.3f} hold {hold}"


@dataclass
class CombinedResult:
    base: LoadedResult
    combined_equity: np.ndarray
    combined_dates: np.ndarray
    combined_vol: float

    def label(self) -> str:
        return f"{self.base.label()} | comb vol {self.combined_vol:.4f}"


def max_one_month_return(log_returns: np.ndarray) -> float:
    if log_returns.size < ROLLING_WINDOW:
        return float("nan")
    cumsum = np.concatenate(([0.0], np.cumsum(log_returns)))
    window_sum = cumsum[ROLLING_WINDOW:] - cumsum[:-ROLLING_WINDOW]
    return float(np.exp(window_sum).max() - 1.0)


def window_simple_return(
    equity: np.ndarray, dates: np.ndarray, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    if equity.size == 0 or dates.size == 0:
        return float("nan")
    idx = pd.to_datetime(dates)
    mask = (idx >= start) & (idx <= end)
    if not mask.any():
        return float("nan")
    eq_slice = equity[mask]
    if eq_slice.size == 0:
        return float("nan")
    base = eq_slice[0]
    if base == 0:
        return float("nan")
    return float(eq_slice[-1] / base - 1.0)


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
    covid_ret = window_simple_return(equity, dates, COVID_UP_START, COVID_UP_END)

    return LoadedResult(
        suffix=suffix,
        params=params,
        sharpe=float(result.sharpe),
        max_drawdown=float(result.max_drawdown),
        max_1m_return=max_one_month_return(log_returns),
        covid_upside_return=covid_ret,
        equity=equity,
        dates=dates,
    )


def plot_top(records: List[LoadedResult], title: str, filename: str) -> None:
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
            f"| max_1m={rec.max_1m_return:.4f} "
            f"| covid_up={rec.covid_upside_return:.4f}"
        )


def normalize_index(dates: np.ndarray) -> pd.DatetimeIndex:
    idx = pd.to_datetime(dates)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York")
        idx = idx.tz_localize(None)
    return idx.normalize()


def constant_notional_underlying(close: np.ndarray, notional: float) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=float)
    returns = np.ones_like(close, dtype=float)
    returns[1:] = close[1:] / close[:-1]
    total = np.empty_like(close, dtype=float)
    total[0] = notional
    if close.size > 1:
        total[1:] = notional + np.cumsum(notional * (returns[1:] - 1.0))
    return total


def constant_notional_equity(
    equity: np.ndarray, notional: float, leverage: float = 1.0
) -> np.ndarray:
    if equity.size == 0:
        return np.array([], dtype=float)
    return notional + leverage * (equity - equity[0])


def load_underlying_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    hist = yf.Ticker(UNDERLYING_TICKER).history(start=start, end=end)
    if hist.empty:
        return pd.Series(dtype=float)
    total = constant_notional_underlying(
        hist["Close"].to_numpy(dtype=float), PROTECTED_NOTIONAL
    )
    idx = normalize_index(hist.index)
    return pd.Series(total, index=idx)


def combined_vol_records(
    results: List[LoadedResult],
    underlying: pd.Series,
    leverage: float = 1.0,
) -> List[CombinedResult]:
    if underlying.empty:
        return []

    combined: List[CombinedResult] = []
    for rec in results:
        if rec.equity.size == 0 or rec.dates.size == 0:
            continue

        eq_scaled = constant_notional_equity(rec.equity, PROTECTED_NOTIONAL, leverage)
        eq_idx = normalize_index(rec.dates)

        shared = eq_idx.intersection(underlying.index)
        if shared.empty:
            continue

        eq_aligned = pd.Series(eq_scaled, index=eq_idx).loc[shared].to_numpy()
        und_aligned = underlying.loc[shared].to_numpy()

        combined_equity = eq_aligned + und_aligned - PROTECTED_NOTIONAL
        combined_returns = np.diff(combined_equity) / PROTECTED_NOTIONAL

        if combined_returns.size == 0:
            continue

        std = combined_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
        vol = float("nan") if std == 0.0 else float(std)

        combined.append(
            CombinedResult(
                base=rec,
                combined_equity=combined_equity,
                combined_dates=shared.to_pydatetime(),
                combined_vol=vol,
            )
        )

    combined.sort(key=lambda r: (np.isnan(r.combined_vol), r.combined_vol))
    return combined[:5]


def plot_combined_top(records: List[CombinedResult], underlying: pd.Series) -> None:
    if not records:
        return
    plt.figure(figsize=(12, 6))
    for rec in records:
        plt.plot(rec.combined_dates, rec.combined_equity, label=rec.base.label())
    plt.plot(
        underlying.index,
        underlying.values,
        color="black",
        linestyle="--",
        label="Underlying (constant notional)",
    )
    plt.title("Lowest combined volatility with underlying (constant notional)")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "case_top_by_combined_vol.png")
    plt.close()


def print_combined(records: List[CombinedResult]) -> None:
    if not records:
        print("\nNo combined volatility results could be calculated.")
        return

    print(
        "\nTop 5 lowest combined volatility (options + underlying, constant notional)"
    )
    for rec in records:
        print(
            f"{rec.base.suffix}: combined_vol={rec.combined_vol:.4f} "
            f"| strat_sharpe={rec.base.sharpe:.3f} "
            f"| max_dd={rec.base.max_drawdown:.4f}"
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

    plot_top(by_sharpe, "Top 5 by Sharpe", "case_top_by_sharpe.png")
    plot_top(by_drawdown, "Top 5 by Lowest Max DD", "case_top_by_drawdown.png")
    plot_top(by_one_month, "Top 5 by 1M Return", "case_top_by_1m_return.png")

    print_top(results, "sharpe", True, "Top 5 Sharpe")
    print_top(results, "max_drawdown", False, "Top 5 Lowest Max Drawdown")
    print_top(results, "max_1m_return", True, "Top 5 Largest 1M Return")
    print_top(
        results,
        "covid_upside_return",
        False,
        "Top 5 Worst COVID upside performance (2020-03-23 to 2020-09-02)",
    )

    # --- Combined volatility with underlying at constant notional
    date_bounds = [
        (pd.to_datetime(rec.dates).min(), pd.to_datetime(rec.dates).max())
        for rec in results
        if rec.dates.size > 0
    ]
    if date_bounds:
        start = min(pair[0] for pair in date_bounds)
        end = max(pair[1] for pair in date_bounds)
        underlying = load_underlying_series(start, end)
        combined = combined_vol_records(results, underlying)
        plot_combined_top(combined, underlying)
        print_combined(combined)
    else:
        print("\nCould not determine date bounds for combined volatility calculation.")

    print(f"\nPlots written to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
