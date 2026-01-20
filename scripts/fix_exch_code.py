"""
Standardize EXCH_CODE values for commodity futures parquet files.

Usage:
    python scripts/fix_exch_code.py            # dry-run, prints planned changes
    python scripts/fix_exch_code.py --apply    # rewrite files in place

The script reads CMDTY_PATH from .env (or --path) and rewrites EXCH_CODE for
known ticks that should live on a single exchange. A backup copy is stored in
CMDTY_PATH/backup_exch_code before any overwrite.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


# Canonical exchange for each tick (looked up from CME/CBOT product pages)
TICK_EXCHANGE = {
    # CBOT (CME Group)
    "O": "CBOT",  # Oats futures (Chicago Board of Trade)
    "SM": "CBOT",  # Soybean Meal futures (CBOT)
    "S": "CBOT",  # Soybean futures (CBOT)
    "W": "CBOT",  # Soft Red Winter Wheat futures (CBOT)
}


def standardize_file(
    path: Path, tick: str, exchange: str, apply: bool, backup_dir: Path
) -> None:
    df = pl.read_parquet(path)
    current_codes = df.get_column("EXCH_CODE").unique().to_list()

    fixed = df.with_columns(pl.lit(exchange).alias("EXCH_CODE"))

    if apply:
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / path.name
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        fixed.write_parquet(path)

    print(
        f"{path.name}: {current_codes} -> {exchange} ({'written' if apply else 'dry-run'})"
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Normalize EXCH_CODE across commodity parquet files."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(os.getenv("CMDTY_PATH", "")),
        help="Directory containing *_FUT.parquet files (defaults to CMDTY_PATH).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in place; otherwise just report planned changes.",
    )
    args = parser.parse_args()

    if not args.path or not args.path.exists():
        raise SystemExit("CMDTY_PATH/--path does not exist; set it first.")

    backup_dir = args.path / "backup_exch_code"

    for file in sorted(args.path.glob("*_FUT.parquet")):
        tick = file.name.split("_")[0]
        if tick not in TICK_EXCHANGE:
            continue
        standardize_file(file, tick, TICK_EXCHANGE[tick], args.apply, backup_dir)


if __name__ == "__main__":
    main()
