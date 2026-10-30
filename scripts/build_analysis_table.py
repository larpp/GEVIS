"""Offline CLI for building data/all_processes_analysis.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pecvd.analysis_table import build_analysis_table, write_analysis_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible PECVD analysis table locally."
    )
    parser.add_argument("--input", type=Path, default=Path("data/all_processes.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/all_processes_analysis.csv")
    )
    parser.add_argument("--process-column", default="process_id")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--step-column", default="step_name")
    parser.add_argument(
        "--declared-start-column",
        default=None,
        help=(
            "Column containing one declared-start timestamp per process. "
            "Required unless the input already has "
            "elapsed_from_declared_start_seconds."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    source = pd.read_csv(args.input)
    analysis = build_analysis_table(
        source,
        process_column=args.process_column,
        timestamp_column=args.timestamp_column,
        step_column=args.step_column,
        declared_start_column=args.declared_start_column,
    )
    destination = write_analysis_table(
        analysis,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"Analysis table written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
