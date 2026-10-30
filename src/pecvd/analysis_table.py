"""Build a reproducible process-analysis table from a combined sensor table.

This module performs no project-data discovery. Callers must pass a DataFrame
or explicitly name an input file through the companion CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DERIVED_COLUMNS = (
    "elapsed_from_declared_start_seconds",
    "elapsed_from_first_sample_seconds",
    "step_number",
    "previous_step",
    "step_progress",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def build_analysis_table(
    frame: pd.DataFrame,
    *,
    process_column: str = "process_id",
    timestamp_column: str = "timestamp",
    step_column: str = "step_name",
    declared_start_column: str | None = None,
) -> pd.DataFrame:
    """Return a sorted table with process- and step-relative coordinates.

    Rows with equal timestamps retain their original order. A step occurrence
    is a contiguous block inside one process, so a repeated step name later in
    the same process receives a new ``step_number``.

    Declared-start elapsed time is never guessed. Supply
    ``declared_start_column`` or provide an existing numeric
    ``elapsed_from_declared_start_seconds`` column.
    """
    if frame.empty:
        raise ValueError("Input table is empty")

    required = (process_column, timestamp_column, step_column)
    _require_columns(frame, required)

    if frame[process_column].isna().any():
        raise ValueError(f"{process_column} contains missing values")
    if frame[step_column].isna().any() or (
        frame[step_column].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"{step_column} contains missing or blank values")

    result = frame.copy(deep=True)
    result["__source_order"] = np.arange(len(result), dtype=np.int64)
    parsed_timestamp = pd.to_datetime(
        result[timestamp_column], errors="coerce", format="mixed"
    )
    if parsed_timestamp.isna().any():
        raise ValueError(f"{timestamp_column} contains invalid timestamps")
    result[timestamp_column] = parsed_timestamp

    result = result.sort_values(
        [process_column, timestamp_column, "__source_order"],
        kind="stable",
    ).reset_index(drop=True)

    process_groups = result.groupby(process_column, sort=False, dropna=False)
    first_timestamp = process_groups[timestamp_column].transform("min")
    result["elapsed_from_first_sample_seconds"] = (
        result[timestamp_column] - first_timestamp
    ).dt.total_seconds()

    if declared_start_column is not None:
        _require_columns(result, (declared_start_column,))
        declared_start = pd.to_datetime(
            result[declared_start_column], errors="coerce", format="mixed"
        )
        if declared_start.isna().any():
            raise ValueError(
                f"{declared_start_column} contains invalid timestamps"
            )
        unique_start_counts = declared_start.groupby(
            result[process_column], sort=False
        ).nunique(dropna=False)
        if (unique_start_counts != 1).any():
            raise ValueError(
                f"{declared_start_column} must contain one value per process"
            )
        result["elapsed_from_declared_start_seconds"] = (
            result[timestamp_column] - declared_start
        ).dt.total_seconds()
    elif "elapsed_from_declared_start_seconds" in result.columns:
        elapsed = pd.to_numeric(
            result["elapsed_from_declared_start_seconds"], errors="coerce"
        )
        if elapsed.isna().any():
            raise ValueError(
                "elapsed_from_declared_start_seconds must be numeric"
            )
        result["elapsed_from_declared_start_seconds"] = elapsed
    else:
        raise ValueError(
            "Declared start cannot be inferred. Supply declared_start_column "
            "or an existing elapsed_from_declared_start_seconds column."
        )

    previous_row_step = process_groups[step_column].shift(1)
    starts_new_step = previous_row_step.isna() | (
        result[step_column] != previous_row_step
    )
    result["step_number"] = starts_new_step.groupby(
        result[process_column], sort=False
    ).cumsum().astype(np.int64)

    segment_keys = [result[process_column], result["step_number"]]
    result["previous_step"] = previous_row_step.where(starts_new_step, pd.NA)
    result["previous_step"] = result["previous_step"].groupby(
        segment_keys, sort=False
    ).transform("first")

    segment_position = result.groupby(segment_keys, sort=False).cumcount()
    segment_size = result.groupby(segment_keys, sort=False)[step_column].transform(
        "size"
    )
    denominator = (segment_size - 1).where(segment_size > 1, 1)
    result["step_progress"] = (
        segment_position / denominator
    ).where(segment_size > 1, 0.0)

    return result.drop(columns="__source_order")


def write_analysis_table(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write atomically, refusing to replace an existing file by default."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
