"""Small, fail-closed helpers for the local GEVIS FDC review app."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd
from .answer_display import english_output


RUN_COLUMN = "process_key"
STEP_COLUMN = "step_label"
ORDER_COLUMN = "row_in_process"
RUNTIME_CELL_IDS = ("03-types", "03-load", "03-function")


def derive_step_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with ``step_label`` derived from one-hot columns when needed."""
    if STEP_COLUMN in frame.columns:
        return frame
    columns = [
        column for column in frame.columns
        if column.startswith("step_name_") and column != "step_name_nan"
    ]
    if not columns:
        raise ValueError("CSV has no step_label or step_name_* columns.")
    values = frame[columns].fillna(0).to_numpy()
    if not np.allclose(values.sum(axis=1), 1):
        raise ValueError("step_name_* columns must be one-hot.")
    result = frame.copy()
    labels = np.asarray([column.removeprefix("step_name_") for column in columns], dtype=object)
    result[STEP_COLUMN] = labels[values.argmax(axis=1)]
    return result


def validate_fdc_frame(frame: pd.DataFrame) -> None:
    required = {RUN_COLUMN, STEP_COLUMN, ORDER_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required CSV columns are missing: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("CSV is empty.")
    if frame[RUN_COLUMN].isna().any() or frame[STEP_COLUMN].isna().any():
        raise ValueError("Process or Step identifiers contain missing values.")


def load_local_fdc_csv(path: str | Path) -> pd.DataFrame:
    """Load the configured local CSV without printing or exporting its rows."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Local FDC CSV not found: {csv_path}")
    frame = derive_step_label(pd.read_csv(csv_path))
    validate_fdc_frame(frame)
    return frame


def select_step_rows(frame: pd.DataFrame, process_id: Any, step: str) -> pd.DataFrame:
    rows = frame.loc[
        frame[RUN_COLUMN].eq(process_id) & frame[STEP_COLUMN].eq(step)
    ].sort_values(ORDER_COLUMN)
    if rows.empty:
        raise ValueError("No data for the selected process and Step.")
    return rows


def window_at_progress(rows: pd.DataFrame, window_length: int, progress: float) -> pd.DataFrame:
    """Select one exact model window at a relative position in the original Step."""
    if window_length < 1 or len(rows) < window_length:
        raise ValueError("Selected Step is shorter than the model window.")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")
    start = round(progress * (len(rows) - window_length))
    return rows.iloc[start:start + window_length]


def window_position_ratio(rows: pd.DataFrame, window_length: int, start: int = 0) -> float:
    """Return the model window's position using original Step coordinates when available."""
    if not 0 <= start <= len(rows) - window_length:
        raise ValueError("window start is out of range")
    if "row_in_step" in rows and "step_row_count" in rows:
        row_start = float(rows.iloc[start]["row_in_step"])
        total = float(rows.iloc[start]["step_row_count"])
        return float(np.clip(row_start / max(1.0, total - window_length), 0.0, 1.0))
    return start / max(1, len(rows) - window_length)


def matching_position_window(
    rows: pd.DataFrame,
    window_length: int,
    target_ratio: float,
) -> pd.DataFrame:
    """Select the available original window nearest to a target Step position."""
    if len(rows) < window_length:
        raise ValueError("Step is shorter than the model window")
    candidates = range(0, len(rows) - window_length + 1)
    start = min(
        candidates,
        key=lambda value: abs(window_position_ratio(rows, window_length, value) - target_ratio),
    )
    return rows.iloc[start:start + window_length]


def _one_hot_label(rows: pd.DataFrame, prefix: str, excluded: tuple[str, ...] = ()) -> str:
    columns = [
        column for column in rows.columns
        if column.startswith(prefix) and column not in excluded
    ]
    if not columns:
        raise ValueError(f"Comparison-context columns are missing: {prefix}*")
    values = rows[columns].fillna(0).to_numpy()
    if not np.allclose(values.sum(axis=1), 1):
        raise ValueError(f"{prefix}* columns must be one-hot.")
    labels = np.asarray([column.removeprefix(prefix) for column in columns], dtype=object)
    selected = labels[values.argmax(axis=1)]
    if len(set(map(str, selected))) != 1:
        raise ValueError("Comparison context changes within the selected Step.")
    return str(selected[0])


def comparison_signature(rows: pd.DataFrame) -> tuple[str, str]:
    return (
        _one_hot_label(rows, "source_group_", ("source_group_nan",)),
        _one_hot_label(rows, "pm_cycle_"),
    )


def compatible_reference_ids(
    frame: pd.DataFrame,
    process_id: Any,
    step: str,
    minimum_rows: int,
) -> tuple[Any, ...]:
    """Find every other process with enough rows at the requested Step."""
    matches: list[Any] = []
    candidates = frame.loc[frame[STEP_COLUMN].eq(step)]
    for candidate_id, rows in candidates.groupby(RUN_COLUMN, sort=True):
        if candidate_id == process_id or len(rows) < minimum_rows:
            continue
        matches.append(candidate_id)
    return tuple(matches)


def choose_typical_reference_ids(
    frame: pd.DataFrame,
    process_id: Any,
    step: str,
    minimum_rows: int,
    sensors: tuple[str, ...],
    count: int = 3,
) -> tuple[tuple[Any, ...], int]:
    """Choose representatives closest to the comparison group's median sensor state."""
    references = compatible_reference_ids(frame, process_id, step, minimum_rows)
    if not references:
        raise ValueError("No comparable reference Process is available")
    if count < 1:
        raise ValueError("count must be positive")
    rows = frame.loc[frame[RUN_COLUMN].isin(references) & frame[STEP_COLUMN].eq(step)]
    available = [sensor for sensor in sensors if sensor in rows.columns]
    if not available:
        raise ValueError("No model sensors are available for reference selection")
    features = rows.groupby(RUN_COLUMN, sort=True)[available].mean()
    median = features.median(axis=0)
    scale = features.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    score = ((features - median) / scale).abs().mean(axis=1)
    selected = tuple(score.sort_values(kind="stable").index[:count].tolist())
    return selected, len(references)


def choose_typical_reference_id(
    frame: pd.DataFrame,
    process_id: Any,
    step: str,
    minimum_rows: int,
    sensors: tuple[str, ...],
) -> tuple[Any, int]:
    selected, total = choose_typical_reference_ids(
        frame, process_id, step, minimum_rows, sensors, count=1
    )
    return selected[0], total


def step_summary(process_rows: pd.DataFrame, steps: tuple[str, ...]) -> pd.DataFrame:
    """Summarize the six original-CSV Step segments in execution order."""
    records = []
    for step in steps:
        rows = process_rows.loc[process_rows[STEP_COLUMN].eq(step)].sort_values(ORDER_COLUMN)
        if rows.empty:
            continue
        duration = (
            float(pd.to_numeric(rows["step_elapsed_seconds"], errors="coerce").max())
            if "step_elapsed_seconds" in rows else float("nan")
        )
        records.append({
            "step": step,
            "samples": len(rows),
            "start_order": int(rows[ORDER_COLUMN].iloc[0]),
            "end_order": int(rows[ORDER_COLUMN].iloc[-1]),
            "duration_seconds": duration,
        })
    return pd.DataFrame.from_records(records)


def reference_envelope(
    frame: pd.DataFrame,
    reference_ids: tuple[Any, ...],
    step: str,
    sensor: str,
    points: int = 101,
) -> pd.DataFrame:
    """Build a display-only percentile envelope from original comparable traces."""
    traces = []
    for process_id in reference_ids:
        rows = select_step_rows(frame, process_id, step)
        values = pd.to_numeric(rows[sensor], errors="coerce").to_numpy(float)
        if len(values) < 2 or not np.isfinite(values).all():
            continue
        progress = np.linspace(0.0, 1.0, points)
        traces.append(np.interp(progress, np.linspace(0.0, 1.0, len(values)), values))
    if not traces:
        raise ValueError("No finite comparable traces are available")
    values = np.vstack(traces)
    return pd.DataFrame({
        "progress": np.linspace(0.0, 100.0, points),
        "lower": np.percentile(values, 10, axis=0),
        "median": np.percentile(values, 50, axis=0),
        "upper": np.percentile(values, 90, axis=0),
    })


def fdc_deviation_grid(
    frame: pd.DataFrame,
    sensor: str,
    steps: tuple[str, ...],
) -> pd.DataFrame:
    """Return conventional per-Step standardized mean differences for overview only."""
    grouped = (
        frame.loc[frame[STEP_COLUMN].isin(steps)]
        .groupby([RUN_COLUMN, STEP_COLUMN], sort=False)[sensor]
        .mean()
        .rename("process_mean")
        .reset_index()
    )
    grouped["deviation"] = 0.0
    for step, index in grouped.groupby(STEP_COLUMN).groups.items():
        values = grouped.loc[index, "process_mean"]
        scale = float(values.std(ddof=0))
        if np.isfinite(scale) and scale > 1e-12:
            grouped.loc[index, "deviation"] = (values - float(values.median())) / scale
    grouped["absolute_deviation"] = grouped["deviation"].abs()
    return grouped


def aligned_sensor_trace(
    current_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    sensor: str,
    points: int = 200,
) -> pd.DataFrame:
    """Align two original traces by relative Step progress for display only."""
    if sensor not in current_rows.columns or sensor not in reference_rows.columns:
        raise KeyError(sensor)
    if points < 2:
        raise ValueError("points must be at least 2")
    current = pd.to_numeric(current_rows[sensor], errors="coerce").to_numpy(float)
    reference = pd.to_numeric(reference_rows[sensor], errors="coerce").to_numpy(float)
    if min(len(current), len(reference)) < 2 or not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Sensor traces require at least two finite values.")
    progress = np.linspace(0.0, 1.0, points)
    return pd.DataFrame(
        {
            "Selected process": np.interp(progress, np.linspace(0.0, 1.0, len(current)), current),
            "Reference process": np.interp(progress, np.linspace(0.0, 1.0, len(reference)), reference),
        },
        index=pd.Index(progress * 100.0, name="Step progress (%)"),
    )


def rank_sensors_by_activity(
    current_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    sensors: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Put sensors with visible variation first while retaining inactive sensors."""
    activity: dict[str, float] = {}
    for sensor in sensors:
        current = pd.to_numeric(current_rows[sensor], errors="coerce").to_numpy(float)
        reference = pd.to_numeric(reference_rows[sensor], errors="coerce").to_numpy(float)
        values = np.concatenate((current, reference))
        values = values[np.isfinite(values)]
        activity[sensor] = 0.0 if not values.size else float(np.ptp(values) + np.std(values))
    ranked = tuple(sorted(sensors, key=lambda sensor: (-activity[sensor], sensors.index(sensor))))
    return ranked, activity


def notebook_runtime_sources(path: str | Path) -> dict[str, str]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {
        cell.get("id"): "".join(cell.get("source", ()))
        for cell in document.get("cells", ())
        if cell.get("cell_type") == "code"
    }
    missing = [cell_id for cell_id in RUNTIME_CELL_IDS if cell_id not in by_id]
    if missing:
        raise ValueError(f"GEVIS runtime cells are missing: {', '.join(missing)}")
    return {cell_id: by_id[cell_id] for cell_id in RUNTIME_CELL_IDS}


@dataclass(frozen=True)
class GevisRuntime:
    module: ModuleType

    @property
    def discovery_enabled(self) -> bool:
        if getattr(self.module,'DECISION_SYSTEM','hybrid')=='fusion_answers':
            return hasattr(self.module,'review_changes')
        return hasattr(self.module,'review_changes') and getattr(self.module,'DECISION_POLICY_MODE',None)=='reference_quantiles_v2'

    @property
    def checkpoint(self) -> dict[str, Any]:
        return self.module.GEVIS_CHECKPOINT

    @property
    def encoders(self) -> dict[str, Any]:
        return self.module.STEP_ENCODERS

    def questions(self) -> list[Any]:
        if self.discovery_enabled:
            from pecvd.experiments.questions import discovery_questions
            from types import SimpleNamespace
            from .experiments.questions import SENSOR_QUESTION
            questions = discovery_questions(self.module.GevisQuestion)
            if getattr(self.module,'DECISION_SYSTEM','hybrid')=='fusion_answers':
                from .experiments.fusion_answers import QUESTIONS, ANSWER_CRITERIA
                return [self.module.GevisQuestion(QUESTIONS[k], k, ANSWER_CRITERIA[k])
                        for k in ('BOOLEAN','CHOICE','ORDINAL')]
            # A UI task descriptor, not a fourth backbone answer head. Its
            # dynamic vocabulary is bound to the selected input at inference.
            questions.append(SimpleNamespace(text=SENSOR_QUESTION, answer_type='CHOICE_SENSOR',
                typed_family='CHOICE', criteria=()))
            return questions
        if "question_bank" in self.checkpoint and "criteria" in self.checkpoint:
            bank = self.checkpoint["question_bank"]["test"]
            criteria = self.checkpoint["criteria"]
            question_type = self.module.GevisQuestion
            return [
                question_type(bank[answer_type][0], answer_type, tuple(criteria[answer_type]))
                for answer_type in ("BOOLEAN", "CHOICE", "ORDINAL")
            ]
        step = next(iter(self.encoders))
        sensors = self.encoders[step][1].get("sensor_columns", ())
        if not sensors or not hasattr(self.module, "make_grounded_questions"):
            raise RuntimeError("Runtime has no supported question contract")
        return list(self.module.make_grounded_questions(step, sensors[0]))

    def grounded_questions(self, current, reference, step):
        """Return only sensor-free questions; the model discovers the sensor."""
        return self.questions()

    @staticmethod
    def _align_reference_coordinates(
        current: pd.DataFrame,
        reference: pd.DataFrame,
        window_length: int,
        order_column: str = ORDER_COLUMN,
    ) -> pd.DataFrame:
        """Assign the current window coordinates to an aggregated reference."""
        current_window = current.sort_values(order_column).tail(window_length)
        aligned = reference.copy()
        if len(current_window) != len(aligned):
            raise ValueError("Current and reference windows must have equal length")
        for column in ("row_in_step", "step_row_count"):
            if column in current_window and column in aligned:
                aligned.loc[:, column] = current_window[column].to_numpy()
        return aligned

    @english_output
    def ask(self, current: pd.DataFrame, reference: pd.DataFrame, step: str) -> list[dict[str, Any]]:
        info = self.module.ReferenceInfo("Observed process with the same comparison context", 1)
        if self.discovery_enabled:
            return self.module.review_changes(current,reference,step,info)['answers']
        return self.module.ask_gevis(current, reference, step, self.grounded_questions(current, reference, step), info)

    @english_output
    def ask_selected_question(
        self,
        current: pd.DataFrame,
        references: tuple[pd.DataFrame, ...],
        step: str,
        question_id: str,
        *, center_sensor: str | None = None,
    ) -> dict[str, Any]:
        """Answer one trained review question against an aggregated reference pool."""
        if center_sensor is not None and not self.discovery_enabled:
            raise ValueError('Center analysis requires the current discovery runtime')
        if question_id not in (('BOOLEAN','CHOICE','ORDINAL','CHOICE_SENSOR') if self.discovery_enabled else ('BOOLEAN','CHOICE','ORDINAL')):
            raise ValueError(f"Unsupported question: {question_id}")
        if not references:
            raise ValueError("At least one reference Process is required")
        meta = self.encoders[step][1]
        reference = self.module.aggregate_reference_windows(list(references), meta)
        reference = self._align_reference_coordinates(
            current, reference, int(meta["window_length"]),getattr(self.module,'ORDER_COLUMN',ORDER_COLUMN)
        )
        info = self.module.ReferenceInfo("position-matched median reference", len(references))
        if self.discovery_enabled:
            if center_sensor is not None:
                from .experiments.discovery import review_changes
                result=review_changes(self.module,current,reference,step,info,center_sensor=center_sensor)
            else:
                result=self.module.review_changes(current,reference,step,info)
            return next(answer for answer in result['answers'] if answer['answer_type']==question_id)
        questions = {question.answer_type: question for question in self.grounded_questions(current, reference, step)}
        return self.module.ask_gevis(
            current, reference, step, [questions[question_id]], info
        )[0]

    @english_output
    def ask_all_questions(
        self,
        current: pd.DataFrame,
        references: tuple[pd.DataFrame, ...],
        step: str,
        *, center_sensor: str | None = None,
    ) -> list[dict[str, Any]]:
        """Answer all trained question types against one aggregated reference pool."""
        if center_sensor is not None and not self.discovery_enabled:
            raise ValueError('Center analysis requires the current discovery runtime')
        if not references:
            raise ValueError("At least one reference Process is required")
        meta = self.encoders[step][1]
        reference = self.module.aggregate_reference_windows(list(references), meta)
        reference = self._align_reference_coordinates(
            current, reference, int(meta["window_length"]),getattr(self.module,'ORDER_COLUMN',ORDER_COLUMN)
        )
        info = self.module.ReferenceInfo(
            "position-matched median reference",
            len(references),
        )
        if self.discovery_enabled:
            if center_sensor is not None:
                from .experiments.discovery import review_changes
                return review_changes(self.module,current,reference,step,info,center_sensor=center_sensor)['answers']
            return self.module.review_changes(current,reference,step,info)['answers']
        return self.module.ask_gevis(
            current,
            reference,
            step,
            self.grounded_questions(current, reference, step),
            info,
        )

    def scan_change_probabilities(
        self,
        current: pd.DataFrame,
        references: tuple[pd.DataFrame, ...],
        step: str,
        reference_pool_count: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Scan one complete Step and return BOOLEAN change probability per window."""
        if not references:
            raise ValueError("At least one reference Process is required")
        meta = self.encoders[step][1]
        length = int(meta["window_length"])
        stride = int(self.module.STEPS[step][2])
        if len(current) < length or any(len(reference) < length for reference in references):
            raise ValueError("A Step is shorter than the model window")
        question = self.questions()[0]
        info = self.module.ReferenceInfo(
            f"Representative processes with matching comparison context: {len(references)}",
            reference_pool_count,
        )
        starts = list(range(0, len(current) - length + 1, stride))
        last = len(current) - length
        if starts[-1] != last:
            starts.append(last)
        total_comparisons = len(starts) * len(references)
        completed_comparisons = 0
        if progress_callback is not None:
            progress_callback(0, total_comparisons)
        records = []
        for start in starts:
            progress = window_position_ratio(current, length, start)
            current_window = current.iloc[start:start + length]
            probabilities = []
            evidence = []
            for reference in references:
                reference_window = matching_position_window(reference, length, progress)
                try:
                    if self.discovery_enabled:
                        answer=self.module.review_changes(current_window,reference_window,step,info)['answers'][0]
                    else:
                        question = self.grounded_questions(current_window, reference_window, step)[0]
                        answer = self.module.ask_gevis(
                            current_window,
                            reference_window,
                            step,
                            [question],
                            info,
                        )[0]
                except ValueError as error:
                    if any(message in str(error) for message in ("Incomparable windows","Incomparable reference position")):
                        completed_comparisons += 1
                        if progress_callback is not None:
                            progress_callback(completed_comparisons, total_comparisons)
                        continue
                    raise
                if self.checkpoint.get('schema') == 'gevis.question_model.v15' and answer.get('answer') is None:
                    completed_comparisons += 1
                    if progress_callback is not None:
                        progress_callback(completed_comparisons, total_comparisons)
                    continue
                probabilities.append(float(answer["probabilities"][question.criteria[1]]))
                evidence.append((answer.get("evidence") or {}).get("sensor"))
                completed_comparisons += 1
                if progress_callback is not None:
                    progress_callback(completed_comparisons, total_comparisons)
            if not probabilities:
                continue
            coordinate = (
                pd.to_numeric(current_window["row_in_step"], errors="coerce").to_numpy()
                if "row_in_step" in current_window else np.arange(start, start + length)
            )
            records.append({
                "progress": progress * 100.0,
                "start_row": int(coordinate[0]),
                "end_row": int(coordinate[-1]),
                "change_probability": float(np.mean(probabilities)),
                "probability_min": float(np.min(probabilities)),
                "probability_max": float(np.max(probabilities)),
                "evidence_sensor": evidence[int(np.argmax(probabilities))],
                "window_start": start,
                "window_length": length,
                "stride": stride,
                "references_used": len(probabilities),
            })
        if not records:
            raise ValueError("No position-matched comparison windows are available")
        return pd.DataFrame.from_records(records)

    @english_output
    def aggregate_typed_probabilities(
        self,
        current_window: pd.DataFrame,
        references: tuple[pd.DataFrame, ...],
        step: str,
        reference_pool_count: int,
    ) -> tuple[pd.DataFrame, int]:
        """Average GEVIS answer-choice probabilities across comparable references."""
        length = int(self.encoders[step][1]["window_length"])
        target_ratio = window_position_ratio(current_window, length, 0)
        if self.discovery_enabled:
            if not references: raise ValueError('At least one reference Process is required')
            meta=self.encoders[step][1]
            aligned=[matching_position_window(reference,length,target_ratio) for reference in references]
            reference=self.module.aggregate_reference_windows(aligned,meta)
            reference=self._align_reference_coordinates(current_window,reference,length,
                getattr(self.module,'ORDER_COLUMN',ORDER_COLUMN))
            result=self.module.review_changes(current_window,reference,step,
                self.module.ReferenceInfo('position-matched median reference',len(references)))
            if not result['accepted']: raise ValueError('Reference criteria incomplete; automatic review withheld')
            rows=[{'answer_type':answer['answer_type'],'option':option,'probability':float(score),
                   'selected':option==answer['answer'],'score_meaning':answer['score_meaning']}
                  for answer in result['answers'] for option,score in answer['probabilities'].items()]
            return pd.DataFrame(rows),len(references)
        questions = self.questions()
        info = self.module.ReferenceInfo(
            f"Representative processes with matching comparison context: {len(references)}",
            reference_pool_count,
        )
        collected: dict[tuple[str, str], list[float]] = {}
        used = 0
        for reference in references:
            reference_window = matching_position_window(reference, length, target_ratio)
            try:
                if self.discovery_enabled:
                    answers=self.module.review_changes(current_window,reference_window,step,info)['answers']
                else:
                    questions = self.grounded_questions(current_window, reference_window, step)
                    answers = self.module.ask_gevis(current_window, reference_window, step, questions, info)
            except ValueError as error:
                if "Incomparable windows" in str(error):
                    continue
                raise
            if self.checkpoint.get('schema') == 'gevis.question_model.v15' and any(answer.get('answer') is None for answer in answers):
                continue
            used += 1
            for answer in answers:
                for option, probability in answer["probabilities"].items():
                    collected.setdefault((answer["answer_type"], option), []).append(float(probability))
        if not collected:
            raise ValueError("No position-matched reference is available for typed probabilities")
        records = [
            {"answer_type": answer_type, "option": option, "probability": float(np.mean(values))}
            for (answer_type, option), values in collected.items()
        ]
        result = pd.DataFrame.from_records(records)
        result["selected"] = result.groupby("answer_type")["probability"].transform("max").eq(result["probability"])
        return result, used


def load_gevis_runtime(
    root: str | Path,
    notebook_path: str | Path,
    data_path: str | Path,
) -> GevisRuntime:
    """Load the exact checked notebook inference cells and bound artifacts."""
    project_root = Path(root).resolve()
    # App/CLI loaders execute only inference cells, not notebook bootstrap.
    # Honor the same selected training namespace before resolving artifacts.
    if (project_root / 'config/experiments/gevis_pipeline.json').is_file():
        from .experiments.pipeline_settings import configure_pipeline
        configure_pipeline(project_root)
    sources = notebook_runtime_sources(notebook_path)
    module_name = "_pecvd_gevis_notebook_runtime"
    module = ModuleType(module_name)
    module.__file__ = str(Path(notebook_path).resolve())
    sys.modules[module_name] = module
    with redirect_stdout(StringIO()):
        exec(compile(sources["03-types"], f"{module.__file__}:03-types", "exec"), module.__dict__)
        module.ROOT = project_root
        module.DATA_PATH = Path(data_path).resolve()
        module.ENCODER_ROOT = project_root / "artifacts" / module.ENCODER_ARTIFACT_VERSION
        exec(compile(sources["03-load"], f"{module.__file__}:03-load", "exec"), module.__dict__)
        exec(compile(sources["03-function"], f"{module.__file__}:03-function", "exec"), module.__dict__)
    return GevisRuntime(module)


def configured_csv_path(root: str | Path) -> Path:
    configured = os.environ.get("PECVD_FDC_CSV")
    if not configured:
        raise RuntimeError(
            "PECVD_FDC_CSV must point to an explicitly approved local analysis CSV. "
            "The application does not select a project CSV automatically."
        )
    path = Path(configured).expanduser()
    return path if path.is_absolute() else Path(root) / path
