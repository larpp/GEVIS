"""Validated process-specific input configuration for the shared GEVIS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


PROCESS_CONFIG_SCHEMA = "gevis.process_config.v1"


@dataclass(frozen=True)
class StepConfig:
    name: str
    window_length: int
    patch_length: int
    stride: int
    description: str = ''

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Step name cannot be empty")
        if min(self.window_length, self.patch_length, self.stride) <= 0:
            raise ValueError(f"Step {self.name!r} lengths must be positive")
        if self.window_length < self.patch_length:
            raise ValueError(f"Step {self.name!r} window is shorter than its patch")
        if self.window_length % self.patch_length:
            raise ValueError(f"Step {self.name!r} window must be divisible by patch_length")


@dataclass(frozen=True)
class ColumnConfig:
    run: str
    step: str
    order: str
    step_one_hot_prefix: str = "step_name_"
    step_one_hot_excluded: tuple[str, ...] = ("step_name_nan",)


@dataclass(frozen=True)
class ArtifactConfig:
    encoder: str
    decision: str
    legacy_decision: str


@dataclass(frozen=True)
class ProcessConfig:
    schema: str
    process_family: str
    data_path: str
    artifacts: ArtifactConfig
    columns: ColumnConfig
    steps: tuple[StepConfig, ...]
    source_path: Path
    sha256: str
    sensor_columns: tuple[str, ...] = ()
    source_columns: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema != PROCESS_CONFIG_SCHEMA:
            raise ValueError(f"Unsupported process config schema: {self.schema!r}")
        if not self.process_family.strip():
            raise ValueError("process_family cannot be empty")
        if Path(self.data_path).is_absolute():
            raise ValueError("data_path must be relative to the project root")
        artifact_names = (
            self.artifacts.encoder,
            self.artifacts.decision,
            self.artifacts.legacy_decision,
        )
        if any(not name.strip() or Path(name).name != name for name in artifact_names):
            raise ValueError("Artifact names must be non-empty directory names")
        column_values = (self.columns.run, self.columns.step, self.columns.order)
        if any(not value.strip() for value in column_values):
            raise ValueError("run, step, and order columns cannot be empty")
        if len(set(column_values)) != len(column_values):
            raise ValueError("run, step, and order columns must be distinct")
        if len(set(self.sensor_columns))!=len(self.sensor_columns) or any(not c.strip() or c in column_values for c in self.sensor_columns):
            raise ValueError('Sensor columns must be distinct physical measurements, not run/step/order metadata')
        if self.source_columns and (not {'run','step','order'}<=self.source_columns.keys() or any(not isinstance(v,str) or not v.strip() for v in self.source_columns.values())):
            raise ValueError('Raw input mapping requires run/step/order source column names')
        if not self.steps:
            raise ValueError("At least one process Step is required")
        for step in self.steps:
            step.validate()
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Process Step names must be unique")

    @property
    def step_specs(self) -> dict[str, tuple[int, int, int]]:
        return {
            step.name: (step.window_length, step.patch_length, step.stride)
            for step in self.steps
        }


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {context}.{key}")
    return mapping[key]


def load_process_config(path: str | Path) -> ProcessConfig:
    source_path = Path(path).resolve()
    raw_bytes = source_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Process config root must be an object")
    columns_payload = _required(payload, "columns", "config")
    artifacts_payload = _required(payload, "artifacts", "config")
    steps_payload = _required(payload, "steps", "config")
    if not isinstance(columns_payload, dict) or not isinstance(artifacts_payload, dict) or not isinstance(steps_payload, list):
        raise ValueError("columns/artifacts must be objects and steps must be a list")
    columns = ColumnConfig(
        run=str(_required(columns_payload, "run", "columns")),
        step=str(_required(columns_payload, "step", "columns")),
        order=str(_required(columns_payload, "order", "columns")),
        step_one_hot_prefix=str(columns_payload.get("step_one_hot_prefix", "step_name_")),
        step_one_hot_excluded=tuple(map(str, columns_payload.get("step_one_hot_excluded", []))),
    )
    steps = tuple(
        StepConfig(
            name=str(_required(item, "name", f"steps[{index}]")),
            window_length=int(_required(item, "window_length", f"steps[{index}]")),
            patch_length=int(_required(item, "patch_length", f"steps[{index}]")),
            stride=int(_required(item, "stride", f"steps[{index}]")),
            description=str(item.get('description','')),
        )
        for index, item in enumerate(steps_payload)
    )
    artifacts = ArtifactConfig(
        encoder=str(_required(artifacts_payload, "encoder", "artifacts")),
        decision=str(_required(artifacts_payload, "decision", "artifacts")),
        legacy_decision=str(_required(artifacts_payload, "legacy_decision", "artifacts")),
    )
    config = ProcessConfig(
        schema=str(_required(payload, "schema", "config")),
        process_family=str(_required(payload, "process_family", "config")),
        data_path=str(_required(payload, "data_path", "config")),
        artifacts=artifacts,
        columns=columns,
        steps=steps,
        source_path=source_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        sensor_columns=tuple(map(str,payload.get('sensor_columns',[]))),
        source_columns=dict(payload.get('source_columns',{})),
    )
    config.validate()
    return config
