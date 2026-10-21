"""The run sink: each run writes only to its own directory, which is the source of truth for analysis.

<run dir>/
├── tracking/run.json       params, tags, summary metrics, status, timing
├── tracking/metrics.jsonl  append-only series; a requeued run keeps appending
├── tracking/artifacts/     result tables (tables/<name>.parquet), configs, plots
└── DONE | FAILED           end markers; FAILED holds the traceback
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

STATUSES = ("DONE", "FAILED", "STARTED", "MISSING")


class Preempted(Exception):
    """Raised by a run that checkpointed and stopped early; it stays STARTED and is resumed later."""


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)  # atomic: readers never see a half-written file


class Run:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.tracking = run_dir / "tracking"
        self.artifacts = self.tracking / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._json = self.tracking / "run.json"
        self._metrics = self.tracking / "metrics.jsonl"

    # run.json
    def read(self) -> dict[str, Any]:
        return json.loads(self._json.read_text()) if self._json.exists() else {}

    def update(self, **fields: Any) -> None:
        data = self.read()
        for key, value in fields.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
        _write_json(self._json, data)

    def set_tags(self, **tags: Any) -> None:
        self.update(tags={k: str(v) for k, v in tags.items()})

    def log_summary(self, **values: float) -> None:
        """Scalars for result tables, e.g. log_summary(**{"final/val_loss": 0.3})."""
        self.update(summary={k: float(v) for k, v in values.items()})

    # series and files
    def log_metric(self, key: str, value: float, step: int) -> None:
        self.log_metrics({key: value}, step)

    def log_metrics(self, values: dict[str, float], step: int) -> None:
        ts = time.time()
        with self._metrics.open("a") as f:
            for key, value in values.items():
                f.write(json.dumps({"key": key, "value": float(value), "step": int(step), "ts": ts}) + "\n")

    def log_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.artifacts / "tables" / f"{name}.parquet"
        path.parent.mkdir(exist_ok=True)
        df.to_parquet(path)
        return path

    def log_artifact(self, src: Path, name: str | None = None) -> Path:
        dst = self.artifacts / (name or src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst


def member_status(run_dir: Path) -> str:
    if (run_dir / "DONE").exists():
        return "DONE"
    if (run_dir / "FAILED").exists():
        return "FAILED"
    if (run_dir / "tracking" / "run.json").exists():
        return "STARTED"  # running, killed, or preempted: check before resuming
    return "MISSING"


def read_run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "tracking" / "run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tracking" / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["key", "value", "step", "ts"])
    return pd.read_json(path, lines=True)
