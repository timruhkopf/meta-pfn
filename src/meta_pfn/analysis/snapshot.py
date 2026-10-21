"""Snapshots: small, committed copies of a finished sweep. Notebooks and figures read only these.

results/<experiment>/<sweep_id>/
├── runs.parquet       load_runs() output, including members that are not DONE
├── constants.yaml     params constant across the sweep
├── columns.json       which runs.parquet columns are params / metrics
├── tables/<name>.parquet   result tables of all members, with a `member` column
├── curves.parquet     metric series, downsampled
├── experiment.yaml    the experiment config as committed at the sweep's commit
├── sweep/             sweep.json + jobs.jsonl: the plan, for `just reproduce <snapshot>`
└── manifest.json      command, commit, backend, host, times, status counts, device hours
"""

import datetime as dt
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..tracking.provenance import git
from ..tracking.sink import read_metrics, read_run
from ..tracking.sweep import counts, jobs, read_sweep
from ..utils.env import PACKAGE, repo_root
from .runs import load_runs


def _downsample(series: pd.DataFrame, points: int) -> pd.DataFrame:
    if len(series) <= points:
        return series
    keep = sorted(set(round(i * (len(series) - 1) / (points - 1)) for i in range(points)))
    return series.iloc[keep]


def _experiment_yaml(meta: dict[str, Any]) -> str:
    rel = f"src/{PACKAGE}/configs/experiment/{meta['experiment']}.yaml"
    try:
        return git(repo_root(), "show", f"{meta['commit']}:{rel}") + "\n"
    except (subprocess.CalledProcessError, FileNotFoundError):
        path = repo_root() / rel
        note = "# NOTE: not found at the sweep's commit; this is the current working-tree version\n"
        return note + path.read_text() if path.exists() else "# experiment config not found\n"


def snapshot(sweep_dir: Path, out_root: Path | None = None, curve_points: int = 500) -> Path:
    meta = read_sweep(sweep_dir)
    out = (out_root or repo_root() / "results") / meta["experiment"] / meta["id"]
    if out.exists():
        shutil.rmtree(out)  # re-running snapshot is how a snapshot is refreshed
    (out / "tables").mkdir(parents=True)

    df = load_runs(sweep_dir)
    constant = df.attrs["constant"]
    df.to_parquet(out / "runs.parquet")
    (out / "constants.yaml").write_text(yaml.safe_dump(constant, sort_keys=True))
    (out / "columns.json").write_text(json.dumps({"params": df.attrs["params"], "metrics": df.attrs["metrics"]}))
    (out / "experiment.yaml").write_text(_experiment_yaml(meta))
    (out / "sweep").mkdir()  # the plan, so `just reproduce <snapshot>` works from a fresh clone
    for name in ("sweep.json", "jobs.jsonl"):
        shutil.copy2(sweep_dir / name, out / "sweep" / name)

    tables: dict[str, list[pd.DataFrame]] = {}
    curves, starts, ends, device_h, wall_s = [], [], [], 0.0, 0.0
    for member in jobs(sweep_dir):
        mdir = sweep_dir / str(member)
        for path in sorted((mdir / "tracking" / "artifacts" / "tables").glob("*.parquet")):
            tables.setdefault(path.stem, []).append(pd.read_parquet(path).assign(member=member))
        metrics = read_metrics(mdir)
        for _, series in metrics.groupby("key"):
            curves.append(_downsample(series.sort_values("step"), curve_points).assign(member=member))
        run = read_run(mdir)
        starts += [run["start"]] if "start" in run else []
        ends += [run["end"]] if "end" in run else []
        device_h += run.get("device_h", 0.0)
        wall_s += run.get("wall_s", 0.0)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(out / "tables" / f"{name}.parquet")
    if curves:
        pd.concat(curves, ignore_index=True)[["member", "key", "step", "value"]].to_parquet(out / "curves.parquet")

    iso = lambda t: dt.datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
    manifest = {
        **{
            k: meta[k]
            for k in (
                "id",
                "experiment",
                "stage",
                "kind",
                "command",
                "overrides",
                "backend",
                "commit",
                "branch",
                "tag",
                "allow_dirty",
                "host",
                "created",
            )
        },
        "editable_dependencies": meta.get("editable_dependencies", {}),
        "status_counts": counts(sweep_dir),
        "first_start": iso(min(starts)) if starts else None,
        "last_end": iso(max(ends)) if ends else None,
        "wall_h": round(wall_s / 3600, 3),
        "device_h": round(device_h, 3),
        "snapshot_created": dt.datetime.now().isoformat(timespec="seconds"),
        "sweep_dir": str(sweep_dir),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


@dataclass
class Snapshot:
    path: Path
    runs: pd.DataFrame
    manifest: dict[str, Any]
    constants: dict[str, Any]
    experiment_yaml: str
    _tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            self._tables[name] = pd.read_parquet(self.path / "tables" / f"{name}.parquet")
        return self._tables[name]

    def curves(self) -> pd.DataFrame:
        path = self.path / "curves.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["member", "key", "step", "value"])

    @property
    def preliminary(self) -> bool:
        counts = self.manifest["status_counts"]
        return counts.get("DONE", 0) < sum(counts.values())


def load_snapshot(path: str | Path, require_stage: str | None = None) -> Snapshot:
    """Paths are relative to the repo root, so notebooks work from any directory."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    manifest = json.loads((p / "manifest.json").read_text())
    if require_stage and manifest["stage"] != require_stage:
        raise ValueError(f"{p} is stage {manifest['stage']!r}, {require_stage!r} required")
    runs = pd.read_parquet(p / "runs.parquet")
    constants = yaml.safe_load((p / "constants.yaml").read_text()) or {}
    runs.attrs.update(constant=constants, **json.loads((p / "columns.json").read_text()))
    return Snapshot(p, runs, manifest, constants, (p / "experiment.yaml").read_text())
