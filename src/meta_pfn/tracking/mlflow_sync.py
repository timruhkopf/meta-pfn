"""One-way, idempotent push of run sinks into MLflow, which is only a viewer: analysis never reads it.

MLflow experiment = experiment, parent run = sweep, child run = member. Sync state lives in
<sweep dir>/.mlflow_sync.json, so running the sync twice changes nothing and a growing metrics.jsonl
is pushed incrementally. Only this process writes the store, so a SQLite file on NFS is fine.
"""

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param, RunStatus, RunTag
from mlflow.tracking import MlflowClient

from ..utils.env import PACKAGE, repo_root, var
from .logbook import find_entry
from .sink import member_status, read_run
from .sweep import jobs, read_sweep

TERMINAL = {"DONE": RunStatus.to_string(RunStatus.FINISHED), "FAILED": RunStatus.to_string(RunStatus.FAILED)}


def tracking_uri() -> str:
    return var("MLFLOW_URI") or f"sqlite:///{repo_root() / 'mlflow.db'}"


def _chunks(items: list, n: int = 900) -> list[list]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _sync_member(client: MlflowClient, exp_id: str, parent: str, mdir: Path, state: dict[str, Any]) -> None:
    run = read_run(mdir)
    if "run_id" not in state:
        child = client.create_run(exp_id, run_name=mdir.name, tags={"mlflow.parentRunId": parent})
        state.update(run_id=child.info.run_id, offset=0, uploaded=[], finished=False)
        params = [Param(k, str(v)[:6000]) for k, v in run.get("params", {}).items()]
        for chunk in _chunks(params, 100):
            client.log_batch(state["run_id"], params=chunk)
    rid = state["run_id"]
    client.log_batch(rid, tags=[RunTag(k, str(v)) for k, v in run.get("tags", {}).items()])

    metrics_file = mdir / "tracking" / "metrics.jsonl"
    if metrics_file.exists():
        with metrics_file.open("rb") as f:
            f.seek(state["offset"])
            data = f.read()
        complete = data[: data.rfind(b"\n") + 1]  # never push a half-written last line
        rows = [json.loads(line) for line in complete.splitlines() if line]
        metrics = [Metric(r["key"], r["value"], int(r["ts"] * 1000), r["step"]) for r in rows]
        for chunk in _chunks(metrics):
            client.log_batch(rid, metrics=chunk)
        state["offset"] += len(complete)

    artifacts = mdir / "tracking" / "artifacts"
    for path in sorted(p for p in artifacts.rglob("*") if p.is_file()):
        rel = str(path.relative_to(artifacts))
        if rel not in state["uploaded"]:
            client.log_artifact(rid, str(path), artifact_path=str(Path(rel).parent) if "/" in rel else None)
            state["uploaded"].append(rel)

    status = member_status(mdir)
    if status in TERMINAL and not state["finished"]:
        summary = [Metric(k, v, 0, 0) for k, v in run.get("summary", {}).items()]
        for chunk in _chunks(summary):
            client.log_batch(rid, metrics=chunk)
        client.set_terminated(rid, TERMINAL[status])
        state["finished"] = True


def sync_sweep(client: MlflowClient, sweep_dir: Path, log_dir: Path) -> None:
    meta = read_sweep(sweep_dir)
    state_file = sweep_dir / ".mlflow_sync.json"
    state: dict[str, Any] = json.loads(state_file.read_text()) if state_file.exists() else {"members": {}}
    name = f"{PACKAGE}/{meta['experiment']}"
    exp = client.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp else client.create_experiment(name)
    if "parent" not in state:
        state["parent"] = client.create_run(exp_id, run_name=meta["id"]).info.run_id
    tags = {k: str(meta[k]) for k in ("stage", "commit", "tag", "backend", "host", "allow_dirty")}
    tags["mlflow.note.content"] = f"`{meta['command']}`"
    entry = find_entry(log_dir, meta["experiment"]) if log_dir.exists() else None
    for decided in entry.sweeps if entry else []:
        if decided.get("id") == meta["id"]:
            tags.update(decision=decided["decision"], decision_reason=decided["reason"])
    client.log_batch(state["parent"], tags=[RunTag(k, v) for k, v in tags.items()])
    try:
        for member in jobs(sweep_dir):
            mdir = sweep_dir / str(member)
            if member_status(mdir) != "MISSING":
                _sync_member(client, exp_id, state["parent"], mdir, state["members"].setdefault(str(member), {}))
    finally:  # persist progress even if the server drops mid-sync
        state_file.write_text(json.dumps(state, indent=1))


def sync(paths: list[Path], log_dir: Path, uri: str | None = None) -> int:
    client = MlflowClient(tracking_uri=uri or tracking_uri())
    sweeps = sorted(
        {
            p.parent
            for path in paths
            for p in ([path / "sweep.json"] if (path / "sweep.json").exists() else path.rglob("sweep.json"))
        }
    )
    for d in sweeps:
        sync_sweep(client, d, log_dir)
    print(f"synced {len(sweeps)} sweeps to {uri or tracking_uri()}")
    return len(sweeps)
