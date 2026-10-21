from pathlib import Path

from mlflow.tracking import MlflowClient

from meta_pfn.tracking import logbook, sweep
from meta_pfn.tracking.mlflow_sync import sync


def test_sync_is_idempotent_and_incremental(runs_root: Path, root: Path, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log = root / "research_log"
    d = sweep.plan(
        "pilot", ["experiment=debug", "seed=0,1"], command="cmd", backend="local", log_dir=log, allow_dirty=True
    )
    sweep.execute(d, members=[0])
    sync([runs_root], log, uri)
    client = MlflowClient(uri)
    (exp,) = [e for e in client.search_experiments() if e.name.endswith("/debug")]

    def children() -> list:
        return client.search_runs([exp.experiment_id], "tags.mlflow.parentRunId != ''")

    (child,) = children()
    n_points = len(client.get_metric_history(child.info.run_id, "train/mse"))
    assert n_points > 0 and child.info.status == "FINISHED"

    sync([runs_root], log, uri)  # nothing new: nothing changes
    assert len(client.get_metric_history(child.info.run_id, "train/mse")) == n_points

    sweep.execute(d)  # the second member finishes; a decision is recorded
    logbook.new_entry(log, "debug", "Debug", "nb")
    logbook.decide(log, sweep.read_sweep(d), sweep.counts(d), "GO", "fine")
    sync([d], log, uri)
    assert len(children()) == 2
    (parent,) = client.search_runs([exp.experiment_id], f"attributes.run_name = '{d.name}'")
    assert parent.data.tags["decision"] == "GO"
