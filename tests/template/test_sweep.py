import json
import subprocess
import sys
from pathlib import Path

import pytest

from meta_pfn.tracking import sweep
from meta_pfn.tracking.logbook import GateError


def test_expand_grid() -> None:
    jobs = sweep.expand(["experiment=debug", "seed=0,1", "task.lr=0.1,0.2,0.3"])
    assert len(jobs) == 6
    assert jobs[0] == ["experiment=debug", "seed=0", "task.lr=0.1"]
    assert sweep.expand(["experiment=debug", "seed=range(3)"])[-1] == ["experiment=debug", "seed=2"]


def test_experiment_required() -> None:
    with pytest.raises(ValueError, match="experiment"):
        sweep.experiment_of(["seed=0"])
    with pytest.raises(ValueError):
        sweep.experiment_of(["experiment=a,b"])


def _plan(root: Path, *overrides: str, stage: str = "pilot", single: bool = False) -> Path:
    return sweep.plan(
        stage,
        ["experiment=debug", *overrides],
        command="test",
        backend="local",
        log_dir=root / "research_log",
        allow_dirty=True,
        single=single,
    )


def test_plan_layout(runs_root: Path, root: Path) -> None:
    d = _plan(root, "seed=0,1,2")
    meta = json.loads((d / "sweep.json").read_text())
    assert d.parent == runs_root / "debug"
    assert meta["stage"] == "pilot" and meta["n_members"] == 3 and meta["allow_dirty"]
    assert meta["tag"] == ""  # dirty launches are not tagged
    assert sweep.status(d) == {0: "MISSING", 1: "MISSING", 2: "MISSING"}


def test_run_takes_one_configuration(runs_root: Path, root: Path) -> None:
    with pytest.raises(ValueError, match="sweep"):
        _plan(root, "seed=0,1", single=True)


def test_gate_blocks_confirm(runs_root: Path, root: Path) -> None:
    with pytest.raises(GateError):
        _plan(root, stage="confirm")
    assert not runs_root.exists()


def test_execute_failure_and_resume(runs_root: Path, root: Path) -> None:
    d = _plan(root, "task.steps=10,0")  # steps=0 raises in the task
    counts = sweep.execute(d)
    assert counts == {"DONE": 1, "FAILED": 1}
    assert "task.steps must be >= 1" in (d / "1" / "FAILED").read_text()

    # simulate a member killed mid-run: it has a sink but no end marker
    (d / "0" / "DONE").unlink()
    assert sweep.status(d) == {0: "STARTED", 1: "FAILED"}
    sweep.execute(d, members=[0])
    assert sweep.status(d)[0] == "DONE"
    run = json.loads((d / "0" / "tracking" / "run.json").read_text())
    assert run["restarts"] == 1 and run["tags"]["sweep_id"] == d.name and run["tags"]["reportable"] == "False"


def test_cli_run_local(runs_root: Path, root: Path) -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "meta_pfn.tracking.cli",
            "--root",
            str(root),
            "run",
            "local",
            "sanity",
            "--allow-dirty",
            "experiment=debug",
            "task.steps=5",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (d,) = (runs_root / "debug").iterdir()
    assert sweep.counts(d) == {"DONE": 1}, out
    assert json.loads((d / "sweep.json").read_text())["command"].startswith("just run local sanity")


def test_start_time() -> None:
    import datetime as dt

    now = dt.datetime.now()
    soon = sweep.start_time(f"{(now + dt.timedelta(minutes=5)):%H:%M}")
    assert dt.timedelta(minutes=3) < soon - now < dt.timedelta(minutes=6)  # later today
    earlier = sweep.start_time(f"{(now - dt.timedelta(minutes=5)):%H:%M}")
    assert dt.timedelta(hours=23) < earlier - now < dt.timedelta(hours=24)  # already past: tomorrow
    assert sweep.start_time("2030-01-02T03:04") == dt.datetime(2030, 1, 2, 3, 4)
