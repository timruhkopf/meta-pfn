"""End to end: sweep -> sinks -> load_runs -> snapshot -> notebook header; scaffolded notebook executes."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from meta_pfn.analysis import load_runs, load_snapshot, snapshot, summarize
from meta_pfn.analysis.header import configuration_md, provenance_md
from meta_pfn.tracking import sweep
from meta_pfn.tracking.scaffold import new_experiment


@pytest.fixture
def finished_sweep(runs_root: Path, root: Path) -> Path:
    d = sweep.plan(
        "pilot",
        ["experiment=debug", "seed=0,1", "task.lr=0.05,0.1"],
        command="just sweep local pilot ...",
        backend="local",
        log_dir=root / "research_log",
        allow_dirty=True,
    )
    sweep.execute(d, members=[0, 1, 2])  # member 3 stays MISSING
    return d


def test_load_runs(finished_sweep: Path) -> None:
    df = load_runs(finished_sweep)
    assert list(df["status"]) == ["DONE", "DONE", "DONE", "MISSING"]
    assert df.attrs["params"] == ["seed", "task.lr"]
    assert df.attrs["constant"]["task.n"] == 64
    assert "final/eval_mse" in df.attrs["metrics"]
    s = summarize(df, "final/eval_mse", by=["task.lr"])
    assert list(s["n_done"]) == [2, 1] and list(s["preliminary"]) == [False, True]


def test_snapshot_and_header(finished_sweep: Path, root: Path) -> None:
    out = snapshot(finished_sweep, root / "results")
    snap = load_snapshot(out)
    assert snap.preliminary
    assert len(snap.table("coefficients")) == 3 * 4  # 3 DONE members x dim 4
    assert set(snap.curves()["key"]) == {"train/mse"}
    prov = provenance_md(snap)
    assert "just sweep local pilot ..." in prov and "Preliminary" in prov and "Not reportable" in prov
    conf = configuration_md(snap)
    assert "task.lr" in conf and "override /task: toy_regression" in conf
    with pytest.raises(ValueError):
        load_snapshot(out, require_stage="final")


def test_new_experiment(root: Path) -> None:
    paths = new_experiment(root, "warmup_variance", "Does warmup reduce seed variance?")
    assert [p.suffix for p in paths] == [".yaml", ".ipynb", ".md"]
    nb = json.loads(paths[1].read_text())
    assert nb["cells"][0]["source"][0].startswith("# Does warmup")
    assert paths[1].name.startswith("01_")
    with pytest.raises(FileExistsError):
        new_experiment(root, "warmup_variance", "again")


@pytest.mark.slow
def test_scaffolded_notebook_executes(finished_sweep: Path, root: Path, tmp_path: Path) -> None:
    pytest.importorskip("nbconvert")
    out = snapshot(finished_sweep, root / "results")
    nb_path = new_experiment(root, "debug_nb", "Debug notebook")[1]
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        src = (
            "".join(cell["source"])
            .replace("results/debug_nb/<sweep_id>", str(out))
            .replace("<metric>", "final/eval_mse")
        )
        cell["source"] = src
    nb_path.write_text(json.dumps(nb))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output-dir",
            str(tmp_path),
            str(nb_path),
        ],
        check=True,
        capture_output=True,
    )
    executed = (tmp_path / nb_path.name).read_text()
    assert "How it was run" in executed and "n_done" in executed


def test_cli_decide_and_status(finished_sweep: Path, root: Path) -> None:
    cli = [sys.executable, "-m", "meta_pfn.tracking.cli", "--root", str(root)]
    subprocess.run([*cli, "new-experiment", "debug", "Debug"], check=False, capture_output=True)
    # `debug` already has a config in the real package; new-experiment writes into root, so this works:
    status = subprocess.run(
        [*cli, "sweep-status", str(finished_sweep)], capture_output=True, text=True, check=True
    ).stdout
    assert "MISSING" in status
    subprocess.run(
        [*cli, "decide", str(finished_sweep), "INCONCLUSIVE", "only 3/4 done"], check=True, capture_output=True
    )
    entry = next((root / "research_log" / "entries").glob("*debug.md")).read_text()
    assert "INCONCLUSIVE" in entry and "3/4 done, 1 missing" in entry
    shutil.rmtree(root / "results", ignore_errors=True)
