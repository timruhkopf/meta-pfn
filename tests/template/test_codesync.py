import os
import subprocess
from pathlib import Path

import pytest

from meta_pfn.tracking import codesync, sweep
from meta_pfn.tracking.codesync import SyncRefused, active_sweeps, deps_fingerprint, sync_code
from meta_pfn.tracking.provenance import git, git_state
from meta_pfn.tracking.workflow import report
from meta_pfn.utils.env import repo_root


def _commit(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", name], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", name], check=True)
    return git(repo, "rev-parse", "HEAD")


def _active_sweep(runs_root: Path) -> Path:
    d = runs_root / "exp" / "pilot_1"
    d.mkdir(parents=True)
    (d / "sweep.json").write_text('{"id": "pilot_1", "experiment": "exp", "stage": "pilot", "backend": "detached"}')
    (d / "jobs.jsonl").write_text('{"member": 0, "overrides": []}\n')
    (d / "queue.pid").write_text(str(os.getpid()))  # a live process
    return d


def test_active_sweeps(runs_root: Path) -> None:
    assert active_sweeps(runs_root) == []
    d = _active_sweep(runs_root)
    assert [p for p, _ in active_sweeps(runs_root)] == [d]
    (d / "queue.pid").write_text("999999999")  # dead
    assert active_sweeps(runs_root) == []


def test_sync_code_guards_dependency_changes(git_repo: Path, runs_root: Path) -> None:
    start = git(git_repo, "rev-parse", "HEAD")
    code_only = _commit(git_repo, "code.py", "x = 2\n")
    deps = _commit(git_repo, "uv.lock", "new lock\n")
    git(git_repo, "checkout", "-q", "--detach", start)
    _active_sweep(runs_root)

    sync_code(code_only, repo=git_repo)  # code changes are fine: sweeps run from frozen worktrees
    assert git(git_repo, "rev-parse", "HEAD") == code_only
    assert deps_fingerprint(deps, git_repo) != deps_fingerprint(None, git_repo)
    with pytest.raises(SyncRefused):
        sync_code(deps, repo=git_repo)
    assert git(git_repo, "rev-parse", "HEAD") == code_only


def test_forced_sync_skips_venv_under_active_sweeps(git_repo: Path, runs_root: Path, capsys) -> None:
    deps = _commit(git_repo, "uv.lock", "new lock\n")
    git(git_repo, "checkout", "-q", "--detach", "HEAD~1")
    _active_sweep(runs_root)
    sync_code(deps, force=True, repo=git_repo)
    assert "NOT syncing .venv" in capsys.readouterr().out


def test_status_suggests_snapshot(runs_root: Path, root: Path) -> None:
    d = sweep.plan(
        "pilot", ["experiment=debug"], command="t", backend="local", log_dir=root / "research_log", allow_dirty=True
    )
    sweep.execute(d)
    out = report()
    assert "## workflow status" in out
    assert f"just snapshot {d}" in out


def test_members_run_from_frozen_code(runs_root: Path, root: Path) -> None:
    """A commit made while a sweep is queued neither breaks nor changes its members."""
    if git_state().dirty:
        pytest.skip("needs a clean, committed checkout (as in CI)")
    d = sweep.plan(
        "sanity", ["experiment=debug", "seed=0,1"], command="t", backend="local", log_dir=root / "research_log"
    )
    try:
        assert (d / ".code" / ".git").exists()
        sweep.execute(d, members=[0])
        marker = repo_root() / "src" / "_moved_on.py"  # untracked: the main tree is now dirty
        marker.write_text("")
        try:
            sweep.execute(d)
        finally:
            marker.unlink()
        assert sweep.counts(d) == {"DONE": 2}
        assert not (d / ".code").exists()  # released once all members are DONE
    finally:
        git(repo_root(), "tag", "-d", f"sweep/debug/{d.name}")
        git(repo_root(), "worktree", "prune")


def test_notify_after_merge_never_fails(capsys) -> None:
    codesync.notify_after_merge()
