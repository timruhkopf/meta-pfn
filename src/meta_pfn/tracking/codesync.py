"""Updating code under running sweeps (pull on a cluster, `just push` to a GPU box).

Code changes are harmless: sweeps run from frozen worktrees (`<sweep dir>/.code`). Dependency changes
are not: every run shares `.venv`, and syncing it under a running sweep can change or crash it. So an
update is refused while sweeps are active *and* uv.lock / pyproject.toml change, unless forced.
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from ..utils.env import repo_root, runs_root, var
from .provenance import git
from .sweep import read_sweep, status

DEPENDENCY_FILES = ("uv.lock", "pyproject.toml")
RECENT_S = 3600  # a STARTED member whose sink changed within this window counts as running


class SyncRefused(RuntimeError):
    pass


def deps_fingerprint(ref: str | None = None, repo: Path | None = None) -> str:
    """Hash of the dependency files in the working tree, or at a git ref."""
    repo = repo or repo_root()
    h = hashlib.sha256()
    for name in DEPENDENCY_FILES:
        if ref is None:
            path = repo / name
            h.update(path.read_bytes() if path.exists() else b"")
        else:
            try:
                h.update(subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{name}"]))
            except subprocess.CalledProcessError:
                h.update(b"")
    return h.hexdigest()


def _pid_alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def _slurm_alive(jobs_file: Path) -> bool:
    if not jobs_file.exists() or shutil.which("squeue") is None:
        return False
    ids = [line.split()[0] for line in jobs_file.read_text().splitlines() if line.strip()]
    out = subprocess.run(["squeue", "-h", "-o", "%A", "-j", ",".join(ids)], capture_output=True, text=True)
    return bool(out.stdout.strip())


def _recently_written(sweep_dir: Path, st: dict[int, str], now: float) -> bool:
    for member, s in st.items():
        sink = sweep_dir / str(member) / "tracking" / "run.json"
        if s == "STARTED" and sink.exists() and now - sink.stat().st_mtime < RECENT_S:
            return True
    return False


def active_sweeps(root: Path | None = None) -> list[tuple[Path, dict[str, int]]]:
    """Sweeps with unfinished members and a sign of life (queue pid, SLURM job, recent writes)."""
    import time

    now, found = time.time(), []
    for meta_file in sorted((root or runs_root()).glob("*/*/sweep.json")):
        d = meta_file.parent
        st = status(d)
        unfinished = [s for s in st.values() if s not in ("DONE", "FAILED")]
        if not unfinished:
            continue
        if _pid_alive(d / "queue.pid") or _slurm_alive(d / "slurm_jobs.txt") or _recently_written(d, st, now):
            counts: dict[str, int] = {}
            for s in st.values():
                counts[s] = counts.get(s, 0) + 1
            found.append((d, counts))
    return found


def describe_active(active: list[tuple[Path, dict[str, int]]]) -> str:
    if not active:
        return "no active sweeps"
    lines = [f"{len(active)} active sweep(s), running from frozen worktrees:"]
    for d, counts in active:
        meta = read_sweep(d)
        lines.append(f"  {d}  [{meta['stage']}, {meta['backend']}]  {counts}")
    return "\n".join(lines)


def sync_code(ref: str, force: bool = False, repo: Path | None = None) -> None:
    """Move this checkout to `ref` (a commit, branch, or `@{u}` to fast-forward), guarding running sweeps."""
    repo = repo or repo_root()
    target = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    deps_change = deps_fingerprint(target, repo) != deps_fingerprint(None, repo)
    active = active_sweeps()
    print(describe_active(active))
    if active and deps_change and not force:
        raise SyncRefused(
            f"{target[:8]} changes {' / '.join(DEPENDENCY_FILES)}, and syncing the shared .venv would change the "
            "environment under the active sweeps. Wait for them to finish, or pass force (they may break)."
        )
    if ref == "@{u}":
        git(repo, "merge", "--ff-only", "--quiet", "@{u}")
    else:
        git(repo, "checkout", "--quiet", "--detach", target)
    print(f"code at {target[:8]}")
    if not deps_change:
        return
    if active:
        print("dependencies changed: NOT syncing .venv while sweeps are active; run `uv sync` once they finish")
        return
    extra = var("TORCH_EXTRA") or "cpu"
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    subprocess.run([uv, "sync", "--extra", extra], cwd=repo, check=True)


def notify_after_merge(repo: Path | None = None) -> None:
    """post-merge hook: say what a plain `git pull` means for running sweeps. Never blocks."""
    repo = repo or repo_root()
    active = active_sweeps()
    try:
        changed = git(repo, "diff", "--name-only", "ORIG_HEAD", "HEAD").splitlines()
    except subprocess.CalledProcessError:
        changed = []
    deps = [f for f in changed if f in DEPENDENCY_FILES]
    if active:
        print(describe_active(active) + "\n  (code changes don't affect them)")
    if deps and active:
        print(f"WARNING: {', '.join(deps)} changed. Don't `uv sync` until the sweeps above finish.")
    elif deps:
        print(f"{', '.join(deps)} changed: run `just setup <extra>` (or `uv sync --extra <extra>`).")
