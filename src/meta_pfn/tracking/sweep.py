"""Sweeps: plan once into a directory, execute members anywhere, resume any time.

runs/<experiment>/<stage>_<YYYYMMDD-HHMMSS>/
├── sweep.json      exact command, stage, commit, tag, backend, host, created
├── jobs.jsonl      {"member": i, "overrides": [...]} per member
└── <i>/            one member: Hydra run dir with the run sink and DONE | FAILED

No process has to stay alive: all state is in the directory, and members are idempotent.

Members run from `<sweep dir>/.code`, a git worktree at the sweep's commit, so pulling or pushing code
while a sweep is queued neither crashes nor changes it. Only dependency changes (the shared .venv) can;
`sync_code` guards against those.
"""

import datetime as dt
import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser

from ..utils.env import PACKAGE, PREFIX, repo_root, runs_root, var
from .logbook import check_gate
from .provenance import git, git_state, require_clean, require_clean_dependencies, tag
from .sink import member_status


def expand(overrides: Sequence[str]) -> list[list[str]]:
    """Hydra sweep syntax (`seed=0,1,2`, `lr=range(...)`, `model=a,b`) -> one override list per member."""
    axes: list[list[str]] = []
    for o in OverridesParser.create().parse_overrides(list(overrides)):
        if o.is_sweep_override():
            key = o.get_key_element()
            axes.append([f"{key}={v}" for v in o.sweep_string_iterator()])
        else:
            axes.append([str(o.input_line)])
    return [list(combo) for combo in itertools.product(*axes)]


def experiment_of(overrides: Sequence[str]) -> str:
    names = [o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0].lstrip("+~") == "experiment"]
    if len(names) != 1 or "," in names[0]:
        raise ValueError("tracked runs need exactly one `experiment=<name>` (one experiment per sweep)")
    return names[0]


def read_sweep(sweep_dir: Path) -> dict[str, Any]:
    return json.loads((sweep_dir / "sweep.json").read_text())


def jobs(sweep_dir: Path) -> dict[int, list[str]]:
    path = sweep_dir / "jobs.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return {(j := json.loads(line))["member"]: j["overrides"] for line in lines if line}


def add_job(sweep_dir: Path, member: int, overrides: Sequence[str]) -> None:
    with (sweep_dir / "jobs.jsonl").open("a") as f:  # O_APPEND: safe for concurrent optuna workers
        f.write(json.dumps({"member": member, "overrides": list(overrides)}) + "\n")


def plan(
    stage: str,
    overrides: Sequence[str],
    *,
    command: str,
    backend: str,
    log_dir: Path,
    kind: str = "grid",
    allow_dirty: bool = False,
    single: bool = False,
) -> Path:
    """Check provenance and the stage gate, write the sweep dir, tag the commit."""
    experiment = experiment_of(overrides)
    check_gate(log_dir, experiment, stage)
    state = git_state() if allow_dirty else require_clean()
    editable = {} if allow_dirty else require_clean_dependencies(stage)
    members = expand(overrides) if kind == "grid" else []
    if single and len(members) != 1:
        raise ValueError(f"`run` takes one configuration, these overrides give {len(members)}; use `sweep`")

    sweep_id = f"{stage}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    sweep_dir = runs_root() / experiment / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=False)
    tag_name = "" if allow_dirty or state.dirty else f"sweep/{experiment}/{sweep_id}"
    meta = {
        "id": sweep_id,
        "experiment": experiment,
        "stage": stage,
        "kind": kind,
        "command": command,
        "overrides": list(overrides),
        "backend": backend,
        "commit": state.commit,
        "branch": state.branch,
        "tag": tag_name,
        "allow_dirty": allow_dirty,
        "dirty_paths": list(state.dirty_paths),
        "editable_dependencies": editable,  # {name: commit}: not frozen by the worktree, see provenance.py
        "host": socket.gethostname(),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "n_members": len(members),
    }
    (sweep_dir / "sweep.json").write_text(json.dumps(meta, indent=1))
    (sweep_dir / "jobs.jsonl").touch()
    for i, job in enumerate(members):
        add_job(sweep_dir, i, job)
    if tag_name:
        tag(tag_name, state.commit)  # keeps the commit reachable whatever happens to branches
    ensure_code(sweep_dir)
    return sweep_dir


def ensure_code(sweep_dir: Path) -> Path | None:
    """The sweep's frozen code (a worktree at its commit); None for --allow-dirty sweeps, which run the
    working tree. Created by single-process callers only (plan, launch, execute), never by array elements."""
    meta = read_sweep(sweep_dir)
    if meta.get("allow_dirty"):
        return None
    code = sweep_dir / ".code"
    if not (code / ".git").exists():
        git(repo_root(), "worktree", "prune")
        git(repo_root(), "worktree", "add", "--detach", "--quiet", str(code), meta["commit"])
    return code


def release_code(sweep_dir: Path) -> None:
    """Remove the worktree once every member is DONE (resume recreates it if ever needed)."""
    code = sweep_dir / ".code"
    if code.exists() and set(status(sweep_dir).values()) == {"DONE"}:
        git(repo_root(), "worktree", "remove", "--force", str(code))


def status(sweep_dir: Path) -> dict[int, str]:
    return {i: member_status(sweep_dir / str(i)) for i in sorted(jobs(sweep_dir))}


def counts(sweep_dir: Path) -> dict[str, int]:
    return dict(Counter(status(sweep_dir).values()))


def member_command(sweep_dir: Path, member: int, python: str | None = None) -> list[str]:
    meta = read_sweep(sweep_dir)
    return [
        python or sys.executable,
        "-m",
        f"{PACKAGE}.main",
        *jobs(sweep_dir)[member],
        f"stage={meta['stage']}",
        f"tracking.sweep_dir='{sweep_dir}'",
        f"tracking.member={member}",
        f"hydra.run.dir='{sweep_dir / str(member)}'",
    ]


def run_member(
    sweep_dir: Path, member: int, python: str | None = None, cwd: Path | None = None, env: dict[str, str] | None = None
) -> int:
    """Run one member in a subprocess, from the sweep's frozen code; forwards SIGUSR1 (SLURM preemption
    warning) to it."""
    out = sweep_dir / str(member)
    out.mkdir(exist_ok=True)
    env = dict(env or os.environ)
    code = sweep_dir / ".code"
    if not read_sweep(sweep_dir).get("allow_dirty"):
        if not (code / ".git").exists():
            raise RuntimeError(f"{code} is missing; `just sweep-resume {sweep_dir}` recreates it")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(code / "src"), env.get("PYTHONPATH")]))
        cwd = cwd or code
        for name, default in (
            ("RUNS_ROOT", runs_root()),
            ("DATA_ROOT", repo_root() / "data"),
            ("CKPT_ROOT", repo_root() / "checkpoints"),
        ):
            env.setdefault(f"{PREFIX}_{name}", str(var(name) or default))  # not the worktree's empty dirs
    with (out / "stdout.log").open("a") as log:
        proc = subprocess.Popen(
            member_command(sweep_dir, member, python), stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env
        )
        previous = signal.signal(signal.SIGUSR1, lambda s, f: proc.send_signal(s))
        try:
            return proc.wait()
        finally:
            signal.signal(signal.SIGUSR1, previous)


def start_time(when: str) -> dt.datetime:
    """'HH:MM' (today, or tomorrow if already past) or an ISO date-time, in local time."""
    if len(when) <= 5 and ":" in when:
        h, m = (int(x) for x in when.split(":"))
        now = dt.datetime.now()
        start = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return start if start > now else start + dt.timedelta(days=1)
    return dt.datetime.fromisoformat(when)


def wait_until(when: str | None) -> None:
    if not when:
        return
    start = start_time(when)
    print(f"{dt.datetime.now():%F %T} waiting until {start:%F %T}", flush=True)
    time.sleep(max(0.0, (start - dt.datetime.now()).total_seconds()))


def execute(
    sweep_dir: Path,
    members: Sequence[int] | None = None,
    python: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, int]:
    """Run members that are not DONE, one after another (local and detached backends)."""
    ensure_code(sweep_dir)
    todo = [i for i, s in status(sweep_dir).items() if s != "DONE" and (members is None or i in members)]
    for i in todo:
        rc = run_member(sweep_dir, i, python, cwd, env)
        print(f"{dt.datetime.now():%F %T} member {i}: {member_status(sweep_dir / str(i))} (rc={rc})", flush=True)
    release_code(sweep_dir)
    return counts(sweep_dir)
