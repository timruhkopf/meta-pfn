"""Where members execute. All backends read the same sweep dir, so status and resume work for each.

local     foreground, one member after another
detached  the same, but in a new session with nohup semantics: survives logout (used on an ssh box)
slurm     one array element per member (or per optuna worker), requeued across the time limit
"""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ..utils.env import PACKAGE, repo_root
from .sweep import ensure_code, execute, status

BACKENDS = ("local", "detached", "slurm")


def launch(
    sweep_dir: Path,
    backend: str,
    members: Sequence[int] | None = None,
    *,
    workers: int = 1,
    preset: str = "gpu_small",
    max_parallel: int = 16,
    start_at: str | None = None,
) -> None:
    """Grid sweeps run `members` (default: all not DONE); optuna sweeps start `workers` workers.
    start_at ('HH:MM' or ISO, detached only): plan and freeze now, start then."""
    if start_at and backend != "detached":
        raise ValueError("--start-at needs the detached backend (slurm: use sbatch --begin in the preset)")
    ensure_code(sweep_dir)  # here, once, not in every array element
    optuna = (sweep_dir / "study.journal").exists()
    if backend == "local":
        if optuna:
            from .cli import main  # runs the worker in this process

            main(["optuna-worker", str(sweep_dir)])
        else:
            print(execute(sweep_dir, members))
    elif backend == "detached":
        _detached(sweep_dir, members, optuna, start_at)
    elif backend == "slurm":
        _slurm(sweep_dir, members, optuna, workers, preset, max_parallel)
    else:
        raise ValueError(f"backend must be one of {BACKENDS}")


def _detached(sweep_dir: Path, members: Sequence[int] | None, optuna: bool, start_at: str | None = None) -> None:
    args = ["optuna-worker", str(sweep_dir)] if optuna else ["execute", str(sweep_dir)]
    if start_at and not optuna:
        args += ["--start-at", start_at]
    if members and not optuna:
        args += ["--members", *map(str, members)]
    log = sweep_dir / "queue.log"
    with log.open("a") as f:
        proc = subprocess.Popen(
            [sys.executable, "-m", f"{PACKAGE}.tracking.cli", *args],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=repo_root(),
        )
    (sweep_dir / "queue.pid").write_text(str(proc.pid))
    print(
        f"launched pid {proc.pid}; log: {log}\n"
        f"verify (don't relaunch on an ssh timeout): ps -o pid,etime,args -p {proc.pid}"
    )


def _slurm(
    sweep_dir: Path, members: Sequence[int] | None, optuna: bool, workers: int, preset: str, max_parallel: int
) -> None:
    script = repo_root() / "scripts" / "slurm" / "presets" / f"{preset}.sbatch"
    if not script.exists():
        raise FileNotFoundError(f"no SLURM preset {script}")
    if optuna:
        indices, mode = list(range(workers)), "optuna-worker"
    else:
        indices = list(members) if members else [i for i, s in status(sweep_dir).items() if s != "DONE"]
        mode = "member"
    if not indices:
        print("nothing to submit: all members DONE")
        return
    cmd = [
        "sbatch",
        "--parsable",
        f"--array={','.join(map(str, indices))}%{max_parallel}",
        f"--job-name={sweep_dir.parent.name}",
        f"--output={sweep_dir}/slurm_%A_%a.out",
        str(script),
        str(repo_root() / "scripts" / "slurm" / "array_task.sh"),
        mode,
        str(sweep_dir),
    ]
    job_id = subprocess.check_output(cmd, text=True).strip()
    with (sweep_dir / "slurm_jobs.txt").open("a") as f:
        f.write(f"{job_id} {mode} {','.join(map(str, indices))}\n")
    print(f"submitted array job {job_id} ({len(indices)} elements)")
