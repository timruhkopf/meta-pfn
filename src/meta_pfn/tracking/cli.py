"""Command line behind the `just` recipes (they map 1:1, so `just sweep ...` == `cli sweep ...`).

Runs on any machine with the package's venv, e.g. on a remote box via `just remote sweep ...`.
"""

import argparse
import re
import shlex
import sys
from pathlib import Path

import pandas as pd

from ..utils.env import load_dotenv, repo_root, runs_root
from . import logbook, sweep
from .backends import BACKENDS, launch
from .codesync import SyncRefused
from .provenance import DirtyTreeError


def _command() -> str:
    args = sys.argv[1:]
    if "--root" in args:  # test-only option, not part of the command a human would type
        i = args.index("--root")
        args = args[:i] + args[i + 2 :]
    return "just " + shlex.join(args)


def cmd_run(a: argparse.Namespace) -> None:
    d = sweep.plan(
        a.stage,
        a.overrides,
        command=_command(),
        backend=a.backend,
        log_dir=a.root / "research_log",
        allow_dirty=a.allow_dirty,
        single=a.cmd == "run",
    )
    print(f"planned {d} ({len(sweep.jobs(d))} members)")
    launch(d, a.backend, preset=a.preset, max_parallel=a.max_parallel, start_at=a.start_at)


def cmd_status(a: argparse.Namespace) -> None:
    st = sweep.status(a.sweep_dir)
    sweep.release_code(a.sweep_dir)
    for i, s in st.items():
        print(f"{i:>4}  {s:8}  {' '.join(sweep.jobs(a.sweep_dir)[i])}")
    print(sweep.counts(a.sweep_dir))
    if "STARTED" in st.values():
        print("STARTED = running, killed or preempted. Check (ps / squeue) before resuming.")


def cmd_resume(a: argparse.Namespace) -> None:
    backend = a.backend or sweep.read_sweep(a.sweep_dir)["backend"]
    launch(
        a.sweep_dir,
        backend,
        a.members or None,
        preset=a.preset,
        max_parallel=a.max_parallel,
    )


def cmd_execute(a: argparse.Namespace) -> None:
    sweep.wait_until(a.start_at)
    print(sweep.execute(a.sweep_dir, a.members or None), flush=True)


def cmd_member(a: argparse.Namespace) -> None:
    sys.exit(sweep.run_member(a.sweep_dir, a.member))


def cmd_runs(a: argparse.Namespace) -> None:
    from ..analysis.runs import load_runs

    df = load_runs(a.sweep_dir)
    print("constant:", " ".join(f"{k}={v}" for k, v in df.attrs["constant"].items()))
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(df.to_string(index=False))


def cmd_decide(a: argparse.Namespace) -> None:
    meta = sweep.read_sweep(a.sweep_dir)
    entry = logbook.decide(a.root / "research_log", meta, sweep.counts(a.sweep_dir), a.verdict, a.reason)
    print(f"recorded {a.verdict} for {meta['id']} in {entry.path}")


def cmd_workflow(a: argparse.Namespace) -> None:
    from .workflow import report

    print(report(brief=a.brief))


def cmd_sync_code(a: argparse.Namespace) -> None:
    from .codesync import sync_code

    sync_code(a.ref, force=a.force)


def cmd_notify_merge(a: argparse.Namespace) -> None:
    from .codesync import notify_after_merge

    notify_after_merge()


def cmd_log_index(a: argparse.Namespace) -> None:
    print(logbook.write_index(a.root / "research_log"))


def cmd_snapshot(a: argparse.Namespace) -> None:
    from ..analysis.snapshot import snapshot

    out = snapshot(a.sweep_dir, a.root / "results")
    print(f"snapshot: {out}")


def cmd_reproduce(a: argparse.Namespace) -> None:
    from .reproduce import reproduce

    sys.exit(0 if reproduce(a.sweep_dir, sync=not a.no_sync, members=a.members or None) else 1)


def cmd_new_experiment(a: argparse.Namespace) -> None:
    from .scaffold import new_experiment

    for path in new_experiment(a.root, a.name, a.title):
        print(f"created {path.relative_to(a.root)}")


def cmd_refs_check(a: argparse.Namespace) -> None:
    md = re.sub(r"```.*?```", "", (a.root / "references.md").read_text(), flags=re.S)  # skip the entry template
    md_keys = set(re.findall(r"^###\s*\[([^\]]+)\]", md, flags=re.M))
    bib_path = a.root / "references.bib"
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_path.read_text() if bib_path.exists() else ""))
    problems = [f"annotated but not in references.bib: {k}" for k in sorted(md_keys - bib_keys)]
    problems += [f"in references.bib but not annotated: {k}" for k in sorted(bib_keys - md_keys)]
    print("\n".join(problems) or f"references ok ({len(md_keys)} entries)")
    sys.exit(1 if problems else 0)


def cmd_mlflow_sync(a: argparse.Namespace) -> None:
    from .mlflow_sync import sync

    sync(a.paths or [runs_root()], a.root / "research_log")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="just", description=__doc__)
    p.add_argument("--root", type=Path, default=repo_root(), help=argparse.SUPPRESS)  # tests only
    sub = p.add_subparsers(dest="cmd", required=True)

    def launch_opts(s: argparse.ArgumentParser) -> None:
        s.add_argument("--preset", default="gpu_small", help="SLURM preset in scripts/slurm/presets/")
        s.add_argument("--max-parallel", type=int, default=16, help="SLURM array throttle")
        s.add_argument("--start-at", help="detached: plan and freeze now, start at 'HH:MM' or an ISO time")

    for name, help_ in (("run", "one tracked run"), ("sweep", "a tracked sweep (Hydra sweep syntax)")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("backend", choices=BACKENDS)
        s.add_argument("stage", choices=logbook.STAGES)
        s.add_argument("--allow-dirty", action="store_true", help="debugging only: not reportable, not tagged")
        launch_opts(s)
        s.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=x seed=0,1,2")
        s.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep-status", help="state of every member")
    s.add_argument("sweep_dir", type=Path)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("sweep-resume", help="re-execute members that are not DONE")
    s.add_argument("sweep_dir", type=Path)
    s.add_argument("backend", nargs="?", choices=BACKENDS)
    s.add_argument("--members", type=int, nargs="*")
    launch_opts(s)
    s.set_defaults(func=cmd_resume)

    s = sub.add_parser("execute", help="run members that are not DONE here, one after another")
    s.add_argument("sweep_dir", type=Path)
    s.add_argument("--members", type=int, nargs="*")
    s.add_argument("--start-at", help="'HH:MM' or ISO local time: wait, then execute")
    s.set_defaults(func=cmd_execute)

    s = sub.add_parser("member", help="run one member (used by SLURM array elements)")
    s.add_argument("sweep_dir", type=Path)
    s.add_argument("member", type=int)
    s.set_defaults(func=cmd_member)

    for name, func, help_ in (
        ("runs", cmd_runs, "table of a sweep"),
        ("snapshot", cmd_snapshot, "export a snapshot into results/"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("sweep_dir", type=Path)
        s.set_defaults(func=func)

    s = sub.add_parser("decide", help="record a verdict in the experiment's log entry")
    s.add_argument("sweep_dir", type=Path)
    s.add_argument("verdict", choices=logbook.VERDICTS)
    s.add_argument("reason")
    s.set_defaults(func=cmd_decide)

    sub.add_parser("log-index", help="regenerate research_log/README.md").set_defaults(func=cmd_log_index)

    s = sub.add_parser("status", help="where the repo stands and the next workflow steps")
    s.add_argument("--brief", action="store_true")
    s.set_defaults(func=cmd_workflow)

    s = sub.add_parser("sync-code", help="move this checkout to a ref, guarding running sweeps")
    s.add_argument("ref", nargs="?", default="@{u}", help="commit/branch, or @{u} to fast-forward (default)")
    s.add_argument("--force", action="store_true", help="also when dependencies change under active sweeps")
    s.set_defaults(func=cmd_sync_code)

    sub.add_parser("notify-merge", help="post-merge hook: report active sweeps").set_defaults(func=cmd_notify_merge)
    sub.add_parser("refs-check", help="references.md and .bib agree").set_defaults(func=cmd_refs_check)

    s = sub.add_parser("reproduce", help="rerun a sweep at its commit and compare")
    s.add_argument("sweep_dir", type=Path)
    s.add_argument("--no-sync", action="store_true", help="reuse this venv instead of `uv sync --frozen`")
    s.add_argument("--members", type=int, nargs="*")
    s.set_defaults(func=cmd_reproduce)

    s = sub.add_parser("new-experiment", help="config + notebook + log entry")
    s.add_argument("name")
    s.add_argument("title")
    s.set_defaults(func=cmd_new_experiment)

    s = sub.add_parser("mlflow-sync", help="push run sinks to MLflow (one way, idempotent)")
    s.add_argument("paths", type=Path, nargs="*")
    s.set_defaults(func=cmd_mlflow_sync)

    return p


def main(argv: list[str] | None = None) -> None:
    load_dotenv(repo_root() / ".env", repo_root() / ".env.secrets")
    p = parser()
    args, extra = p.parse_known_args(argv)  # options may come after the overrides
    if extra and hasattr(args, "overrides") and not any(e.startswith("-") for e in extra):
        args.overrides += extra
    elif extra:
        p.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        args.func(args)
    except (logbook.GateError, DirtyTreeError, FileExistsError, SyncRefused) as e:  # refusals by design: no traceback
        sys.exit(f"refused: {e}")


if __name__ == "__main__":
    main()
