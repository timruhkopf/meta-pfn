"""`just status`: where this repository stands, and the next workflow steps with their commands.

Read by people, by the `workflow` Claude skill, and (brief form) at the start of every Claude session.
It only reads state; it never fetches, commits or pushes.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass, field

from ..utils.env import repo_root, runs_root
from .codesync import active_sweeps
from .logbook import find_entry
from .provenance import git, git_state
from .sweep import read_sweep, status


@dataclass
class Report:
    facts: list[str] = field(default_factory=list)
    steps: list[tuple[str, str]] = field(default_factory=list)  # (why, command)

    def step(self, why: str, command: str) -> None:
        self.steps.append((why, command))

    def render(self, brief: bool = False) -> str:
        lines = ["## workflow status", *(f"- {f}" for f in self.facts)]
        if self.steps:
            lines.append("next:" if brief else "\n## suggested next steps")
            lines += [f"- {why}: `{cmd}`" for why, cmd in self.steps[: 4 if brief else None]]
        return "\n".join(lines)


def _try(*args: str) -> str | None:
    try:
        return git(repo_root(), *args)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git(report: Report) -> None:
    state = git_state()
    branch = state.branch
    porcelain = _try("status", "--porcelain=v1") or ""
    notes = len(porcelain.splitlines()) - len(state.dirty_paths)
    issue = re.search(r"/(\d+)-", branch)
    report.facts.append(f"branch `{branch}`" + (f" (issue #{issue.group(1)})" if issue else ""))
    if branch in ("main", "master"):
        report.step("work happens on a branch per issue", "gh issue create  # then: git switch -c exp/<N>-<slug>")
    if state.dirty_paths:
        report.facts.append(f"{len(state.dirty_paths)} uncommitted change(s) that block tracked runs")
        report.step("commit code before any tracked run (hooks run ruff)", "git add -A && git commit -m '...'")
    if notes > 0:
        report.facts.append(f"{notes} uncommitted note/result file(s) (log, notebooks, results)")
    counts = _try("rev-list", "--left-right", "--count", "@{u}...HEAD")
    if counts is None and branch not in ("HEAD", "none"):
        report.step("branch has no upstream yet", "git push -u origin HEAD")
    elif counts:
        behind, ahead = (int(x) for x in counts.split())
        if ahead:
            report.facts.append(f"{ahead} commit(s) not pushed")
            report.step("push (backup, and needed before remote launches)", "git push")
        if behind:
            report.step(f"{behind} commit(s) to pull", "git pull --ff-only")


def _pull_request(report: Report) -> None:
    if shutil.which("gh") is None:
        return
    try:
        out = subprocess.run(
            ["gh", "pr", "view", "--json", "number,state,isDraft"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root(),
        )
    except subprocess.TimeoutExpired:
        return
    if out.returncode == 0 and out.stdout:
        import json

        pr = json.loads(out.stdout)
        kind = "draft PR" if pr["isDraft"] else "PR"
        report.facts.append(f"{kind} #{pr['number']} ({pr['state'].lower()})")


def _sweeps(report: Report, limit: int = 8) -> None:
    root = runs_root()
    active = {d for d, _ in active_sweeps(root)}
    if active:
        report.facts.append(
            f"{len(active)} sweep(s) running: " + ", ".join(f"{d.parent.name}/{d.name}" for d in active)
        )
    log_dir = repo_root() / "research_log"
    sweeps = sorted(root.glob("*/*/sweep.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for meta_file in sweeps[:limit]:
        d = meta_file.parent
        if d in active or "_repro_" in d.name:
            continue
        meta = read_sweep(d)
        states = set(status(d).values())
        name = f"{meta['experiment']}/{meta['id']}"
        if states - {"DONE", "FAILED"}:
            report.step(f"{name} is unfinished and not running", f"just sweep-status {d}")
            continue
        snap = repo_root() / "results" / meta["experiment"] / meta["id"]
        if not snap.exists():
            report.step(f"{name} finished: snapshot it for its notebook", f"just snapshot {d}")
            continue
        entry = find_entry(log_dir, meta["experiment"]) if log_dir.exists() else None
        decided = entry is not None and any(s.get("id") == meta["id"] for s in entry.sweeps)
        if not decided and meta["stage"] != "sanity" and not meta.get("allow_dirty"):
            report.step(
                f"{name}: run its notebook, then record the verdict",
                f'just notebook notebooks/<nb> && just decide {d} <GO|NO_GO|INCONCLUSIVE> "<reason>"',
            )


def report(brief: bool = False) -> str:
    r = Report()
    _git(r)
    if not brief:
        _pull_request(r)
    _sweeps(r)
    return r.render(brief)
