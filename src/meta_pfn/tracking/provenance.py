"""Git state of the package's repository: tracked runs refuse uncommitted changes that can affect results."""

import json
import subprocess
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..utils.env import repo_root

# Changes here cannot affect results, so they don't block a tracked launch. Everything else does,
# including untracked files, uv.lock, pyproject.toml, justfile and scripts/.
ALLOWED_DIRTY = (
    "research_log/",
    "notebooks/",
    "results/",
    "reports/",
    "docs/",
    "references.md",
    "references.bib",
    "README.md",
    "CLAUDE.md",
)


class DirtyTreeError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitState:
    commit: str
    branch: str
    dirty_paths: tuple[str, ...]

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_paths)


def git(repo: Path, *args: str) -> str:
    # -C: never rely on the cwd (Hydra may chdir into the run dir).
    # stderr captured: e.g. `rev-parse HEAD` before a repo's first commit is expected to fail, quietly
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def git_state(repo: Path | None = None) -> GitState:
    repo = repo or repo_root()
    try:
        commit = git(repo, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return GitState(commit="none", branch="none", dirty_paths=("<not a git repository or no commit>",))
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    porcelain = subprocess.check_output(  # not git(): stripping would eat the first line's status column
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )
    paths = []
    for line in porcelain.splitlines():
        path = line[3:].split(" -> ")[-1].strip('"')
        if not path.startswith(ALLOWED_DIRTY):
            paths.append(path)
    return GitState(commit=commit, branch=branch, dirty_paths=tuple(paths))


def require_clean(repo: Path | None = None) -> GitState:
    state = git_state(repo)
    if state.dirty:
        listing = "\n  ".join(state.dirty_paths[:20])
        raise DirtyTreeError(
            f"uncommitted changes that can affect results:\n  {listing}\n"
            "Commit them first (or use a scratch run: `uv run <package> ...`)."
        )
    return state


def tag(name: str, commit: str, repo: Path | None = None) -> None:
    """Annotated, so `git push` carries it (push.followTags, set by `just setup`)."""
    repo = repo or repo_root()
    identity = []
    if subprocess.run(["git", "-C", str(repo), "config", "user.email"], capture_output=True).returncode != 0:
        identity = ["-c", "user.name=sweep", "-c", "user.email=sweep@localhost"]  # e.g. a GPU box without git identity
    subprocess.check_output(["git", *identity, "-C", str(repo), "tag", "-a", "-m", f"sweep {name}", name, commit])


def editable_dependencies() -> dict[str, Path]:
    """Packages installed in editable mode from another local checkout (e.g. `uv add --editable ../lib`).

    Their code is not covered by this repo's commit or uv.lock, so tracked runs check and record them."""
    out = {}
    for dist in distributions():
        text = dist.read_text("direct_url.json")
        if not text:
            continue
        info = json.loads(text)
        if not info.get("dir_info", {}).get("editable") or not info.get("url", "").startswith("file://"):
            continue
        path = Path(unquote(urlparse(info["url"]).path))
        if path.resolve() != repo_root().resolve():
            out[dist.metadata["Name"]] = path
    return out


def require_clean_dependencies(stage: str) -> dict[str, str]:
    """-> {name: commit} of editable dependencies; refuses uncommitted ones, and any at confirm/final,
    where only git-pinned dependencies (fixed by uv.lock) make a result reproducible."""
    deps = editable_dependencies()
    if deps and stage in ("confirm", "final"):
        raise DirtyTreeError(
            f"stage {stage!r} needs git-pinned dependencies, but these are editable: {', '.join(sorted(deps))}. "
            "Point [tool.uv.sources] at a tag (see the dependency guide) and `uv sync` first."
        )
    commits = {}
    for name, path in sorted(deps.items()):
        state = git_state(path)
        if state.dirty:
            raise DirtyTreeError(
                f"editable dependency {name} ({path}) has uncommitted changes: {', '.join(state.dirty_paths[:5])}"
            )
        commits[name] = state.commit
    return commits
