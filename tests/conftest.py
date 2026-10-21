import os
import subprocess
from pathlib import Path

import pytest
from hypothesis import settings

settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("ci", max_examples=500, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

PREFIX = "META_PFN"


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Runs (and member subprocesses, which inherit the env) write under tmp_path."""
    root = tmp_path / "runs"
    monkeypatch.setenv(f"{PREFIX}_RUNS_ROOT", str(root))
    return root


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A stand-in repo root for research_log/, results/, notebooks/, configs/."""
    r = tmp_path / "root"
    (r / "research_log" / "entries").mkdir(parents=True)
    (r / "notebooks").mkdir()
    (r / "results").mkdir()
    (r / "src" / "meta_pfn" / "configs" / "experiment").mkdir(parents=True)
    return r


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (repo / "code.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "init")
    return repo
