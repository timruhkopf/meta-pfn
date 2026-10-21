from pathlib import Path

import pytest

from meta_pfn.analysis.snapshot import snapshot
from meta_pfn.tracking import sweep
from meta_pfn.tracking.provenance import git, git_state
from meta_pfn.tracking.reproduce import reproduce
from meta_pfn.utils.env import repo_root


@pytest.mark.slow
def test_reproduce_from_sweep_and_from_snapshot(runs_root: Path, root: Path) -> None:
    if git_state().dirty:
        pytest.skip("needs a clean, committed checkout (as in CI)")
    d = sweep.plan(
        "sanity", ["experiment=debug", "seed=0,1"], command="test", backend="local", log_dir=root / "research_log"
    )
    try:
        sweep.execute(d)
        assert reproduce(d, sync=False)
        # a fresh clone has no runs/, only the committed snapshot
        assert reproduce(snapshot(d, root / "results"), sync=False, members=[1])
    finally:
        git(repo_root(), "tag", "-d", f"sweep/debug/{d.name}")
