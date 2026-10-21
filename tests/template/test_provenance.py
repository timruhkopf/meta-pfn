import subprocess
from pathlib import Path

import pytest

from meta_pfn.tracking.provenance import DirtyTreeError, git_state, require_clean
from meta_pfn.tracking.run import _check_member_provenance


def test_clean_repo_passes(git_repo: Path) -> None:
    assert not require_clean(git_repo).dirty


@pytest.mark.parametrize("path", ["new_module.py", "uv.lock", "scripts/run.sh", "src/pkg/configs/x.yaml"])
def test_untracked_or_lockfile_changes_block(git_repo: Path, path: str) -> None:
    (git_repo / path).parent.mkdir(parents=True, exist_ok=True)
    (git_repo / path).write_text("changed\n")
    with pytest.raises(DirtyTreeError):
        require_clean(git_repo)


def test_modified_tracked_file_blocks(git_repo: Path) -> None:
    (git_repo / "code.py").write_text("x = 2\n")
    assert git_state(git_repo).dirty_paths == ("code.py",)


@pytest.mark.parametrize(
    "path", ["research_log/entries/x.md", "notebooks/01.ipynb", "results/a/b.parquet", "references.md"]
)
def test_allowlisted_paths_do_not_block(git_repo: Path, path: str) -> None:
    (git_repo / path).parent.mkdir(parents=True, exist_ok=True)
    (git_repo / path).write_text("notes\n")
    assert not require_clean(git_repo).dirty


def test_not_a_repository_counts_as_dirty(tmp_path: Path) -> None:
    assert git_state(tmp_path).dirty


def test_member_refuses_other_commit() -> None:
    """A member must not run code committed after its sweep was planned."""
    with pytest.raises(DirtyTreeError):
        _check_member_provenance({"id": "x", "commit": "0" * 40, "allow_dirty": False})
    _check_member_provenance({"id": "x", "commit": "0" * 40, "allow_dirty": True})


def test_tag_created(git_repo: Path) -> None:
    from meta_pfn.tracking.provenance import tag

    commit = git_state(git_repo).commit
    tag("sweep/exp/pilot_1", commit, git_repo)
    out = subprocess.check_output(["git", "-C", str(git_repo), "rev-list", "-n1", "sweep/exp/pilot_1"], text=True)
    assert out.strip() == commit


def test_editable_dependencies_are_checked(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from meta_pfn.tracking import provenance

    monkeypatch.setattr(provenance, "editable_dependencies", lambda: {"dep": git_repo})
    commits = provenance.require_clean_dependencies("pilot")
    assert commits == {"dep": git_state(git_repo).commit}
    with pytest.raises(DirtyTreeError, match="git-pinned"):
        provenance.require_clean_dependencies("confirm")
    (git_repo / "code.py").write_text("x = 3\n")
    with pytest.raises(DirtyTreeError, match="uncommitted"):
        provenance.require_clean_dependencies("pilot")


def test_no_editable_dependencies_in_a_plain_project() -> None:
    from meta_pfn.tracking.provenance import editable_dependencies

    assert editable_dependencies() == {}  # the project itself is excluded
