import subprocess
import sys
from pathlib import Path


def test_scratch_run(runs_root: Path) -> None:
    """`uv run <package> experiment=debug` works without a commit and lands in runs/_scratch/."""
    subprocess.run([sys.executable, "-m", "meta_pfn.main", "experiment=debug"], check=True, capture_output=True)
    (run_dir,) = (runs_root / "_scratch" / "debug").iterdir()
    assert (run_dir / "DONE").exists()
    assert (run_dir / "tracking" / "artifacts" / "tables" / "coefficients.parquet").exists()


def test_test_split_embargo(runs_root: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "meta_pfn.main", "experiment=debug", "task.eval_split=test"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    (run_dir,) = (runs_root / "_scratch" / "debug").iterdir()
    assert "SplitEmbargoError" in (run_dir / "FAILED").read_text()
