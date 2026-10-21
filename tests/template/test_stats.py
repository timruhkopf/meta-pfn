import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from meta_pfn.analysis.stats import bootstrap_ci, paired_diff, summarize


def _runs() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "status": ["DONE", "DONE", "FAILED", "DONE", "DONE", "MISSING"],
            "config_hash": ["a", "a", "a", "b", "b", "b"],
            "git_commit": ["c"] * 6,
            "method": ["x", "x", "x", "y", "y", "y"],
            "seed": [0, 1, 2, 0, 1, 2],
            "final/loss": [1.0, 3.0, np.nan, 2.0, 2.0, np.nan],
        }
    )
    df.attrs["params"] = ["method", "seed"]
    return df


def test_summarize_uses_done_only_and_reports_the_rest() -> None:
    s = summarize(_runs(), "final/loss").set_index("method")
    assert s.loc["x", "mean"] == 2.0 and s.loc["x", "n_done"] == 2 and s.loc["x", "n_failed"] == 1
    assert s.loc["y", "n_missing"] == 1
    assert s["preliminary"].all()


def test_paired_diff_counts_unpaired() -> None:
    r = paired_diff(_runs(), "final/loss", "method", "x", "y")
    assert r["n_pairs"] == 2 and r["n_unpaired"] == 1 and r["preliminary"]
    assert r["mean_diff"] == 0.0


@given(st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=50))
def test_ci_brackets_are_ordered_and_within_range(xs: list[float]) -> None:
    lo, hi = bootstrap_ci(xs, n_boot=200)
    assert min(xs) - 1e-6 <= lo <= hi <= max(xs) + 1e-6
