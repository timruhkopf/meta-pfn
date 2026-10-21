"""Aggregation over DONE members only; the other statuses are counted and reported, never dropped silently.

A group with any member not DONE is flagged `preliminary`: its numbers may still change.
"""

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STATUS_COLUMNS = ["n_done", "n_failed", "n_started", "n_missing"]


def bootstrap_ci(
    values: np.ndarray | Sequence[float],
    stat: Callable = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx])
    alpha = (1 - level) / 2
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def _status_counts(group: pd.DataFrame) -> dict[str, int]:
    s = group["status"]
    return {
        "n_done": int((s == "DONE").sum()),
        "n_failed": int((s == "FAILED").sum()),
        "n_started": int((s == "STARTED").sum()),
        "n_missing": int((s == "MISSING").sum()),
    }


def summarize(df: pd.DataFrame, metric: str, by: Sequence[str] | None = None, level: float = 0.95) -> pd.DataFrame:
    """Mean and bootstrap CI of `metric` per configuration, over DONE members only.

    by: grouping columns; default = the varying params except replicate axes (seed), plus git_commit so
    replicates of different code versions are never pooled.
    """
    if by is None:
        by = [p for p in df.attrs.get("params", []) if p != "seed"] + ["git_commit"]
    by = list(by)
    if "config_hash" in df and bool(df.groupby("config_hash", dropna=True)["git_commit"].nunique().gt(1).any()):
        warnings.warn("some configurations ran at several commits; they are reported separately", stacklevel=2)
    rows = []
    for key, group in df.groupby(by, dropna=False, sort=True) if by else [((), df)]:
        done = group.loc[group["status"] == "DONE", metric].dropna().to_numpy(dtype=float)
        lo, hi = bootstrap_ci(done, level=level)
        counts = _status_counts(group)
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                **counts,
                "mean": done.mean() if len(done) else np.nan,
                "std": done.std(ddof=1) if len(done) > 1 else np.nan,
                "ci_lo": lo,
                "ci_hi": hi,
                "preliminary": counts["n_done"] < len(group),
            }
        )
    out = pd.DataFrame(rows)
    if "git_commit" in out:
        out["git_commit"] = out["git_commit"].astype(str).str[:8]
    return out


def paired_diff(
    df: pd.DataFrame, metric: str, column: str, a: object, b: object, on: str = "seed", level: float = 0.95
) -> dict[str, float | int | bool]:
    """Mean of metric(a) - metric(b), paired on `on` (seed / instance), over pairs where both are DONE."""
    done = df.loc[df["status"] == "DONE"]
    left = done.loc[done[column] == a].set_index(on)[metric]
    right = done.loc[done[column] == b].set_index(on)[metric]
    common = left.index.intersection(right.index)
    n_all = len(set(df.loc[df[column] == a, on]) | set(df.loc[df[column] == b, on]))
    d = (left.loc[common] - right.loc[common]).to_numpy(dtype=float)
    lo, hi = bootstrap_ci(d, level=level)
    return {
        "mean_diff": float(d.mean()) if len(d) else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_pairs": len(d),
        "n_unpaired": n_all - len(d),
        "preliminary": len(d) < n_all,
    }
