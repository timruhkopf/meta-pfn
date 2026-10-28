"""Sanity ladder S0/S1 analysis (notebook 08): per-cell summaries with SEs over pairs and the drafted rules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# one fixed colour per predictor in every figure (dataviz reference palette, light mode; a_only neutral)
COLORS = {
    "a_only": "#6b6a64",
    "oracle_q": "#2a78d6",
    "grid_pseudo": "#eb6834",
    "teacher_b": "#1baf7a",
    "map_pseudo": "#eda100",
    "bayes_null": "#e87ba4",
    "teacher_a": "#008300",
    "grid_exact": "#6250d6",
    "naive": "#e34948",
}
S0_PRED = ("a_only", "oracle_q", "teacher_a", "teacher_b", "naive")
S1_PRED = ("a_only", "oracle_q", "grid_pseudo", "map_pseudo", "grid_exact", "bayes_null")


def with_cell(df: pd.DataFrame, snap_path: Path) -> pd.DataFrame:
    """Attach d and n_b from the members' overrides (the snapshot's sweep/jobs.jsonl)."""
    rows = []
    for line in (snap_path / "sweep" / "jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        ov = dict(o.split("=", 1) for o in job["overrides"])
        rows.append({"member": job["member"], "d": int(ov["task.d"]), "n_b": int(ov["task.n_b"])})
    return df.merge(pd.DataFrame(rows), on="member")


def nll_summary(nll: pd.DataFrame) -> pd.DataFrame:
    g = nll.groupby(["d", "n_b", "kind", "n_a", "predictor"]).nll
    return pd.DataFrame({"mean": g.mean(), "se": g.sem(), "n": g.size()}).reset_index()


def paired(nll: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """NLL(a) - NLL(b) per (d, n_b, kind, n_a): mean, SE and 95% CI over pairs (positive: b is better)."""
    w = nll.pivot_table(index=["d", "n_b", "kind", "n_a", "pair"], columns="predictor", values="nll")
    diff = (w[a] - w[b]).groupby(level=["d", "n_b", "kind", "n_a"])
    out = pd.DataFrame({"mean": diff.mean(), "se": diff.sem(), "n": diff.size()})
    out["lo"], out["hi"] = out["mean"] - 1.96 * out.se, out["mean"] + 1.96 * out.se
    return out.reset_index()


def s0_verdict(nll: pd.DataFrame, n_b: int = 80) -> tuple[str, pd.DataFrame]:
    """P1: a_only - oracle_q >= 0.5 with CI lo > 0; P2: |oracle_q - teacher_b| <= 0.1, |oracle_q - teacher_a| <= 0.25;
    for both d at n_A in {0, 1, 2}, shift stratum."""
    sel = lambda df: df[(df.kind == "shift") & (df.n_b == n_b) & df.n_a.isin([0, 1, 2])].set_index(["d", "n_a"])  # noqa: E731
    p1 = sel(paired(nll, "a_only", "oracle_q"))
    tb = sel(paired(nll, "oracle_q", "teacher_b"))
    ta = sel(paired(nll, "oracle_q", "teacher_a"))
    t = pd.DataFrame({"gain_oracle": p1["mean"], "gain_lo": p1.lo, "oracle-teacher_b": tb["mean"],
                      "oracle-teacher_a": ta["mean"]})  # fmt: skip
    t["P1"] = (t.gain_oracle >= 0.5) & (t.gain_lo > 0)
    t["P2"] = (t["oracle-teacher_b"].abs() <= 0.1) & (t["oracle-teacher_a"].abs() <= 0.25)
    return ("PASS" if bool(t.P1.all() and t.P2.all()) else "FAIL"), t


def s1_table(nll: pd.DataFrame, post: pd.DataFrame, gd: pd.DataFrame, kind: str = "shift") -> pd.DataFrame:
    """Per (d, n_b, n_a): gap closure of each S1 predictor (a_only -> oracle_q, ratio of means), posterior summaries
    (pseudo / exact) and GD success."""
    s = nll_summary(nll.loc[nll.kind == kind]).pivot_table(
        index=["d", "n_b", "n_a"], columns="predictor", values="mean"
    )
    gap = s.a_only - s.oracle_q
    out = pd.DataFrame({f"close_{k}": (s.a_only - s[k]) / gap for k in ("grid_pseudo", "map_pseudo", "grid_exact")})
    for meth in ("pseudo", "exact"):
        p = post[(post.kind == kind) & (post.method == meth)].groupby(["d", "n_b", "n_a"])
        out[f"unimodal_{meth}"] = p.n_modes.apply(lambda x: (x == 1).mean())
        out[f"med_err_map_{meth}"] = p.err_map.median()
        out[f"entropy_{meth}"] = p.entropy.mean()
        out[f"mass_true_{meth}"] = p.mass_true.mean()
    for start, g in gd[gd.kind == kind].groupby("start"):
        g = g.groupby(["d", "n_b", "n_a"])
        out[f"gd_success_{start}"] = g.success.mean()
        out[f"gd_med_err_true_{start}"] = g.err_true.median()
    return out.reset_index()


def s1_verdict(tab: pd.DataFrame, n_b: int = 80, n_a: int = 5) -> str:
    """GO: for both d, grid_pseudo closes >= 50% and unimodal (pseudo) >= 0.6; NO-GO: closes < 20% for either d."""
    t = tab[(tab.n_b == n_b) & (tab.n_a == n_a)].set_index("d")
    if bool((t.close_grid_pseudo < 0.2).any()):
        return "NO-GO"
    if bool(((t.close_grid_pseudo >= 0.5) & (t.unimodal_pseudo >= 0.6)).all()) and len(t) == 2:
        return "GO"
    return "INCONCLUSIVE"


def mc_error(nll: pd.DataFrame) -> float:
    """Largest SE over pairs of any mean NLL (the oracle's and every predictor's MC error)."""
    return float(np.nanmax(nll_summary(nll).se))
