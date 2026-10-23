"""Phase-1 analysis (notebook 04): effective-data curves, m*, and the E3 / E4 checks.

- `curve`: the effective-data curve NLL(m) per (cell, n_a), averaged over random subsets, made monotone
  (running minimum over m), since more correct data cannot hurt a Bayes learner.
- `m_star`: the m at which the curve reaches a model's NLL, interpolated in log2(1 + m). 0 if the model is no
  better than A alone; the largest m (inf flag) if it beats the whole curve.
"""

import numpy as np
import pandas as pd


def curve(aonly_eval: pd.DataFrame) -> pd.DataFrame:
    c = aonly_eval.groupby(["cell", "n_a", "m"], as_index=False)[["nll", "m_inside"]].mean().sort_values("m")
    c["nll_mono"] = c.groupby(["cell", "n_a"]).nll.cummin()
    return c


def m_star(cv: pd.DataFrame, cell: str, n_a: int, nll: float) -> float:
    h = cv.loc[(cv.cell == cell) & (cv.n_a == n_a)].sort_values("m")
    m, y = h.m.to_numpy(float), h.nll_mono.to_numpy()
    if nll >= y[0]:
        return 0.0
    if nll <= y[-1]:
        return float("inf")
    i = int(np.argmax(y <= nll))  # first m at which the curve is at or below the model
    t0, t1 = np.log2(1 + m[i - 1]), np.log2(1 + m[i])
    f = (y[i - 1] - nll) / max(y[i - 1] - y[i], 1e-12)
    return float(2 ** (t0 + f * (t1 - t0)) - 1)


def with_m_star(back_eval: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    out = back_eval.copy()
    out["m_star"] = [m_star(cv, r.cell, r.n_a, r.nll) for r in out.itertuples()]
    lower = cv.loc[cv.m == 0].set_index(["cell", "n_a"]).nll
    out["nll_lower"] = [lower[(r.cell, r.n_a)] for r in out.itertuples()]
    return out


def e3_checks(true_eval: pd.DataFrame, n_a=(2, 5, 10, 20)) -> dict[str, object]:
    """The pre-registered E3 rules on the true-seed, n_iter = 0 run (with m_star, nll_lower)."""
    t = true_eval.loc[true_eval.n_a.isin(n_a) & (true_eval.kind == "none")]
    c00 = t.loc[t.cell == "C00"]
    e3a = bool((c00.nll <= c00.nll_lower + 0.02).all() and (c00.m_star >= 128).all())
    worst = t.groupby("cell").m_star.min()
    e3b = "OK" if (worst >= 128).all() else ("FAIL" if (worst < 16).any() else "WEAK")
    return {"E3a_pass": e3a, "E3b": e3b, "min_m_star_per_cell": worst.round(1).to_dict()}


def e4_tables(ev8: pd.DataFrame, ev0: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (helps, radius). helps: per (cell, kind, sigma, n_a), the refining model with all vs 0 iterations and the
    read-out-only model. radius: per (cell, kind), the largest sigma at which refinement cuts the field error by
    >= 50% at n_A = 10."""
    a = ev8.loc[ev8.n_iter == ev8.n_iter.max()].set_index(["cell", "kind", "sigma", "n_a"])
    a0 = ev8.loc[ev8.n_iter == 0].set_index(["cell", "kind", "sigma", "n_a"])
    b = ev0.loc[ev0.n_iter == 0].set_index(["cell", "kind", "sigma", "n_a"])
    helps = pd.DataFrame(
        {
            "nll_refined": a.nll,
            "nll_same_model_no_iter": a0.nll,
            "nll_readout_only_model": b.nll,
            "m_star_refined": a.m_star,
            "m_star_readout_only": b.m_star,
            "err_seed": a.err_seed,
            "err_final": a.err_final,
            "w_null_refined": a.w_null,
            "w_null_readout_only": b.w_null,
        }
    ).reset_index()
    helps["refinement_helps"] = helps.nll_refined < helps.nll_readout_only_model
    r = helps.loc[(helps.n_a == 10) & (helps.kind != "none")]
    r = r.assign(cut=1 - r.err_final / r.err_seed.clip(lower=1e-9))
    radius = (
        (
            r.loc[r.cut >= 0.5]
            .groupby(["cell", "kind"])
            .sigma.max()
            .reindex(r.groupby(["cell", "kind"]).size().index)
            .fillna(0.0)
        )
        .rename("radius")
        .reset_index()
    )
    return helps, radius
