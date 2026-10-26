"""Field-only analysis (notebook 07): per-pair field-error summaries with pair-bootstrap CIs, and the pre-registered
rule."""

import numpy as np
import pandas as pd

LEARNED = ("operator", "optimizer")


def summary(ev: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per (cell, n_a, init, method), over pairs: share with err_x < 0.05, median err_x (95% bootstrap CIs over
    pairs), median err_y and pred_rmse, and the share of failures (err_x ≥ 0.05) whose fit explains A's points at
    least as well as the truth does (e_fit ≤ e_true: the data cannot tell them apart)."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in ev.groupby(["cell", "n_a", "init", "method"]):
        cell, n, init, m = tuple(key)  # pyright: ignore[reportArgumentType]
        e = g.err_x.to_numpy()
        idx = rng.integers(0, len(e), (n_boot, len(e)))
        share_b, med_b = (e[idx] < 0.05).mean(1), np.median(e[idx], 1)
        fail = g.loc[g.err_x >= 0.05]
        rows.append(
            dict(cell=cell, n_a=n, init=init, method=m, share=(e < 0.05).mean(),
                 share_lo=np.percentile(share_b, 2.5), share_hi=np.percentile(share_b, 97.5),
                 med_err_x=np.median(e), med_lo=np.percentile(med_b, 2.5), med_hi=np.percentile(med_b, 97.5),
                 med_err_y=g.err_y.median(), med_pred=g.pred_rmse.median(),
                 fail_consistent=(fail.e_fit <= fail.e_true).mean() if len(fail) else np.nan, n_pairs=len(g))
        )  # fmt: skip
    return pd.DataFrame(rows)


def verdict(sm: pd.DataFrame) -> dict[str, object]:
    """The rule of notebook 07 (C10, n_A = 10, random start): USABLE / PARTIAL / NOT USABLE for the best learned
    method; whether it beats gd_best8 (share +0.10 at n_A = 10 and 20); the C00 no-harm sanity check."""
    r = sm.loc[(sm.cell == "C10") & (sm.init == "random")].set_index(["method", "n_a"])
    best = max((m for m in LEARNED if (m, 10) in r.index), key=lambda m: r.loc[(m, 10), "share"])
    b, st = r.loc[(best, 10)], r.loc[("start", 10)]
    if b.share >= 0.50 and b.med_err_x <= 0.05:
        v = "USABLE"
    elif b.share_lo > st.share_hi and b.med_hi < st.med_lo:
        v = "PARTIAL"
    else:
        v = "NOT USABLE"
    beats = {
        m: bool(all(r.loc[(m, n), "share"] >= r.loc[("gd_best8", n), "share"] + 0.10 for n in (10, 20)))
        for m in LEARNED
        if (m, 10) in r.index
    }
    c00 = sm.loc[(sm.cell == "C00") & (sm.init == "identity") & (sm.n_a == 10)].set_index("method").med_err_x
    no_harm = {m: bool(c00[m] <= 0.02) for m in LEARNED if m in c00.index}
    return {"verdict": v, "best": best, "share": round(b.share, 3), "median_err_x": round(b.med_err_x, 4),
            "learned_beats_gd": beats, "c00_identity_no_harm": no_harm}  # fmt: skip
