"""GW-only analysis (notebook 06): landmark-error summaries with pair-bootstrap CIs, and the pre-registered rule."""

import numpy as np
import pandas as pd


def summary(ev: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per (cell, n_a, method), over covered A points: share with err_x < 0.05, median err_x, mean π-mass on the
    truth; 95% CIs by resampling pairs."""
    rng = np.random.default_rng(seed)
    rows = []
    for (cell, n, m), g in ev.loc[ev.covered].groupby(["cell", "n_a", "method"]):
        pairs = g.pair.unique()
        by_pair = {p: d for p, d in g.groupby("pair")}
        stats = []
        for _ in range(n_boot):
            s = pd.concat([by_pair[p] for p in rng.choice(pairs, len(pairs))])
            stats.append(((s.err_x < 0.05).mean(), s.err_x.median()))
        lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
        rows.append(
            dict(cell=cell, n_a=n, method=m, share=(g.err_x < 0.05).mean(), share_lo=lo[0], share_hi=hi[0],
                 med_err_x=g.err_x.median(), med_lo=lo[1], med_hi=hi[1], mass=g.mass.mean(), n_points=len(g),
                 covered_frac=len(g) / len(ev.loc[(ev.cell == cell) & (ev.n_a == n) & (ev.method == m)]))
        )  # fmt: skip
    return pd.DataFrame(rows)


def verdict(sm: pd.DataFrame, solvers=("peyre", "unrolled", "gru")) -> dict[str, object]:
    """The rule of notebook 06 (C10, n_A = 5): USABLE / PARTIAL / NOT USABLE for the best solver, and whether a
    learned solver beats Peyré (share +0.10 at n_A = 5 and 10)."""
    c = sm.loc[(sm.cell == "C10") & (sm.n_a == 5)].set_index("method")
    best = max((s for s in solvers if s in c.index), key=lambda s: c.loc[s, "share"])
    b, ident = c.loc[best], c.loc["identity"]
    if b.share >= 0.70 and b.med_err_x <= 0.05:
        v = "USABLE"
    elif b.share_lo > ident.share_hi and b.med_hi < ident.med_lo:
        v = "PARTIAL"
    else:
        v = "NOT USABLE"
    c10 = sm.loc[sm.cell == "C10"].set_index(["method", "n_a"]).share
    learned = {
        s: bool(all(c10[(s, n)] >= c10[("peyre", n)] + 0.10 for n in (5, 10)))
        for s in ("unrolled", "gru")
        if (s, 5) in c10
    }
    return {"verdict": v, "best_solver": best, "share": round(b.share, 3), "median_err_x": round(b.med_err_x, 4),
            "learned_beats_peyre": learned}  # fmt: skip
