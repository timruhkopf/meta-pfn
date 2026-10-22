"""E0 headroom: what is a dense related task B worth for predicting a scarce task A, to a Bayes learner?

One member = one cell (d, kind, rho). For every pair and n_A in the grid, the four predictors of
`oracle.py`, scored by the mean NLL of A's held-out queries. A duplicate run with an independent
hypothesis set on the first `n_mc_pairs` pairs measures the Bayes oracle's Monte-Carlo error.

Tables: `nll` (pair, seed, n_a, predictor, nll), `posterior` (pair, n_a, mass per hypothesis kind,
ESS of the window and warp particles), `mc` (pair, n_a, bayes NLL of both hypothesis sets).
"""

import time

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pfn_repro.checkpoints import load

from ..evaluation import metric_names as M
from ..oracle import PREDICTORS, evaluate_pair, hypotheses
from ..pairs import KINDS, WarpFamily, draw_pairs
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def headroom(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    if t.kind not in KINDS:
        raise ValueError(f"task.kind must be one of {KINDS}, got {t.kind}")
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    family = WarpFamily(**OmegaConf.to_container(t.family))
    pfn = load(t.pfn)[0].eval()
    k = int(t.k_by_d[t.d]) if t.d in t.k_by_d else int(t.k)
    mass = dict(OmegaConf.to_container(t.hypothesis_mass))
    grid = list(t.n_a_grid)

    t0 = time.time()
    pairs = draw_pairs(t.d, t.kind, t.split, t.n_pairs, t.n_b, max(grid), t.n_q, family, t.rho)
    hyp = hypotheses(t.d, family, k, mass, torch.Generator().manual_seed(cfg.seed))
    hyp_mc = hypotheses(t.d, family, k, mass, torch.Generator().manual_seed(cfg.seed + 1))
    nll, post, mc = [], [], []
    for j in range(pairs.P):
        r = evaluate_pair(pfn, pairs, j, hyp, grid, t.chunk)
        for a, n in enumerate(grid):
            for b, name in enumerate(PREDICTORS):
                nll.append((j, pairs.seeds[j], n, name, r["nll"][a, b]))
            post.append((j, pairs.seeds[j], n, *r["post"][a], *r["ess"][a]))
        if j < t.n_mc_pairs:
            r2 = evaluate_pair(pfn, pairs, j, hyp_mc, grid, t.chunk)
            for a, n in enumerate(grid):
                mc.append((j, pairs.seeds[j], n, r["nll"][a, 3], r2["nll"][a, 3]))
        if (j + 1) % 10 == 0:
            run.log_metric(M.PAIRS_DONE, float(j + 1), j + 1)
    run.log_summary(**{M.TIME_PREFIX + "pairs_s": time.time() - t0})

    nll_df = pd.DataFrame(nll, columns=["pair", "seed", "n_a", "predictor", "nll"])
    run.log_table("nll", nll_df)
    run.log_table(
        "posterior",
        pd.DataFrame(
            post,
            columns=[
                "pair",
                "seed",
                "n_a",
                "p_unrelated",
                "p_identity",
                "p_window",
                "p_warp",
                "ess_window",
                "ess_warp",
            ],
        ),
    )
    if mc:
        run.log_table("mc", pd.DataFrame(mc, columns=["pair", "seed", "n_a", "bayes_1", "bayes_2"]))
    w = nll_df.pivot_table(index=["pair", "n_a"], columns="predictor", values="nll").reset_index()
    for n in grid:
        g = w.loc[w.n_a == n]
        run.log_summary(
            **{
                f"{M.GAIN_BAYES}/n{n}": float((g.lower - g.bayes).mean()),
                f"{M.GAIN_ORACLE}/n{n}": float((g.lower - g.oracle_T).mean()),
            }
        )
    g = w.loc[w.n_a == grid[min(2, len(grid) - 1)]]
    return float((g.lower - g.bayes).mean())
