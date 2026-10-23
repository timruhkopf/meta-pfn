"""E0b outmap: what is B worth for A when A differs from B only by a monotone output map g (x shared)?

One member = one cell (d, kind). n_A and n_B scale with d (`n_a_mult`, `n_b_by_d`). For every pair and n_A
in the grid, the five predictors of `experimental/outmap.py`, scored by the mean NLL of A's held-out
queries. A duplicate run with an independently offset g-grid on the first `n_mc_pairs` pairs measures the
Bayes learner's Monte-Carlo (quadrature) error.

Tables: `nll` (pair, seed, n_a, predictor, nll), `posterior` (pair, n_a, mass of identity / outmap /
unrelated, ESS of the outmap grid, posterior RMS error of g), `mc` (pair, n_a, bayes NLL of both grids).
"""

import time

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pfn_repro.checkpoints import load

from ..evaluation import metric_names as M
from ..experimental.outmap import KINDS, PREDICTORS, OutputFamily, draw_pairs, evaluate_pair
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def outmap(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    if t.kind not in KINDS:
        raise ValueError(f"task.kind must be one of {KINDS}, got {t.kind}")
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    family = OutputFamily(**OmegaConf.to_container(t.family))
    pfn = load(t.pfn)[0].eval()
    mass = {k: float(v) for k, v in OmegaConf.to_container(t.hypothesis_mass).items()}
    grid = [m * t.d for m in t.n_a_mult]
    n_b = int(t.n_b_by_d[t.d])
    g = torch.Generator().manual_seed(cfg.seed)
    grid_ab = family.grid(t.grid, 0.5)
    grid_mc = family.grid(t.grid, float(torch.rand(1, generator=g)))

    t0 = time.time()
    pairs = draw_pairs(t.d, t.kind, t.split, t.n_pairs, n_b, max(grid), t.n_q, family, t.p_map)
    nll, post, mc = [], [], []
    for j in range(pairs.P):
        r = evaluate_pair(pfn, pairs, j, grid_ab, mass, t.p_map, grid, t.chunk)
        for a, n in enumerate(grid):
            for b, name in enumerate(PREDICTORS):
                nll.append((j, pairs.seeds[j], n, name, r["nll"][a, b]))
            post.append((j, pairs.seeds[j], n, *r["post"][a], r["ess"][a], r["g_err"][a]))
        if j < t.n_mc_pairs:
            r2 = evaluate_pair(pfn, pairs, j, grid_mc, mass, t.p_map, grid, t.chunk)
            for a, n in enumerate(grid):
                mc.append((j, pairs.seeds[j], n, r["nll"][a, 4], r2["nll"][a, 4]))
        run.log_metric(M.PAIRS_DONE, float(j + 1), j + 1)
    run.log_summary(**{M.TIME_PREFIX + "pairs_s": time.time() - t0})

    nll_df = pd.DataFrame(nll, columns=["pair", "seed", "n_a", "predictor", "nll"])
    run.log_table("nll", nll_df)
    cols = ["pair", "seed", "n_a", "p_identity", "p_outmap", "p_unrelated", "ess_outmap", "g_err"]
    run.log_table("posterior", pd.DataFrame(post, columns=cols))
    if mc:
        run.log_table("mc", pd.DataFrame(mc, columns=["pair", "seed", "n_a", "bayes_1", "bayes_2"]))
    w = nll_df.pivot_table(index=["pair", "n_a"], columns="predictor", values="nll").reset_index()
    for n in grid:
        h = w.loc[w.n_a == n]
        run.log_summary(
            **{
                f"{M.GAIN_BAYES}/n{n}": float((h.lower_bayes - h.bayes).mean()),
                f"{M.GAIN_ORACLE_G}/n{n}": float((h.lower_bayes - h.oracle_g).mean()),
            }
        )
    h = w.loc[w.n_a == grid[min(2, len(grid) - 1)]]
    return float((h.lower_bayes - h.bayes).mean())
