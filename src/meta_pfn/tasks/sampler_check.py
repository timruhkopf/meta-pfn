"""E0 follow-up: is the Bayes learner's low share G on warp pairs the sampler or the data?

On the first `n_pairs` pilot pairs of one cell (σ = 0), the E0 predictors are recomputed (same hypothesis
set as the pilot, so `bayes_is` reproduces it), and the Bayes learner is recomputed with SMC (`smc.py`) for
the window and warp families. Within the pair's true kind, the importance-sampling predictive (`kind_is`)
and the SMC one (`kind_smc`, twice with independent seeds) are compared with the known-T `oracle_T`:
the gap kind_is → kind_smc is the sampler, the gap kind_smc → oracle_T is the price of inferring T.
`bayes_smc` mixes over the four kinds with the SMC family evidences.

Tables: `nll` (pair, seed, n_a, predictor, nll), `posterior` (kind masses under bayes_smc, posterior RMS
T error of both SMC runs), `smc` (per SMC run: family, rep, seconds, acceptance, tempering stages).
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pfn_repro.checkpoints import load

from ..evaluation import metric_names as M
from ..oracle import KINDS4, evaluate_pair, hypotheses
from ..pairs import WarpFamily, draw_pairs
from ..smc import Data, smc
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def sampler_check(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    if t.kind not in ("window", "warp"):
        raise ValueError(f"task.kind must be window or warp, got {t.kind}")
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    family = WarpFamily(**OmegaConf.to_container(t.family))
    pfn = load(t.pfn)[0].eval()
    k_is = int(t.k_by_d[t.d]) if t.d in t.k_by_d else int(t.k)
    mass = dict(OmegaConf.to_container(t.hypothesis_mass))
    log_mass = torch.tensor([np.log(mass[k]) for k in KINDS4])
    grid = list(t.n_a_grid)
    other = "warp" if t.kind == "window" else "window"
    ki = KINDS4.index(t.kind)

    pairs = draw_pairs(t.d, t.kind, t.split, t.n_pairs, t.n_b, max(grid), t.n_q, family, 0.0)
    hyp = hypotheses(t.d, family, k_is, mass, torch.Generator().manual_seed(cfg.seed))
    nll, post, runs = [], [], []
    for j in range(pairs.P):
        s = pairs.seeds[j]
        r = evaluate_pair(pfn, pairs, j, hyp, grid, t.chunk)
        data = Data(pairs.XB[j], pairs.YB[j], pairs.XA[j], pairs.YA[j], pairs.XQ[j], pairs.YQ[j])
        fits = {}
        for fam, rep in ((t.kind, 0), (t.kind, 1), (other, 0)):
            t0 = time.time()
            g = torch.Generator().manual_seed(s * 10 + rep + 5 * (fam != t.kind))
            truth = pairs.theta[j] if fam == t.kind else None
            fits[fam, rep] = res = smc(pfn, data, fam, family, int(t.k_smc), grid, g, truth, t.n_moves, chunk=t.chunk)
            runs.append((j, s, fam, rep, time.time() - t0, res.accept, int(res.stages[-1])))
        for a, n in enumerate(grid):
            lz = torch.from_numpy(r["kind_logz"][a]).float()
            lq = torch.from_numpy(r["kind_logq"][a]).float()  # [4, Q]
            lz[KINDS4.index(t.kind)] = float(fits[t.kind, 0].logz[a])
            lz[KINDS4.index(other)] = float(fits[other, 0].logz[a])
            lq[KINDS4.index(t.kind)] = torch.from_numpy(fits[t.kind, 0].logq[a])
            lq[KINDS4.index(other)] = torch.from_numpy(fits[other, 0].logq[a])
            w = log_mass + lz
            w = w - torch.logsumexp(w, 0)
            mix = torch.logsumexp(w[:, None] + lq, 0)
            for name, v in (
                ("lower", r["nll"][a, 0]),
                ("oracle_T", r["nll"][a, 2]),
                ("bayes_is", r["nll"][a, 3]),
                ("bayes_smc", -float(mix.mean())),
                ("kind_is", -float(r["kind_logq"][a, ki].mean())),
                ("kind_smc", -float(fits[t.kind, 0].logq[a].mean())),
                ("kind_smc_rep", -float(fits[t.kind, 1].logq[a].mean())),
            ):
                nll.append((j, s, n, name, float(v)))
            post.append((j, s, n, *w.exp().tolist(), fits[t.kind, 0].t_err[a], fits[t.kind, 1].t_err[a]))
        run.log_metric(M.PAIRS_DONE, float(j + 1), j + 1)

    nll_df = pd.DataFrame(nll, columns=["pair", "seed", "n_a", "predictor", "nll"])
    run.log_table("nll", nll_df)
    cols = ["pair", "seed", "n_a", *(f"p_{k}" for k in KINDS4), "t_err", "t_err_rep"]
    run.log_table("posterior", pd.DataFrame(post, columns=cols))
    run.log_table("smc", pd.DataFrame(runs, columns=["pair", "seed", "family", "rep", "seconds", "accept", "stages"]))
    w = nll_df.pivot_table(index=["pair", "n_a"], columns="predictor", values="nll").reset_index()
    for n in grid:
        g = w.loc[w.n_a == n]
        run.log_summary(**{f"{M.GAIN_KIND_SMC}/n{n}": float((g.lower - g.kind_smc).mean())})
    g = w.loc[w.n_a == grid[min(2, len(grid) - 1)]]
    return float((g.lower - g.kind_smc).mean())
