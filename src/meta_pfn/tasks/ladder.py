"""Sanity ladder S0/S1 (issue #16, notebook 08): one member = one (d, N_B). Frozen pfn_large, no training.

Tables: `nll` (per pair, n_A and predictor: mean query NLL), `post` (per pair, n_A and method: posterior over s),
`gd` (gradient descent on s from 0 vs the grid MAP), and 1d/2d examples: `ex_post` (log posterior on the grid),
`ex_gd` (GD trajectories), `ex_curve` (d = 1: f_A, the oracle particle and the A-only PFN), `ex_points`.
"""

import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..evaluation import metric_names as M
from ..experimental import ladder as L
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def ladder(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    pfn = load(t.pfn, device=dev)[0].eval()
    grid = L.shift_grid(t.d, t.s_max, t.grid1, t.grid2).to(dev)
    n_grid = list(t.n_a)
    n_gd = [n for n in n_grid if n > 0]
    tabs: dict[str, list] = {k: [] for k in ("nll", "post", "gd", "ex_post", "ex_gd", "ex_curve", "ex_points")}
    t0, done = time.time(), 0
    for kind in L.KINDS:
        for i in range(t.n_pairs):
            p = L.draw_pair(t.d, kind, t.split, i, t.n_b, max(n_grid), t.n_q, t.s_max)
            key = {"kind": kind, "pair": i, "s_true": p.s.tolist()}
            r = L.evaluate_pair(pfn, p, grid, n_grid, t.chunk)
            tabs["nll"] += [key | row for row in r["nll"]]
            tabs["post"] += [key | {"n_b_in_a": r["n_b_in_a"]} | row for row in r["post"]]
            ex = i < t.n_ex
            s_gd, ll_gd, traj = L.gd_shift(pfn, p, n_gd, t.s_max, t.gd_steps, t.gd_lr, dev, every=10 if ex else 0)
            for j, n in enumerate(n_gd):
                llp = r["ll_pseudo"][:, :n].sum(1)
                m = int(llp.argmax())
                tabs["gd"].append(key | {
                    "n_a": n, "s_gd": s_gd[j].tolist(), "ll_gd": float(ll_gd[j]), "ll_map": float(llp[m]),
                    "err_true": float((s_gd[j] - p.s).norm()), "err_map": float((s_gd[j] - grid[m].cpu()).norm()),
                    "success": bool(ll_gd[j] >= llp[m] - t.gd_tol),
                })  # fmt: skip
            if ex:
                g = grid.cpu().numpy()
                for (meth, n), lw in r["lw"].items():
                    cols = {f"s{k}": g[:, k] for k in range(t.d)}
                    tabs["ex_post"].append(
                        pd.DataFrame({"kind": kind, "pair": i, "method": meth, "n_a": n, **cols, "lw": lw.numpy()})
                    )
                for step, sv in enumerate(traj if traj is not None else []):
                    for j, n in enumerate(n_gd):
                        tabs["ex_gd"].append(key | {"step": step * 10, "n_a": n, "s": sv[j].tolist()})
                pts = [("B", p.xb, p.yb), ("A", p.xa, p.ya), ("Q", p.xq, p.yq)]
                tabs["ex_points"].append(pd.concat([pd.DataFrame({"kind": kind, "pair": i, "set": nm, "y": y.numpy(),
                                                                  **{f"x{k}": x[:, k].numpy() for k in range(t.d)}})
                                                    for nm, x, y in pts]))  # fmt: skip
                if t.d == 1:
                    tabs["ex_curve"].append(L.curves(pfn, p, n_show=5).assign(kind=kind, pair=i))
            done += 1
            if done % 10 == 0:
                print(f"{done} pairs, {time.time() - t0:.0f}s", flush=True)
                run.log_metrics({M.PAIRS_DONE: done}, step=done)
    for k, v in tabs.items():
        if v:
            run.log_table(k, pd.concat(v, ignore_index=True) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v))
    nll = pd.DataFrame(tabs["nll"])
    w = nll[(nll.kind == "shift") & (nll.n_a == 2)].pivot_table(index="pair", columns="predictor", values="nll")
    score = float(np.mean(w.a_only - w.oracle_q))
    run.log_summary(**{M.LADDER_S0_GAIN_N2: score, M.TIME_PREFIX + "total_s": time.time() - t0})
    return score
