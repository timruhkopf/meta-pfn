"""MTPFN training and evaluation on two priors (issue #20, notebook 12).

- arm R (`icm_*`): the paper's robust ICM prior (icm.py); evaluation against exact GP references.
- arm A (`air_*`): the AIR A→B prior (experimental/air/prior.py, ECDF g, Z0, H0): B = one dense source task,
  A = the scarce target; evaluation on pilot 3's evaluation tasks (same seed, J, n_A) so its NLLs join AIR's.
Training follows the paper (§5): AdamW, lr 1e-4, cosine annealing (pfn-repro's cosine_warmup_lr), NLL of the
observed y at the target's query points; batch and steps are config (paper: batch 16, ~50M datasets).
"""

import math
import time

import numpy as np
import pandas as pd
import torch
from pfn_repro.training.trainer import cosine_warmup_lr

from ..air.prior import PriorCfg, Tasks, sample_tasks
from ..air.train import eval_tasks
from ..air.train_rb import N_A_RB
from . import icm as I
from .model import MTPFN

# ---- data adapters: -> (x [P,T,N,d], y [P,T,N], mask [P,T,N], xq [P,Q,d], yq [P,Q], n_features [P]) --------------


def icm_batch(b: I.ICMBatch):
    P, d = b.x.shape[0], b.x.shape[-1]
    return b.x, b.y, b.mask, b.xq, b.yq, torch.full((P,), d, device=b.x.device)


def air_batch(t: Tasks):
    P, n, n_b = t.xa.shape[0], t.xa.shape[1], t.xb.shape[1]
    N = max(n, n_b)
    x = t.xa.new_zeros(P, 2, N)
    y = t.xa.new_zeros(P, 2, N)
    mask = torch.zeros(P, 2, N, dtype=torch.bool, device=t.xa.device)
    x[:, 0, :n], y[:, 0, :n], mask[:, 0, :n] = t.xa, t.ya, True
    x[:, 1, :n_b], y[:, 1, :n_b], mask[:, 1, :n_b] = t.xb, t.yb, True
    return x[..., None], y, mask, t.xq[..., None], t.yq, torch.ones(P, device=t.xa.device, dtype=torch.long)


def icm_sampler(cfg, device):
    """cfg: t_range, d_max, n_range, p, ard, q. Batch-uniform T, d, n_tot (as pfn-repro's BNNPrior draws d)."""

    def draw(rng: np.random.Generator, g: torch.Generator, P: int):
        T = int(rng.integers(cfg.t_range[0], cfg.t_range[1] + 1))
        d = int(rng.integers(1, cfg.d_max + 1))
        n = int(rng.integers(cfg.n_range[0], cfg.n_range[1] + 1))
        return icm_batch(I.sample_icm(P, T, d, cfg.q, g, device, p=cfg.p, n_tot=n, ard=cfg.ard))

    return draw


def air_sampler(cfg, device):
    """cfg: j_mix, g_norm. A: n_A ~ U{1..10}; B: 128 points (PriorCfg defaults); Z0, H0."""

    def draw(rng: np.random.Generator, g: torch.Generator, P: int):
        J, n = int(rng.choice(list(cfg.j_mix))), int(rng.integers(1, 11))
        return air_batch(sample_tasks(PriorCfg(J=J, g_norm=cfg.g_norm), P, n, g, device))

    return draw


def nll_per_task(model: MTPFN, batch, chunk: int = 64) -> torch.Tensor:
    x, y, mask, xq, yq, nf = batch
    out = []
    for c in range(0, x.shape[0], chunk):
        s = slice(c, c + chunk)
        o = model(x[s], y[s], mask[s], xq[s], nf[s])
        out.append(-model.log_density(o, yq[s]).mean(-1))
    return torch.cat(out)


def train(model: MTPFN, draw, steps: int, device, *, seed: int, batch: int, lr: float, warmup: int, clip: float,
          every: int, val, metrics=None, log=print) -> pd.DataFrame:  # fmt: skip
    """val: a fixed batch (tuple as from the adapters)."""
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    hist, t0 = [], time.time()
    for step in range(steps + 1):
        f = cosine_warmup_lr(step, warmup, steps)
        for pg in opt.param_groups:
            pg["lr"] = lr * f
        x, y, mask, xq, yq, nf = draw(rng, g, batch)
        model.train()
        loss = -model.log_density(model(x, y, mask, xq, nf), yq).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                v = float(nll_per_task(model, val).mean())
            row = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": lr * f, "val_nll": v,
                   "s": time.time() - t0}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({"train/loss": row["loss"], "train/grad_norm": row["grad_norm"], "train/lr": row["lr"],
                         "val/nll": v}, step=step)  # fmt: skip
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
    return pd.DataFrame(hist)


# ---- evaluation --------------------------------------------------------------------------------------------------


def icm_eval_batch(d: int, n_t: int, u: int, cfg, device) -> I.ICMBatch:
    """The fixed evaluation datasets of one cell (shared by every model and the references)."""
    g = torch.Generator(device=device).manual_seed(cfg.eval_seed + 1000 * d + 37 * n_t + u)
    return I.sample_icm(cfg.n_eval, cfg.t_eval, d, cfg.q_eval, g, device, counts=[n_t] + [cfg.n_src] * (cfg.t_eval - 1),
                        force_unrel=u, ard=cfg.ard)  # fmt: skip


@torch.no_grad()
def icm_evaluate(models: dict[str, MTPFN], cfg, device, log=print) -> pd.DataFrame:
    rows = []
    for d in cfg.eval_d:
        for n_t in cfg.eval_n_target:
            for u in range(cfg.t_eval):
                b = icm_eval_batch(d, n_t, u, cfg, device)
                for name, m in models.items():
                    m.eval()
                    nll = nll_per_task(m, icm_batch(b))
                    rows += [{"d": d, "n_t": n_t, "u": u, "task": i, "method": name, "nll": float(v)}
                             for i, v in enumerate(nll)]  # fmt: skip
                log(f"d={d} n_t={n_t} u={u} done")
    return pd.DataFrame(rows)


@torch.no_grad()
def icm_references(cfg, device, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ess_rows = [], []
    for d in cfg.eval_d:
        for n_t in cfg.eval_n_target:
            for u in range(cfg.t_eval):
                t0 = time.time()
                b = icm_eval_batch(d, n_t, u, cfg, device)
                g = torch.Generator(device=device).manual_seed(cfg.eval_seed + 7 + 1000 * d + 37 * n_t + u)
                for i in range(b.x.shape[0]):
                    lp, ess = I.references(b, i, cfg.p_ref, cfg.k_ref, g)
                    rows += [{"d": d, "n_t": n_t, "u": u, "task": i, "method": k, "nll": -float(v.mean())}
                             for k, v in lp.items()]  # fmt: skip
                    ess_rows += [{"d": d, "n_t": n_t, "u": u, "task": i, "metric": k, "value": v}
                                 for k, v in ess.items()]  # fmt: skip
                log(f"refs d={d} n_t={n_t} u={u} done in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), pd.DataFrame(ess_rows)


@torch.no_grad()
def air_evaluate(model: MTPFN, cfg, device, log=print) -> pd.DataFrame:
    """Pilot 3's evaluation tasks (eval_tasks(J, n_eval, eval_seed, g_norm)); rows join AIR's `nll` table
    (J, sig = -1, n_a, task, method = 'mtpfn', nll)."""
    model.eval()
    rows = []
    for J in cfg.eval_j:
        base = eval_tasks(J, cfg.n_eval, cfg.eval_seed, device, g_norm=cfg.g_norm)
        for n in N_A_RB:
            nll = nll_per_task(model, air_batch(base.head(n)))
            rows += [{"J": J, "sig": -1.0, "n_a": n, "task": i, "method": "mtpfn", "nll": float(v)}
                     for i, v in enumerate(nll)]  # fmt: skip
        log(f"J={J} done")
    return pd.DataFrame(rows)
