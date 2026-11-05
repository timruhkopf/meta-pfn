"""Training and evaluation of the GMM-token PFN (issue #23, notebook 18)."""

import math
import time

import pandas as pd
import torch
from pfn_repro.training.trainer import cosine_warmup_lr

from . import oracle as O
from . import prior as P
from .model import GMMPFN

N_EVAL = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def eval_set(d: int, n_tasks: int, Q: int, seed: int) -> P.Batch:
    """The fixed evaluation tasks of dimension d (CPU-generated, so identical on every machine). Context sets are
    nested: cell n uses the first n tokens (`Batch.head`)."""
    g = torch.Generator().manual_seed(seed + 1000 * d)
    return P.sample(n_tasks, max(N_EVAL), d, Q, g)


@torch.no_grad()
def model_logp(model: GMMPFN, b: P.Batch, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query joint log density and the sum of its per-dimension marginal log densities, [B, Q] each."""
    model.eval()
    j, f = [], []
    for c in range(0, b.x.shape[0], chunk):
        bc = P.Batch(**{k: v[c : c + chunk] for k, v in b.__dict__.items()})
        p = model(bc)
        j.append(model.log_density(p, bc.yq))
        f.append(model.log_marginals(p, bc.yq).sum(-1))
    return torch.cat(j), torch.cat(f)


def train(model: GMMPFN, d: int, steps: int, device, *, seed: int, batch: int, q: int, n_max: int, lr: float,
          warmup: int, clip: float, every: int, metrics=None, log=print) -> pd.DataFrame:  # fmt: skip
    g = torch.Generator(device=device).manual_seed(seed)
    gn_ = torch.Generator().manual_seed(seed + 1)
    val = eval_set(d, 256, q, seed=9999).head(16).to(device)  # validation: own seed, not the evaluation tasks
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    hist, t0 = [], time.time()
    for step in range(steps + 1):
        fct = cosine_warmup_lr(step, warmup, steps)
        for pg in opt.param_groups:
            pg["lr"] = lr * fct
        n = int(torch.randint(0, n_max + 1, (1,), generator=gn_))
        b = P.sample(batch, n, d, q, g, device)
        model.train()
        loss = -model.log_density(model(b), b.yq).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % every == 0 or step == steps:
            v = float(-model_logp(model, val)[0].mean())
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(gnorm), "lr": lr * fct,
                   "val_nll_n16": v, "s": time.time() - t0}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({"train/loss": row["loss"], "train/grad_norm": row["grad_norm"], "val/nll_n16": v}, step=step)
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
    return pd.DataFrame(hist)


@torch.no_grad()
def evaluate(model: GMMPFN, d: int, device, *, n_tasks: int, q: int, seed: int, log=print) -> pd.DataFrame:
    """Rows (d, n, task, m, ell, method, nll): nll = mean over the task's q queries of -log p(u*) (joint, nats).
    Methods: model, model_fact (product of the model's marginals), oracle, oracle_fact, prior."""
    base = eval_set(d, n_tasks, q, seed)
    rows = []
    for n in N_EVAL:
        b = base.head(n)
        ref = O.references(b)
        mj, mf = model_logp(model, b.to(device))
        lp = {"model": mj.cpu().double(), "model_fact": mf.cpu().double(), **ref}
        for k, v in lp.items():
            nll = -v.mean(-1)
            rows += [{"d": d, "n": n, "task": i, "m": int(b.m[i]), "ell": float(b.ell[i]), "method": k,
                      "nll": float(nll[i])} for i in range(len(nll))]  # fmt: skip
        log(f"d={d} n={n}: " + " ".join(f"{k}={-float(v.mean()):.3f}" for k, v in lp.items()))
    return pd.DataFrame(rows)
