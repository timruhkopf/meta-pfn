"""Training and evaluation of the A→B model for S3/S4 (issue #18, notebook 09).

Training: fresh pairs every step (strata `identity` : `warp` = TRAIN_P), n_A ~ U{0..n_a_max}; loss = mean over steps
r = 0..R of the mixture NLL on A's queries (deep supervision) + `sup_w` × a privileged term (mean over r):
- `loc` (only valid with h = id): winner-takes-all over particles of the squared location error of T_k vs the true T
  on A's train and query points, (1 - δ) min_k + δ mean_k;
- `func`: the same winner-takes-all over a discretized KL(p* || p_k) on a y-grid at A's queries, p* the oracle particle
  (true T and h) — the composite, invariant to the T/h trade-off;
- `none`.
Evaluation: fixed pairs per stratum (identity / warp / unrelated), n_A ∈ N_A_EVAL (nested), per pair:
- model NLL at every step r (mixture), best single particle (min_k NLL: coverage), top-weight particle;
- location error (RMSE over A's queries) of the top-weight particle and min over k; for learned h, the composite
  discretized KL to the oracle particle;
- references: `a_only` (frozen PFN on A), `oracle` (B at T(x*), true h), `teacher_b` (PFN(B ∪ A_inB) at T(x*), true h),
  `is_bayes` (importance sampling over H prior draws of (T, h), pseudo-likelihood weights; the S1 analogue) and its ESS.
"""

import math
import time

import numpy as np
import pandas as pd
import torch

from ...oracle import logp
from .model import A2B, FrozenQB, mix_eps, mixture_lp
from .prior import STRATA, Inst, PairPrior, PriorCfg, apply_t, h_inv_logjac

TRAIN_P = {"identity": 0.2, "warp": 0.8}
N_A_EVAL = (0, 1, 2, 5, 10)
DELTA = 0.05  # winner-takes-all: weight on the non-winning particles
Y_GRID = 64


def _lr(opt, step: int, steps: int, lr: float, warm: int = 300):
    for pg in opt.param_groups:
        pg["lr"] = lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))


def wta(err: torch.Tensor) -> torch.Tensor:
    """err [B, K] -> mean over B of (1 - δ) min_k + δ mean_k."""
    K = err.shape[1]
    if K == 1:
        return err.mean()
    return ((1 - DELTA) * err.min(1).values + DELTA * err.mean(1)).mean()


def oracle_lp(qb: FrozenQB, kv, nf, inst: Inst, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log density of y at A's x under the oracle particle (true T, h): [B, M]."""
    lg, _ = qb.query(kv, nf, apply_t(x, inst.t_al, inst.t_be))
    u, lj = h_inv_logjac(y, inst.h_kind, inst.h_par)
    return mix_eps(qb.logpdf(lg, u) + lj)


def func_kl(model: A2B, kv, nf, inst: Inst, o: dict) -> torch.Tensor:
    """Discretized KL(p* || p_k) on a y-grid at A's queries, mean over queries: [B, K]."""
    B, K, Q = o["lq"].shape
    yg = (torch.arange(Y_GRID, device=inst.xq.device) + 0.5) / Y_GRID
    ys = yg.expand(B, Q, -1).reshape(B, -1)  # [B, Q * Y]
    with torch.no_grad():
        lg, _ = model.qb.query(kv, nf, apply_t(inst.xq, inst.t_al, inst.t_be))  # [B, Q, G]
        u, lj = h_inv_logjac(ys, inst.h_kind, inst.h_par)
        ls = mix_eps(model.qb.logpdf_grid(lg, u.reshape(B, Q, Y_GRID)) + lj.reshape(B, Q, Y_GRID))
        ps = ls.log_softmax(-1)  # [B, Q, Y]
    kvK = [(k.repeat_interleave(K, 0), v.repeat_interleave(K, 0)) for k, v in kv]
    lgk, _ = model.qb.query(kvK, nf.repeat_interleave(K, 0), o["tq"].reshape(B * K, Q, -1))
    uk, ljk = model.h_inv(o["P"], ys)  # [B, K, Q * Y]
    lk = mix_eps(
        model.qb.logpdf_grid(lgk.reshape(B, K, Q, -1), uk.reshape(B, K, Q, Y_GRID)) + ljk.reshape(B, K, Q, Y_GRID)
    )
    pk = lk.log_softmax(-1)
    return (ps.exp()[:, None] * (ps[:, None] - pk)).sum(-1).mean(-1)


def train(model: A2B, cfg: PriorCfg, steps: int, device: str, *, seed: int, batch: int, n_b: int, n_q: int,
          n_a_max: int, lr: float, sup: str, sup_w: float, every: int, metrics=None,
          log=print) -> pd.DataFrame:  # fmt: skip
    prior = PairPrior(cfg, batch, seed, device)
    val = PairPrior(cfg, 64, seed + 99_991, device).sample("warp", 5, n_q, n_b)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    hist, t0 = [], time.time()
    for step in range(steps):
        _lr(opt, step, steps, lr)
        stratum = str(rng.choice(list(TRAIN_P), p=list(TRAIN_P.values())))
        n = int(rng.integers(0, n_a_max + 1))
        inst = prior.sample(stratum, n, n_q, n_b)
        model.train()
        outs = model(inst.xb, inst.yb, inst.xa, inst.ya, inst.xq, inst.yq)
        nll = torch.stack([-mixture_lp(o).mean() for o in outs]).mean()
        loss_sup = torch.zeros((), device=device)
        if sup == "loc":
            true = [apply_t(x, inst.t_al, inst.t_be)[:, None] for x in (inst.xa, inst.xq)]
            loss_sup = torch.stack([
                wta(torch.cat([((o["ta"] - true[0]) ** 2).sum(-1), ((o["tq"] - true[1]) ** 2).sum(-1)], 2).mean(-1))
                for o in outs
            ]).mean()  # fmt: skip
        elif sup == "func":
            kv, nf, _ = model.qb.encode(inst.xb, inst.yb)
            loss_sup = torch.stack([wta(func_kl(model, kv, nf, inst, o)) for o in outs]).mean()
        loss = nll + sup_w * loss_sup
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % every == 0 or step == steps - 1:
            v = validate(model, val)
            row = {"step": step, "loss": float(loss), "nll": float(nll), "sup": float(loss_sup), "grad_norm": float(gn),
                   "n_a": n, "stratum": stratum, "s": time.time() - t0, **v}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({f"train/{k}": float(row[k]) for k in ("loss", "nll", "sup", "grad_norm")} |
                        {f"val/{k}": float(x) for k, x in v.items()}, step=step)  # fmt: skip
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
    return pd.DataFrame(hist)


@torch.no_grad()
def validate(model: A2B, inst: Inst) -> dict:
    model.eval()
    o = model(inst.xb, inst.yb, inst.xa, inst.ya, inst.xq, inst.yq)[-1]
    true = apply_t(inst.xq, inst.t_al, inst.t_be)
    top = o["logw"].argmax(1)
    tq = o["tq"][torch.arange(len(top)), top]
    return {
        "val_nll_n5": float(-mixture_lp(o).mean()),
        "val_rmse_top_n5": float(((tq - true) ** 2).sum(-1).mean().sqrt()),
    }


def eval_set(cfg: PriorCfg, stratum: str, n_pairs: int, n_q: int, n_b: int, seed: int, device: str) -> Inst:
    """Fixed pairs per stratum (nested in n_A: the first max(N_A_EVAL) A points are the pool)."""
    return PairPrior(cfg, n_pairs, seed + 1009 * STRATA.index(stratum), device).sample(stratum, max(N_A_EVAL), n_q, n_b)


def _sub(inst: Inst, sl: slice) -> Inst:
    f = lambda t: t[sl]  # noqa: E731
    return Inst(f(inst.xb), f(inst.yb), f(inst.xa), f(inst.ya), f(inst.xq), f(inst.yq), f(inst.t_al), f(inst.t_be),
                f(inst.h_par), inst.h_kind, f(inst.related))  # fmt: skip


@torch.no_grad()
def references(qb: FrozenQB, pfn, cfg: PriorCfg, inst: Inst, n_hyp: int, seed: int, chunk: int = 64) -> list[dict]:
    """Per pair and n_A: a_only, oracle, teacher_b, is_bayes NLL, and is_bayes' ESS."""
    B = inst.xq.shape[0]
    dev = str(inst.xq.device)
    kv, nf, _ = qb.encode(inst.xb, inst.yb)
    lo = oracle_lp(qb, kv, nf, inst, inst.xq, inst.yq)  # [B, Q]
    # importance sampling over prior draws of (T, h), shared across pairs
    g = PairPrior(cfg, n_hyp, seed, dev)
    hyp = g.sample("warp", 0, 1, 1)
    xs = torch.cat([inst.xa, inst.xq], 1)
    ys = torch.cat([inst.ya, inst.yq], 1)
    n_max = inst.xa.shape[1]
    rows = []
    for j in range(B):
        lp = []
        for s in range(0, n_hyp, chunk):
            sl = slice(s, s + chunk)
            H = min(chunk, n_hyp - s)
            t = apply_t(xs[j : j + 1].expand(H, -1, -1), hyp.t_al[sl], hyp.t_be[sl])
            lg, _ = qb.query([(k[j : j + 1].expand(H, -1, -1), v[j : j + 1].expand(H, -1, -1)) for k, v in kv],
                             nf[:1].expand(H), t)  # fmt: skip
            u, lj = h_inv_logjac(ys[j : j + 1].expand(H, -1), cfg.h_kind, hyp.h_par[sl])
            lp.append(mix_eps(qb.logpdf(lg, u) + lj))
        lp = torch.cat(lp)  # [H, n_max + Q]
        for n in N_A_EVAL:
            lw = lp[:, :n].sum(1)
            w = lw - torch.logsumexp(lw, 0)
            is_q = torch.logsumexp(w[:, None] + lp[:, n_max:], 0)
            ess = float(1 / w.exp().pow(2).sum())
            xa, ya = inst.xa[j : j + 1, :n], inst.ya[j : j + 1, :n]
            a = logp(pfn, xa, ya, torch.ones_like(ya, dtype=torch.bool), inst.xq[j : j + 1], inst.yq[j : j + 1], 1)[0]
            ta = apply_t(xa, inst.t_al[j : j + 1], inst.t_be[j : j + 1])
            ua, _ = h_inv_logjac(ya, inst.h_kind, inst.h_par[j : j + 1])
            kvt, nft, _ = qb.encode(torch.cat([inst.xb[j : j + 1], ta], 1), torch.cat([inst.yb[j : j + 1], ua], 1))
            lgt, _ = qb.query(kvt, nft, apply_t(inst.xq[j : j + 1], inst.t_al[j : j + 1], inst.t_be[j : j + 1]))
            uq, ljq = h_inv_logjac(inst.yq[j : j + 1], inst.h_kind, inst.h_par[j : j + 1])
            tb = mix_eps(qb.logpdf(lgt, uq) + ljq)[0]
            for name, v in (("a_only", a), ("oracle", lo[j]), ("teacher_b", tb), ("is_bayes", is_q)):
                rows.append({"pair": j, "n_a": n, "method": name, "nll": -float(v.mean())})
            rows.append({"pair": j, "n_a": n, "method": "is_ess", "nll": ess})
    return rows


@torch.no_grad()
def evaluate(model: A2B, pfn, cfg: PriorCfg, *, n_pairs: int, n_q: int, n_b: int, seed: int, device: str,
             n_hyp: int, refs: bool = True, chunk: int = 32) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """-> (nll rows: stratum, pair, n_a, method, r, nll; registration rows: stratum, pair, n_a, r, rmse_top, rmse_best,
    w_top, kl_top, kl_best)."""
    model.eval()
    nll_rows, reg_rows = [], []
    for stratum in STRATA:
        inst = eval_set(cfg, stratum, n_pairs, n_q, n_b, seed, device)
        if refs:
            nll_rows += [
                {"stratum": stratum, "r": -1} | r for r in references(model.qb, pfn, cfg, inst, n_hyp, seed + 5)
            ]
        for s in range(0, n_pairs, chunk):
            sub = _sub(inst, slice(s, s + chunk))
            Bc = sub.xq.shape[0]
            true_q = apply_t(sub.xq, sub.t_al, sub.t_be)[:, None]
            kv, nf, _ = model.qb.encode(sub.xb, sub.yb)
            for n in N_A_EVAL:
                outs = model(sub.xb, sub.yb, sub.xa[:, :n], sub.ya[:, :n], sub.xq, sub.yq)
                for r, o in enumerate(outs):
                    mix = -mixture_lp(o).mean(-1)
                    per_k = -o["lq"].mean(-1)  # [B, K]
                    top = o["logw"].argmax(1)
                    ar = torch.arange(Bc, device=device)
                    se = ((o["tq"] - true_q) ** 2).sum(-1).mean(-1)  # [B, K]
                    kl = func_kl(model, kv, nf, sub, o) if cfg.h_kind != "id" else None
                    w = (o["logw"] - torch.logsumexp(o["logw"], -1, keepdim=True)).exp()
                    for i in range(Bc):
                        base = {"stratum": stratum, "pair": s + i, "n_a": n, "r": r}
                        nll_rows += [base | {"method": "model", "nll": float(mix[i])},
                                     base | {"method": "model_best_k", "nll": float(per_k[i].min())},
                                     base | {"method": "model_top", "nll": float(per_k[i, top[i]])}]  # fmt: skip
                        reg_rows.append(base | {
                            "rmse_top": float(se[i, top[i]].sqrt()), "rmse_best": float(se[i].min().sqrt()),
                            "w_top": float(w[i, top[i]]),
                            "kl_top": float(kl[i, top[i]]) if kl is not None else np.nan,
                            "kl_best": float(kl[i].min()) if kl is not None else np.nan,
                        })  # fmt: skip
                    del ar
    return pd.DataFrame(nll_rows), pd.DataFrame(reg_rows)
