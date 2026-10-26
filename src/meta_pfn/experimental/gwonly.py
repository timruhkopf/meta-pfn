"""GW-only (issue #13, notebook 06): can order-based Gromov–Wasserstein recover B_inA landmarks, in isolation?

For one pair (A's first n points, B's N_B points; phase-1 prior, d = 1, h = id), a coupling π (n × N_B, rows sum to
1: semi-relaxed) matches every A point to B points. Relations are within-task order only, s(i, j) =
(sign(x_i - x_j), sign(y_i - y_j)), invariant to monotone T and h. The truth for A point a is u*_a = T⁻¹(x_a)
(interpolated from B_inA; `covered` = x_a inside T's image of B's window).

Solvers (all iterate the same message m_{b,a'} = Σ_b' r_B(b, b') π_a'b', weight-tied):
- `peyre`: semi-relaxed GW with square loss on the sign relations, mirror descent on the row logits,
  L ← L - η ∇_π GW(π), from the uniform coupling (+ restarts from random logits). No learning.
- `unrolled`: the same iteration with learned relation embeddings r = φ(s) ∈ R^h, a bilinear score
  S_ab = Σ_a' <W r_A(a, a'), m_{b,a'}> / (h n), a learned column-mass term and step size: L ← L + η S - κ q_b.
- `gru`: a RAFT-style operator on the logits: per (a, b) a GRU state updated from [S_ab, q_b, L_ab, π_ab],
  L ← L + Linear(state).
The learned solvers are trained with a LUPI cross-entropy of every iterate's rows against a narrow soft target
around u*_a (isolated component test, not part of a deployed model's loss).
Metrics per A point: the landmark of row a is b̂ = argmax_b π_ab (the mode: rows can be bimodal); its error in A's
frame err_x = |x_a - T(u_b̂)| (comparable with notebook 04's E4 σ), in B's frame err_u = |u_b̂ - u*_a|, and the
row's π-mass within 0.05 of u*_a.
"""

import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .regpfn.prior import Batch, PairPrior, PriorCfg
from .regpfn.train import eval_set

N_A_EVAL = (2, 5, 10, 20)
CELLS = ("C00", "C10")  # h = id (decision 0003: isolation)


def relations(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[P, N, d], [P, N] -> [P, N, N, d + 1] order relations (0 on the diagonal)."""
    z = torch.cat([x, y[..., None]], -1)
    return torch.sign(z[:, :, None, :] - z[:, None, :, :])


def truth(b: Batch, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """-> u*_a [P, n] and covered [P, n] (d = 1; T is increasing in u)."""
    u, x = b.ub[..., 0], b.xb_inA[..., 0]
    order = u.argsort(1)
    us, xs = torch.gather(u, 1, order), torch.gather(x, 1, order)
    xa = b.xa[:, :n, 0]
    j = torch.searchsorted(xs.contiguous(), xa.contiguous()).clamp(1, xs.shape[1] - 1)
    x0, x1 = torch.gather(xs, 1, j - 1), torch.gather(xs, 1, j)
    u0, u1 = torch.gather(us, 1, j - 1), torch.gather(us, 1, j)
    w = ((xa - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
    return u0 + w * (u1 - u0), (xa >= xs[:, :1]) & (xa <= xs[:, -1:])


def soft_target(ub: torch.Tensor, u_star: torch.Tensor, s: float = 0.02) -> torch.Tensor:
    """[P, Nb, 1], [P, n] -> [P, n, Nb] target rows ∝ exp(-(u_b - u*_a)² / 2s²)."""
    return torch.softmax(-((ub[:, None, :, 0] - u_star[..., None]) ** 2) / (2 * s**2), -1)


def gw_grad(sa: torch.Tensor, sb: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """∇_π of Σ_{a,a',b,b'} |s_A(a,a') - s_B(b,b')|² π_ab π_a'b' / n (rows of π sum to 1) -> [P, n, Nb]."""
    n = pi.shape[1]
    q = pi.sum(1)  # [P, Nb] column mass
    t_a = (sa**2).sum((-1, -2))  # [P, n]  Σ_a' |s_A|² (row mass 1)
    t_b = torch.einsum("pbjc,pj->pb", sb**2, q)  # Σ_b' |s_B(b,b')|² q_b'
    cross = sum(sa[..., c] @ pi @ sb[..., c].transpose(1, 2) for c in range(sa.shape[-1]))
    return 2 * (t_a[..., None] + t_b[:, None, :] - 2 * cross) / n


def peyre(sa, sb, n_iter: int = 200, eta: float = 1.0, init: torch.Tensor | None = None):
    """Semi-relaxed GW by mirror descent on row logits -> (π [P, n, Nb], energy [P])."""
    P, n, _, _ = sa.shape
    Nb = sb.shape[1]
    L = torch.zeros(P, n, Nb, device=sa.device) if init is None else init
    for _ in range(n_iter):
        pi = torch.softmax(L, -1)
        L = L - eta * gw_grad(sa, sb, pi)
        L = L - L.max(-1, keepdim=True).values
    pi = torch.softmax(L, -1)
    return pi, (gw_grad(sa, sb, pi) * pi).sum((1, 2)) / 2  # quadratic objective: E = π·∇E / 2


class LearnedGW(nn.Module):
    def __init__(self, d: int = 1, h: int = 16, n_iter: int = 10, gru: bool = False, hs: int = 16):
        super().__init__()
        self.h, self.n_iter, self.gru = h, n_iter, gru
        self.phi = nn.Sequential(nn.Linear(d + 1, h), nn.GELU(), nn.Linear(h, h))
        self.W = nn.Parameter(torch.eye(h))
        self.log_eta = nn.Parameter(torch.tensor(0.0))
        self.kappa = nn.Parameter(torch.tensor(0.0))
        if gru:
            self.cell = nn.GRUCell(4, hs)
            self.out = nn.Linear(hs, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def score(self, ra, rb, pi):
        m = torch.einsum("pbjh,paj->pbah", rb, pi)  # m_{b,a'}
        return torch.einsum("pkah,hg,pbag->pkb", ra, self.W, m) / (self.h * ra.shape[1])

    def forward(self, sa, sb):
        """-> list of π iterates [P, n, Nb] (index 0 = uniform)."""
        P, n = sa.shape[:2]
        Nb = sb.shape[1]
        mask_a = 1 - torch.eye(n, device=sa.device)[None, :, :, None]
        mask_b = 1 - torch.eye(Nb, device=sb.device)[None, :, :, None]
        ra, rb = self.phi(sa) * mask_a, self.phi(sb) * mask_b
        L = torch.zeros(P, n, Nb, device=sa.device)
        st = torch.zeros(P * n * Nb, self.cell.hidden_size, device=sa.device) if self.gru else None
        out = [torch.softmax(L, -1)]
        for _ in range(self.n_iter):
            pi = out[-1]
            S = self.score(ra, rb, pi)
            q = pi.sum(1, keepdim=True).expand_as(L)
            if self.gru:
                feats = torch.stack([S, q, L, pi], -1).reshape(-1, 4)
                st = self.cell(feats, st)
                L = L + self.out(st).view(P, n, Nb)
            else:
                L = L + self.log_eta.exp() * S - self.kappa * q
            L = L - L.max(-1, keepdim=True).values.detach()
            out.append(torch.softmax(L, -1))
        return out


@torch.no_grad()
def validate(model: LearnedGW, val: dict, device: str) -> dict[str, float]:
    """Landmark metrics on fixed validation pairs (C10, n_A = 5 and 10), for the training curves."""
    model.eval()
    out = {}
    for n in (5, 10):
        b = val[n]
        m = landmark_metrics(model(relations(b.xa[:, :n], b.ya[:, :n]), relations(b.ub, b.yb))[-1], b, n)
        cv = m["covered"]
        out[f"val/med_err_x_n{n}"] = float(m["err_x"][cv].median())
        out[f"val/frac_005_n{n}"] = float((m["err_x"][cv] < 0.05).float().mean())
        out[f"val/mass_n{n}"] = float(m["mass"][cv].mean())
    model.train()
    return out


def train_learned(
    model: LearnedGW, steps: int, device: str, seed=0, batch=32, n_b=64, lr=1e-3, log=print, metrics=None, every=250
):
    """metrics(values, step): called every `every` steps with the training CE and the validation metrics."""
    val_pr = PairPrior(PriorCfg(d=1), 128, seed + 999, device)
    val = {n: val_pr.sample("C10", n, 1, n_b) for n in (5, 10)}
    pr = {c: PairPrior(PriorCfg(d=1), batch, seed + 17 * i, device) for i, c in enumerate(CELLS)}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = np.random.default_rng(seed)
    hist, t0 = [], time.time()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        cell = CELLS[int(g.random() < 0.8)]
        n = int(g.integers(2, 21))
        b = pr[cell].sample(cell, n, 1, n_b)
        u_star, cov = truth(b, n)
        sa, sb = relations(b.xa[:, :n], b.ya[:, :n]), relations(b.ub, b.yb)
        its = model(sa, sb)
        tgt = soft_target(b.ub, u_star)
        T = len(its) - 1
        ce = [-(tgt * p.clamp_min(1e-12).log()).sum(-1) for p in its[1:]]
        per_it = torch.stack([(c * cov).sum() / cov.sum().clamp_min(1) for c in ce])
        loss = (0.8 ** torch.arange(T - 1, -1, -1, device=per_it.device) * per_it).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % every == 0:
            vals = {"train/ce_final": float(per_it[-1].detach()), "train/loss": float(loss.detach())}
            vals |= validate(model, val, device)
            vals |= {"train/eta": float(model.log_eta.exp()), "train/kappa": float(model.kappa)}
            hist.append({"step": step + 1, "seconds": time.time() - t0, **vals})
            if metrics is not None:
                metrics(vals, step + 1)
            log(f"gw step {step + 1} ce {vals['train/ce_final']:.3f} val frac(n=5) {vals['val/frac_005_n5']:.2f}")
    return pd.DataFrame(hist)


def landmark_metrics(pi: torch.Tensor, b: Batch, n: int) -> dict[str, torch.Tensor]:
    u_star, cov = truth(b, n)
    bh = pi.argmax(-1)  # [P, n]
    ub, xb = b.ub[..., 0], b.xb_inA[..., 0]
    u_hat, x_hat = torch.gather(ub, 1, bh), torch.gather(xb, 1, bh)
    near = (ub[:, None, :] - u_star[..., None]).abs() < 0.05
    return {
        "err_x": (x_hat - b.xa[:, :n, 0]).abs(),
        "err_u": (u_hat - u_star).abs(),
        "mass": (pi * near).sum(-1),
        "covered": cov,
    }


@torch.no_grad()
def evaluate(method: str, model: LearnedGW | None, device: str, n_pairs=256, n_b=64, restarts=8, seed=12345):
    """Rows (cell, n_a, method, pair, row, covered, err_x, err_u, mass) incl. the `identity` and `value` baselines;
    for peyre also `peyre_oracle_k` (the restart closest to the truth: recall@K)."""
    rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for cell in CELLS:
        b = eval_set(1, cell, n_pairs, 128, n_b, seed, device)
        for n in N_A_EVAL:
            sa, sb = relations(b.xa[:, :n], b.ya[:, :n]), relations(b.ub, b.yb)
            res = {}
            if method == "peyre":
                pis, ens = [], []
                for r in range(restarts):
                    init = None if r == 0 else torch.randn(n_pairs, n, n_b, generator=g).to(device)
                    pi, en = peyre(sa, sb, init=init)
                    pis.append(pi)
                    ens.append(en)
                best = torch.stack(ens).argmin(0)
                stack = torch.stack(pis)  # [R, P, n, Nb]
                res["peyre"] = stack[best, torch.arange(n_pairs)]
                errs = torch.stack([landmark_metrics(p, b, n)["err_x"].amax(1) for p in pis])  # worst row per pair
                res["peyre_oracle_k"] = stack[errs.argmin(0), torch.arange(n_pairs)]
                # baselines (once, with the reference solver's member)
                d_id = (b.ub[:, None, :, 0] - b.xa[:, :n, 0, None]).abs()
                res["identity"] = torch.nn.functional.one_hot(d_id.argmin(-1), n_b).float()
                d_val = (b.yb[:, None, :] - b.ya[:, :n, None]).abs()
                res["value"] = torch.nn.functional.one_hot(d_val.argmin(-1), n_b).float()
            else:
                assert model is not None
                model.eval()
                res[method] = model(sa, sb)[-1]
            for name, pi in res.items():
                m = {k: v.cpu().numpy() for k, v in landmark_metrics(pi, b, n).items()}
                for p in range(n_pairs):
                    for a in range(n):
                        rows.append(
                            (cell, n, name, p, a, bool(m["covered"][p, a]), m["err_x"][p, a], m["err_u"][p, a],
                             m["mass"][p, a])
                        )  # fmt: skip
    cols = ["cell", "n_a", "method", "pair", "row", "covered", "err_x", "err_u", "mass"]
    return pd.DataFrame(rows, columns=cols)


@torch.no_grad()
def examples(method: str, model: LearnedGW | None, device: str, n_pairs=8, n_b=64, seed=12345):
    """Long tables for 1d plots, for the first `n_pairs` evaluation pairs per cell and n_A ∈ {5, 10}:
    `points` (role A / B / A_test with x or u, y, x_inA, u*, landmark) and `coupling` (π per A row and B point)."""
    pts, cps = [], []
    for cell in CELLS:
        b = eval_set(1, cell, 256, 128, n_b, seed, device)
        b = Batch(*(t[:n_pairs] if torch.is_tensor(t) and t.dim() > 0 else t for t in b.__dict__.values()))
        for n in (5, 10):
            sa, sb = relations(b.xa[:, :n], b.ya[:, :n]), relations(b.ub, b.yb)
            if method == "peyre":
                pi = peyre(sa, sb)[0]
            else:
                assert model is not None
                model.eval()
                pi = model(sa, sb)[-1]
            u_star, cov = truth(b, n)
            bh = pi.argmax(-1)
            for p in range(n_pairs):
                for i in range(n_b):
                    pts.append((cell, n, p, "B", i, float(b.ub[p, i, 0]), float(b.yb[p, i]),
                                float(b.xb_inA[p, i, 0]), float(b.yb_inA[p, i]), bool(b.inside[p, i])))  # fmt: skip
                for i in range(b.xa.shape[1]):
                    role = "A" if i < n else ("A_test" if i >= 20 else "A_pool")
                    if role == "A_pool":
                        continue
                    pts.append((cell, n, p, role, i, float(b.xa[p, i, 0]), float(b.ya[p, i]),
                                float(u_star[p, i]) if i < n else float("nan"),
                                float(b.ub[p, bh[p, i], 0]) if i < n else float("nan"),
                                bool(cov[p, i]) if i < n else True))  # fmt: skip
                for a in range(n):
                    for i in range(n_b):
                        cps.append((cell, n, p, a, i, float(pi[p, a, i])))
    cols = ["cell", "n_a", "pair", "role", "i", "pos", "y", "aux1", "aux2", "flag"]
    return (
        pd.DataFrame(pts, columns=cols),
        pd.DataFrame(cps, columns=["cell", "n_a", "pair", "row", "b", "pi"]),
    )
