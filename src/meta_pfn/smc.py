"""A converged reference for the Bayes learner within one warp family: sequential Monte Carlo over T.

E0's `bayes` predictor weights a fixed set of prior draws θ_1..K by the likelihood of A's points. The pair
prior's observation noise is tiny (sd ≈ 1e-3 on the [0, 1] scale), so that likelihood is sharp and the
weights collapse onto one or two draws (pilot ESS ≈ 1-3 at n_A ≥ 4). This module targets the same posterior,
p(θ | D_B, A_<n) ∝ p(θ) Π_i PFN(y_i | B ∪ T_θ(A_<i), T_θ(x_i)), with adaptive tempering, resampling and
random-walk Metropolis moves, so that the particles follow the posterior instead of the prior.

Particles live in an unconstrained space u in which the family prior factorizes:
  b = b_min + (1 - b_min) σ(u_0), a = (1 - b) σ(u_1)     (uniform b and a: logistic u_0, u_1)
  α = exp(u_2), β = exp(u_3)                              (warp only: normal u_2, u_3)
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .oracle import _contexts, logp
from .pairs import WarpFamily, apply_warp

N_FREE = {"window": 2, "warp": 4}


def to_theta(u: torch.Tensor, kind: str, family: WarpFamily) -> torch.Tensor:
    """u [K, d, N_FREE[kind]] -> θ [K, d, 4]."""
    b = family.b_min + (1 - family.b_min) * torch.sigmoid(u[..., 0])
    a = (1 - b) * torch.sigmoid(u[..., 1])
    if kind == "warp":
        al, be = u[..., 2].exp(), u[..., 3].exp()
    else:
        al = be = torch.ones_like(b)
    return torch.stack([a, b, al, be], -1)


def log_prior(u: torch.Tensor, kind: str, family: WarpFamily) -> torch.Tensor:
    """log p(u) [K], the family prior pushed to u (the Jacobians make uniform a, b logistic in u)."""
    lp = (F.logsigmoid(u[..., :2]) + F.logsigmoid(-u[..., :2])).sum((-1, -2))
    if kind == "warp":
        s = family.kum_log_sd
        lp = lp + (-0.5 * (u[..., 2:] / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))).sum((-1, -2))
    return lp


def sample_prior(k: int, d: int, kind: str, family: WarpFamily, g: torch.Generator) -> torch.Tensor:
    u = torch.logit(torch.rand(k, d, 2, generator=g).clamp(1e-6, 1 - 1e-6))
    if kind == "warp":
        u = torch.cat([u, family.kum_log_sd * torch.randn(k, d, 2, generator=g)], -1)
    return u


def ess(logw: torch.Tensor) -> float:
    return float(torch.exp(2 * torch.logsumexp(logw, 0) - torch.logsumexp(2 * logw, 0)))


def next_increment(logw: torch.Tensor, ll: torch.Tensor, remaining: float, target: float) -> float:
    """The largest tempering step ≤ remaining that keeps the ESS of logw + step·ll at ≥ target (bisection)."""
    if ess(logw + remaining * ll) >= target:
        return remaining
    lo, hi = 0.0, remaining
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ess(logw + mid * ll) >= target else (lo, mid)
    return max(lo, 1e-6)


def systematic(logw: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    k = len(logw)
    c = torch.cumsum(torch.softmax(logw, 0).double(), 0)
    pos = (torch.rand(1, generator=g, dtype=torch.float64) + torch.arange(k, dtype=torch.float64)) / k
    return torch.searchsorted(c, pos).clamp(max=k - 1)


@dataclass
class Data:
    xb: torch.Tensor
    yb: torch.Tensor
    xa: torch.Tensor
    ya: torch.Tensor
    xq: torch.Tensor
    yq: torch.Tensor


def point_ll(pfn, data: Data, theta: torch.Tensor, js: list[int], chunk: int) -> torch.Tensor:
    """ll[k, m] = log PFN(y_j | B ∪ T_k(A_<j), T_k(x_j)) for j = js[m], all in one batched call."""
    K, n_ctx = len(theta), max(js)
    use_b = torch.ones(K, dtype=torch.bool)
    Xs, Ys, Ms, Qs, Yq = [], [], [], [], []
    for j in js:
        X, Y, M = _contexts(data.xb, data.yb, data.xa, data.ya, theta, use_b, j)
        pad = n_ctx - j  # pad to a common length; padded tokens are masked out
        Xs.append(F.pad(X, (0, 0, 0, pad)))
        Ys.append(F.pad(Y, (0, pad)))
        Ms.append(F.pad(M, (0, pad), value=False))
        Qs.append(apply_warp(theta, data.xa[None, j : j + 1].expand(K, -1, -1)))
        Yq.append(data.ya[j : j + 1][None].expand(K, -1))
    out = logp(pfn, torch.cat(Xs), torch.cat(Ys), torch.cat(Ms), torch.cat(Qs), torch.cat(Yq), chunk)[:, 0]
    return torch.nan_to_num(out, nan=-1e4, neginf=-1e4).view(len(js), K).T


@dataclass
class SMCResult:
    logq: np.ndarray  # [len(grid), Q] log posterior-predictive density of each held-out query
    logz: np.ndarray  # [len(grid)] log p(A_<n | D_B, family)
    t_err: np.ndarray  # [len(grid)] posterior RMS of T(x) - T_true(x) over the box (nan without a truth)
    stages: np.ndarray  # [len(grid)] tempering stages used so far
    accept: float  # mean Metropolis acceptance rate


@torch.no_grad()
def smc(
    pfn,
    data: Data,
    kind: str,
    family: WarpFamily,
    k: int,
    grid: list[int],
    g: torch.Generator,
    theta_true: torch.Tensor | None = None,
    n_moves: int = 3,
    ess_frac: float = 0.5,
    chunk: int = 512,
) -> SMCResult:
    d = data.xb.shape[-1]
    u = sample_prior(k, d, kind, family, g)
    lp = log_prior(u, kind, family)
    S = torch.zeros(k)  # Σ_{j<n} ll_j at each particle
    logw = torch.full((k,), -np.log(k))
    logz, stages, scale, acc = 0.0, 0, 1.0, []
    xgrid = torch.rand(256, d, generator=g)
    rows = []
    for n in range(max(grid) + 1):
        theta = to_theta(u, kind, family)
        if n in grid:
            X, Y, M = _contexts(data.xb, data.yb, data.xa, data.ya, theta, torch.ones(k, dtype=torch.bool), n)
            q = apply_warp(theta, data.xq[None].expand(k, -1, -1))
            lq = logp(pfn, X, Y, M, q, data.yq[None].expand(k, -1), chunk)
            w = logw - torch.logsumexp(logw, 0)
            err = float("nan")
            if theta_true is not None:
                t = apply_warp(theta, xgrid[None].expand(k, -1, -1))
                t0 = apply_warp(theta_true[None], xgrid[None])
                err = float(((t - t0) ** 2).mean((1, 2)).mul(w.exp()).sum().sqrt())
            rows.append((torch.logsumexp(w[:, None] + lq, 0).numpy(), logz, err, stages))
        if n == max(grid):
            break
        ln = point_ll(pfn, data, theta, [n], chunk)[:, 0]
        phi = 0.0
        while phi < 1.0:
            step = next_increment(logw, ln, 1.0 - phi, ess_frac * k)
            inc = logw + step * ln
            logz += float(torch.logsumexp(inc, 0))
            logw = inc - torch.logsumexp(inc, 0)
            phi = 1.0 if step >= 1.0 - phi - 1e-9 else phi + step
            stages += 1
            if ess(logw) >= ess_frac * k and phi >= 1.0:
                break
            idx = systematic(logw, g)
            u, lp, S, ln = u[idx], lp[idx], S[idx], ln[idx]
            logw = torch.full((k,), -np.log(k))
            # random-walk Metropolis on prior × Π_{j<n} ll_j × ll_n^φ, proposal shaped by the particle cloud
            flat = u.reshape(k, -1)
            cov = torch.cov(flat.T).reshape(flat.shape[1], flat.shape[1]) + 1e-4 * torch.eye(flat.shape[1])
            L = torch.linalg.cholesky(cov)
            for _ in range(n_moves):
                eps = torch.randn(flat.shape, generator=g)
                prop = (flat + scale * 2.38 / np.sqrt(flat.shape[1]) * eps @ L.T).reshape(u.shape)
                lpp = log_prior(prop, kind, family)
                llp = point_ll(pfn, data, to_theta(prop, kind, family), list(range(n + 1)), chunk)
                Sp, lnp = llp[:, :n].sum(1), llp[:, n]
                a = (lpp + Sp + phi * lnp) - (lp + S + phi * ln)
                ok = torch.log(torch.rand(k, generator=g)) < a
                u, lp, S, ln = (
                    torch.where(ok.view(-1, *[1] * (t.dim() - 1)), tp, t)
                    for t, tp in ((u, prop), (lp, lpp), (S, Sp), (ln, lnp))
                )
                flat = u.reshape(k, -1)
                rate = float(ok.float().mean())
                acc.append(rate)
                scale *= 1.3 if rate > 0.4 else (0.7 if rate < 0.15 else 1.0)
        S = S + ln
    return SMCResult(
        np.stack([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
        np.array([r[3] for r in rows]),
        float(np.mean(acc)) if acc else float("nan"),
    )
