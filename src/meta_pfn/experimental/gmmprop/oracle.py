"""Closed-form references for the GMM-token propagation experiment (issue #23).

- `oracle`: the exact Bayes predictive of u* given the *labelled* token data: a mixture over the length-scale grid
  (posterior from the GP evidence of all hypotheses and dimensions) and over the hypotheses (weights w_m) of the GP
  posteriors. Exact for M = 1; for M >= 2 optimistic, since the model sees shuffled (unlabelled) components.
- `oracle_fact`: the product over output dimensions of the oracle's 1-dim marginals (the best per-dimension head).
- `prior`: N(0, ALPHA^2 (1 + JIT) I).
All per query, in nats (joint over the d dimensions), float64.
"""

import math

import torch

from .prior import ALPHA, JIT, M_MAX, Batch, ells, se_kernel

LOG2PI = math.log(2 * math.pi)


def _lognorm(y, mean, var):
    return -0.5 * (LOG2PI + torch.log(var) + (y - mean) ** 2 / var)


@torch.no_grad()
def predictive(b: Batch) -> dict:
    """Mixture parameters of the oracle per query: log weights [B, Q, L*M] (-inf for absent hypotheses),
    means [B, Q, L*M, d], variances [B, Q, L*M] (isotropic), and the posterior over the length scales [B, L]."""
    f = lambda t: t.double()  # noqa: E731
    x, xq, s, mu = f(b.x), f(b.xq), f(b.s), f(b.mu_lab)
    B, n, d = x.shape
    Q = xq.shape[1]
    ell = ells(d).to(x.device)  # [L]
    L_ = len(ell)
    present = torch.arange(M_MAX, device=x.device)[None] < b.m[:, None]  # [B, M]
    logw = torch.where(present, torch.log(f(b.w_lab).clamp_min(1e-300)), -math.inf)  # [B, M]
    if n == 0:
        logpost = torch.full((B, L_), -math.log(L_), dtype=torch.float64, device=x.device)
        mean = torch.zeros(B, L_, Q, M_MAX, d, dtype=torch.float64, device=x.device)
        var = torch.full((B, L_, Q), ALPHA**2 * (1 + JIT), dtype=torch.float64, device=x.device)
    else:
        ellb = ell[None].expand(B, -1)  # [B, L]
        K = se_kernel(x[:, None], x[:, None], ellb) + torch.diag_embed(s.square())[:, None]  # [B, L, n, n]
        C = torch.linalg.cholesky(K)
        Y = mu.reshape(B, 1, n, M_MAX * d).expand(-1, L_, -1, -1)
        z = torch.linalg.solve_triangular(C, Y, upper=False)  # [B, L, n, M*d]
        a = torch.linalg.solve_triangular(C.transpose(-1, -2), z, upper=True)
        # GP evidence per column, summed over present hypotheses and all dimensions
        col = -0.5 * z.square().sum(-2) - torch.log(torch.diagonal(C, dim1=-2, dim2=-1)).sum(-1, keepdim=True)
        col = col - 0.5 * n * LOG2PI  # [B, L, M*d]
        colmask = present[:, None, :, None].expand(-1, L_, -1, d).reshape(B, L_, M_MAX * d)
        ev = torch.where(colmask, col, 0.0).sum(-1)  # [B, L]
        logpost = torch.log_softmax(ev, -1)
        Kq = se_kernel(xq[:, None], x[:, None], ellb)  # [B, L, Q, n]
        mean = (Kq @ a).view(B, L_, Q, M_MAX, d)
        v = torch.linalg.solve_triangular(C, Kq.transpose(-1, -2), upper=False)  # [B, L, n, Q]
        var = (ALPHA**2 * (1 + JIT) - v.square().sum(-2)).clamp_min(ALPHA**2 * JIT)  # [B, L, Q]
    lw = logpost[:, :, None, None] + logw[:, None, None, :]  # [B, L, 1, M]
    lw = lw.expand(-1, -1, Q, -1)
    perm = lambda t: t.permute(0, 2, 1, *range(3, t.dim()))  # noqa: E731  [B, L, Q, ...] -> [B, Q, L, ...]
    return {
        "logw": perm(lw).reshape(B, Q, L_ * M_MAX),
        "mean": perm(mean).reshape(B, Q, L_ * M_MAX, d),
        "var": perm(var[..., None].expand(-1, -1, -1, M_MAX)).reshape(B, Q, L_ * M_MAX),
        "logpost_ell": logpost,
    }


def mixture_logpdf(p: dict, y: torch.Tensor, fact: bool = False) -> torch.Tensor:
    """log density of y [B, Q, d] (or, with fact, the sum over dimensions of the 1-dim marginal log densities)."""
    y = y.double()[:, :, None]  # [B, Q, 1, d]
    lp = _lognorm(y, p["mean"], p["var"][..., None])  # [B, Q, C, d]
    if fact:
        return torch.logsumexp(p["logw"][..., None] + lp, -2).sum(-1)
    return torch.logsumexp(p["logw"] + lp.sum(-1), -1)


@torch.no_grad()
def references(b: Batch, chunk: int = 64) -> dict[str, torch.Tensor]:
    """Per-query log densities [B, Q] of the target under oracle, oracle_fact and prior."""
    out = {"oracle": [], "oracle_fact": [], "prior": []}
    for c in range(0, b.x.shape[0], chunk):
        bc = Batch(**{k: v[c : c + chunk] for k, v in b.__dict__.items()})
        p = predictive(bc)
        out["oracle"].append(mixture_logpdf(p, bc.yq))
        out["oracle_fact"].append(mixture_logpdf(p, bc.yq, fact=True))
        out["prior"].append(_lognorm(bc.yq.double(), 0.0, torch.tensor(ALPHA**2 * (1 + JIT))).sum(-1))
    return {k: torch.cat(v) for k, v in out.items()}
