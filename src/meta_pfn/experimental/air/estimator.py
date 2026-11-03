"""Inference (proposal §3.4): self-normalized importance sampling over θ̃ with the decoder's chain-rule evidence.

For each task and draw θ̃_k: B_k = T_k⁻¹(x_b); ev_k = Σ_a log D(y_a | x_a, A_<a, B_k), averaged over `n_perm` random
orders of A; P_k = D(· | x*, A, B_k). log w_k = log p(θ̃_k) + ev_k − log q(θ̃_k); p̂₁ = Σ softmax(log w)_k P_k(y*).
Null: log Z₀ = Σ_a log D_null(y_a | x_a, A_<a); P₁ = σ(logit ρ̄ + log Ẑ₁ − log Z₀); output = (1 − P₁) D_null + P₁ p̂₁.
All in log space. Batched over tasks × draws in chunks.
"""

import math

import torch

from .decoder import Decoder, bar_logpdf
from .prior import Tasks, flow, log_prior, log_prior_s


@torch.no_grad()
def decode(dec: Decoder, t: Tasks, theta: torch.Tensor, n_perm: int = 2, chunk: int = 2048, use_b: bool = True,
           sig: torch.Tensor | None = None):  # fmt: skip
    """theta [P, K, J] -> (evidence [P, K] (mean over orders), log density of y* [P, K, Q], permutation std [P, K])."""
    P, K, J = theta.shape
    n, Q = t.xa.shape[1], t.xq.shape[1]
    rep = lambda x: x[:, None].expand(P, K, *x.shape[1:]).reshape(P * K, *x.shape[1:])  # noqa: E731
    xa, ya, xq, yq, yb = rep(t.xa), rep(t.ya), rep(t.xq), rep(t.yq), rep(t.yb)
    sg = None if sig is None else rep(torch.as_tensor(sig, device=xa.device, dtype=xa.dtype).expand(P))
    xb_t = flow(rep(t.xb), theta.reshape(P * K, J), t.s_t, inverse=True)
    evs, lqs = [], []
    for s in range(0, P * K, chunk):
        sl = slice(s, s + chunk)
        ev_p, lq = [], None
        for _ in range(max(n_perm, 1)):
            lg_q, lg_e = dec(xb_t[sl], yb[sl], xa[sl], ya[sl], xq[sl], use_b=use_b,
                             sig=None if sg is None else sg[sl])  # fmt: skip
            ev_p.append(bar_logpdf(lg_e, ya[sl]).sum(-1) if n else lg_e.new_zeros(lg_e.shape[0]))
            lq = bar_logpdf(lg_q, yq[sl]) if lq is None else lq  # the x* predictive sees all of A: order-free
        evs.append(torch.stack(ev_p, -1))
        lqs.append(lq)
    ev = torch.cat(evs).reshape(P, K, -1)
    return (
        ev.mean(-1),
        torch.cat(lqs).reshape(P, K, Q),
        ev.std(-1) if ev.shape[-1] > 1 else torch.zeros_like(ev[..., 0]),
    )


@torch.no_grad()
def importance(dec: Decoder, t: Tasks, theta: torch.Tensor, log_q: torch.Tensor, n_perm: int = 2, chunk: int = 2048,
               sig: float | None = None):  # fmt: skip
    """-> dict: log Ẑ₁ [P], log p̂₁(y*) [P, Q], ESS/K [P], log weights [P, K]. sig = Σ (pilot 3): theta are θ_c draws,
    with prior N(0, (1 + Σ) I), and the decoder is told Σ."""
    K = theta.shape[1]
    ev, lq, _ = decode(dec, t, theta, n_perm, chunk, sig=sig)
    lw = (log_prior(theta) if sig is None else log_prior_s(theta, sig)) + ev - log_q
    wn = lw - torch.logsumexp(lw, 1, keepdim=True)
    ess = torch.exp(-torch.logsumexp(2 * wn, 1)) / K
    return {"log_z1": torch.logsumexp(lw, 1) - math.log(K), "lp1": torch.logsumexp(wn[..., None] + lq, 1),
            "ess": ess, "lw": lw}  # fmt: skip


@torch.no_grad()
def null_branch(dec: Decoder, t: Tasks, n_perm: int = 2):
    """A-only: (log Z₀ [P], log density of y* [P, Q])."""
    th0 = t.theta.new_zeros(t.theta.shape[0], 1, t.theta.shape[1])
    ev, lq, _ = decode(dec, t, th0, n_perm, use_b=False)
    return ev[:, 0], lq[:, 0]


def gate(log_z1: torch.Tensor, log_z0: torch.Tensor, lp1: torch.Tensor, lp0: torch.Tensor, rho: float):
    """P₁ and the mixed log density: (1 − P₁) p₀ + P₁ p̂₁."""
    if rho >= 1.0:
        return torch.ones_like(log_z1), lp1
    logit = math.log(rho / (1 - rho)) + log_z1 - log_z0
    p1 = torch.sigmoid(logit)
    lp = torch.logaddexp(torch.nn.functional.logsigmoid(-logit)[:, None] + lp0,
                         torch.nn.functional.logsigmoid(logit)[:, None] + lp1)  # fmt: skip
    return p1, lp
