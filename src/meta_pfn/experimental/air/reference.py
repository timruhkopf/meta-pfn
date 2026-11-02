"""Exact references in the dense-B limit (proposal §5): g known exactly, h = id (H0), true σ_A.

- G-ref (J = 2): posterior on a 101² grid over θ̃ ∈ [-4, 4]² (standardized, so the grid prior is N(0, I));
  likelihood Π_a N(y_a; g(T_θ(x_a)), σ_A²); predictive = grid mixture of N(y*; g(T_θ(x*)), σ_A²) integrated over the
  bar buckets (so it is scored on the same bar scale as the models).
- IS-exact (J > 2): the same likelihood and predictive with draws from a proposal (q_mix), self-normalized;
  ESS reported.
These know g exactly while the models see 128 noisy B points: a floor, not a target (proposal §5 caveat).
"""

import math

import torch

from .decoder import N_BINS, Y_HI, Y_LO
from .prior import Tasks, flow, log_prior


def bucket_lp(mu: torch.Tensor, sig: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log (mass of y's bucket under N(mu, sig²)) − log width; broadcastable."""
    w = (Y_HI - Y_LO) / N_BINS
    idx = ((y - Y_LO) / w).floor().clamp(0, N_BINS - 1)
    lo, hi = Y_LO + idx * w, Y_LO + (idx + 1) * w
    c = lambda b: 0.5 * (1 + torch.erf((b - mu) / (sig * math.sqrt(2))))  # noqa: E731
    return torch.log((c(hi) - c(lo)).clamp_min(1e-30)) - math.log(w)


def grid(n: int = 101, lim: float = 4.0, device="cpu") -> torch.Tensor:
    ax = torch.linspace(-lim, lim, n, device=device)
    return torch.cartesian_prod(ax, ax)  # [n², 2]


@torch.no_grad()
def exact_terms(t: Tasks, j: int, theta: torch.Tensor):
    """Task j, draws theta [G, J] -> (log-lik of A [G], log bucket density of y* [G, Q])."""
    G = theta.shape[0]
    x = torch.cat([t.xa[j], t.xq[j]])[None].expand(G, -1)
    u = flow(x, theta, t.s_t)
    gj = t.g.at(torch.full((G,), j, device=theta.device, dtype=torch.long))
    m = gj(u)
    n = t.xa.shape[1]
    sig = t.sig_a[j]
    ll = (-0.5 * ((t.ya[j] - m[:, :n]) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)).sum(1)
    lq = bucket_lp(m[:, n:], sig, t.yq[j][None])
    return ll, lq


@torch.no_grad()
def grid_ref(t: Tasks, j: int, g_pts: torch.Tensor) -> dict:
    """-> {"lp": log density of y* [Q], "logpost": normalized log posterior on the grid [G], "ll": [G]}."""
    ll, lq = exact_terms(t, j, g_pts)
    lpost = log_prior(g_pts) + ll
    lpost = lpost - torch.logsumexp(lpost, 0)
    return {"lp": torch.logsumexp(lpost[:, None] + lq, 0), "logpost": lpost, "ll": ll}


@torch.no_grad()
def is_exact(t: Tasks, j: int, theta: torch.Tensor, log_q: torch.Tensor) -> dict:
    """theta [K, J] from a proposal with log density log_q [K] -> {"lp": [Q], "ess": float}."""
    ll, lq = exact_terms(t, j, theta)
    lw = log_prior(theta) + ll - log_q
    wn = lw - torch.logsumexp(lw, 0)
    return {"lp": torch.logsumexp(wn[:, None] + lq, 0), "ess": float(torch.exp(-torch.logsumexp(2 * wn, 0)))}
