"""Forward filtering, backward sampling over candidate correspondences (spec §3.4), batched over tasks, log space.

A's points are sorted by x; point a has candidates u_{a,i} (i < C, invalid slots masked) with log claim scores
log C_a(i). Transition from candidate i of point a to candidate j of point a + 1, with Δx = x_{a+1} − x_a and
Δu = u_{a+1,j} − u_{a,i}:
    K(j | i) ∝ (1/Δu) exp(−(log(Δu/Δx) − m_K)² / (2 s_K²))   if Δu > 0   (m_K: deviation D3)
               ε₀                                              if Δu = 0
               δ                                               if Δu < 0   (soft monotonicity: q(c) > 0 on every path)
normalised over the valid j. q(c) ∝ C_1(c_1) Π_a K(c_{a+1} | c_a) C_{a+1}(c_{a+1}); samples come with their exact
log q (the sum of the log draw probabilities), and `log_q_of_path` evaluates any path (equal by construction)."""

import math

import torch

EPS0, DELTA = 1e-2, 1e-6


def log_trans(u: torch.Tensor, valid: torch.Tensor, dx: torch.Tensor, m: torch.Tensor, s: torch.Tensor,
              eps0: float = EPS0, delta: float = DELTA) -> torch.Tensor:  # fmt: skip
    """u, valid [P, n, C]; dx, m, s [P, n − 1] -> log K [P, n − 1, C (i), C (j)], normalised over valid j."""
    du = u[:, 1:, None, :] - u[:, :-1, :, None]
    dxe, me, se = dx[..., None, None], m[..., None, None], s[..., None, None]
    pos = du > 1e-9
    dus = du.clamp_min(1e-9)
    lp = -torch.log(dus) - (torch.log(dus / dxe) - me) ** 2 / (2 * se**2)
    lk = torch.where(pos, lp, torch.where(du.abs() <= 1e-9, torch.full_like(du, math.log(eps0)),
                                          torch.full_like(du, math.log(delta))))  # fmt: skip
    lk = lk.masked_fill(~valid[:, 1:, None, :], -math.inf)
    return lk - torch.logsumexp(lk, -1, keepdim=True)


def forward(logC: torch.Tensor, logK: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (log α [P, n, C], log Z [P])."""
    la = [logC[:, 0]]
    for a in range(logK.shape[1]):
        la.append(logC[:, a + 1] + torch.logsumexp(la[-1][..., None] + logK[:, a], -2))
    la = torch.stack(la, 1)
    return la, torch.logsumexp(la[:, -1], -1)


def _draw(logits: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [..., C] -> (index [...], log probability [...])."""
    lp = logits - torch.logsumexp(logits, -1, keepdim=True)
    sh = lp.shape[:-1]
    idx = torch.multinomial(lp.reshape(-1, lp.shape[-1]).exp(), 1, generator=g)[:, 0].reshape(sh)
    return idx, lp.gather(-1, idx[..., None])[..., 0]


def sample(la: torch.Tensor, logK: torch.Tensor, K: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward sampling: -> (paths [P, K, n] candidate indices, log q [P, K])."""
    P, n, C = la.shape
    c, lq = _draw(la[:, -1, None, :].expand(P, K, C), g)
    path = [c]
    for a in range(n - 2, -1, -1):
        lk = logK[:, a].gather(-1, path[-1][:, :, None].expand(P, K, C).transpose(1, 2)).transpose(1, 2)  # [P, K, i]
        c, lc = _draw(la[:, a, None, :] + lk, g)
        path.append(c)
        lq = lq + lc
    return torch.stack(path[::-1], -1), lq


def log_q_of_path(path: torch.Tensor, logC: torch.Tensor, logK: torch.Tensor, logZ: torch.Tensor) -> torch.Tensor:
    """path [P, K, n] -> log q(path) [P, K] (differentiable in logC, logK)."""
    P, K, n = path.shape
    lc = logC[:, None].expand(P, K, n, logC.shape[-1]).gather(-1, path[..., None])[..., 0].sum(-1)
    lt = torch.zeros_like(lc)
    for a in range(n - 1):
        rows = logK[:, a].gather(1, path[:, :, a, None].expand(P, K, logK.shape[-1]))  # [P, K, C(j)]
        lt = lt + rows.gather(-1, path[:, :, a + 1, None])[..., 0]
    return lc + lt - logZ[:, None]
