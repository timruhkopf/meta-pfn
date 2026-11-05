"""Prior for the GMM-token propagation experiment (issue #23, notebook 18).

One task: M hypotheses u_m: [0,1]^d -> R^d (independent GP draws per output dimension, SE kernel, amplitude
ALPHA, length scale ell0 * sqrt(d)), weights w ~ Dir(1_M). Context token i at x_i carries the GMM
{(w_m, u_m(x_i) + s_i eps, s_i)}_m with its components shuffled per token. The target at a query x* is
u_{m*}(x*), m* ~ Cat(w) independently per query.
"""

from dataclasses import dataclass
from typing import Any

import torch

ALPHA = 0.1
ELL0 = (0.2, 0.4, 0.8)
P_M = (0.4, 0.3, 0.3)
M_MAX = 3
S_RANGE = (0.005, 0.05)
JIT = 1e-6  # relative jitter of the joint GP draw; the oracle adds the same variance to the latent target


@dataclass
class Batch:
    x: torch.Tensor  # [B, n, d] context locations
    w: torch.Tensor  # [B, n, M_MAX] component weights (shuffled per token; 0 for absent components)
    mu: torch.Tensor  # [B, n, M_MAX, d] component means (shuffled)
    s: torch.Tensor  # [B, n] per-token noise std
    cmask: torch.Tensor  # [B, n, M_MAX] component present (shuffled)
    xq: torch.Tensor  # [B, Q, d]
    yq: torch.Tensor  # [B, Q, d] target displacements
    # oracle-only (labelled, unshuffled)
    mu_lab: torch.Tensor  # [B, n, M_MAX, d]
    w_lab: torch.Tensor  # [B, M_MAX]
    m: torch.Tensor  # [B] number of hypotheses
    ell: torch.Tensor  # [B] true length scale
    mstar: torch.Tensor  # [B, Q] the hypothesis of each query

    def to(self, device) -> "Batch":
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def head(self, n: int) -> "Batch":
        """The same tasks with only the first n context tokens (nested context sets)."""
        kw = dict(self.__dict__)
        for k in ("x", "w", "mu", "s", "cmask", "mu_lab"):
            kw[k] = kw[k][:, :n]
        return Batch(**kw)

    def task(self, i: int) -> "Batch":
        return Batch(**{k: v[i : i + 1] for k, v in self.__dict__.items()})


def ells(d: int) -> torch.Tensor:
    return torch.tensor(ELL0, dtype=torch.float64) * d**0.5


def se_kernel(a: torch.Tensor, b: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    """a [..., N, d], b [..., K, d], ell broadcastable to [...] -> ALPHA^2 exp(-|a-b|^2 / (2 ell^2))."""
    d2 = torch.cdist(a, b).square()
    return ALPHA**2 * torch.exp(-0.5 * d2 / ell[..., None, None] ** 2)


@torch.no_grad()
def sample(B: int, n: int, d: int, Q: int, g: torch.Generator, device="cpu") -> Batch:
    """B tasks with n context tokens and Q queries; all random numbers from g (on `device`)."""
    f64: dict[str, Any] = dict(dtype=torch.float64, device=device)
    pm = torch.tensor(P_M, **f64)
    m = torch.multinomial(pm.expand(B, -1), 1, generator=g)[:, 0] + 1  # [B] in 1..M_MAX
    present = torch.arange(M_MAX, device=device)[None] < m[:, None]  # [B, M_MAX]
    gam = -torch.log(torch.rand(B, M_MAX, generator=g, **f64))  # Dir(1) via normalised Exp(1)
    w_lab = torch.where(present, gam, 0.0)
    w_lab = w_lab / w_lab.sum(-1, keepdim=True)
    ell = ells(d).to(device)[torch.randint(len(ELL0), (B,), generator=g, device=device)]

    x = torch.rand(B, n, d, generator=g, **f64)
    xq = torch.rand(B, Q, d, generator=g, **f64)
    pts = torch.cat([x, xq], 1)  # [B, n+Q, d]
    K = se_kernel(pts, pts, ell) + JIT * ALPHA**2 * torch.eye(n + Q, **f64)
    L = torch.linalg.cholesky(K)
    f = L @ torch.randn(B, n + Q, M_MAX * d, generator=g, **f64)  # [B, n+Q, M*d]
    f = f.view(B, n + Q, M_MAX, d)
    u_ctx, u_q = f[:, :n], f[:, n:]

    lo, hi = torch.log(torch.tensor(S_RANGE, **f64))
    s = torch.exp(lo + (hi - lo) * torch.rand(B, n, generator=g, **f64))
    mu_lab = u_ctx + s[..., None, None] * torch.randn(B, n, M_MAX, d, generator=g, **f64)
    mu_lab = torch.where(present[:, None, :, None], mu_lab, 0.0)

    perm = torch.argsort(torch.rand(B, n, M_MAX, generator=g, **f64), -1)  # per-token shuffle
    mu = torch.gather(mu_lab, 2, perm[..., None].expand(-1, -1, -1, d))
    w = torch.gather(w_lab[:, None].expand(-1, n, -1), 2, perm)
    cmask = torch.gather(present[:, None].expand(-1, n, -1), 2, perm)

    mstar = torch.multinomial(w_lab, Q, replacement=True, generator=g)  # [B, Q]
    yq = torch.gather(u_q, 2, mstar[..., None, None].expand(-1, -1, 1, d))[:, :, 0]

    f32 = lambda t: t.float()  # noqa: E731
    return Batch(x=f32(x), w=f32(w), mu=f32(mu), s=f32(s), cmask=cmask, xq=f32(xq), yq=f32(yq), mu_lab=f32(mu_lab),
                 w_lab=f32(w_lab), m=m, ell=f32(ell), mstar=mstar)  # fmt: skip
