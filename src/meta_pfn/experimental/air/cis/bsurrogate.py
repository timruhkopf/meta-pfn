"""B surrogate for the proposal (spec §3.1): a per-task GP on B with an RBF kernel; lengthscale from {0.02, 0.05,
0.1, 0.2} and noise variance from a log grid on [0.005², 0.03²] by marginal likelihood; signal variance = var(y_B),
constant mean = mean(y_B). Gives μ_B(u), σ_B²(u) (latent, without noise) and μ_B′(u) in closed form. Proposal side
only: its errors cost ESS, never bias.

`Exact` wraps a known function (the true g) in the same interface, with σ_B = 0 (dense-B toys and tests)."""

import math
from dataclasses import dataclass

import torch

ELLS = (0.02, 0.05, 0.1, 0.2)
NOISES = tuple(float(v) for v in torch.logspace(math.log10(0.005**2), math.log10(0.03**2), 6))


@dataclass
class GP:
    X: torch.Tensor  # [P, N]
    alpha: torch.Tensor  # [P, N]  K⁻¹ (y − m)
    L: torch.Tensor  # [P, N, N] Cholesky of K
    ell: torch.Tensor  # [P]
    s2: torch.Tensor  # [P]
    m: torch.Tensor  # [P]
    noise: torch.Tensor  # [P]

    def _k(self, u: torch.Tensor) -> torch.Tensor:
        """u [P, M] -> k(u, X) [P, M, N]."""
        d = u[..., None] - self.X[:, None]
        return self.s2[:, None, None] * torch.exp(-0.5 * (d / self.ell[:, None, None]) ** 2)

    def _flat(self, u):
        return u.reshape(u.shape[0], -1)

    def mu(self, u: torch.Tensor) -> torch.Tensor:
        """u [P, ...] -> μ_B(u) [P, ...]."""
        uf = self._flat(u)
        return (self.m[:, None] + (self._k(uf) * self.alpha[:, None]).sum(-1)).reshape(u.shape)

    def var(self, u: torch.Tensor) -> torch.Tensor:
        uf = self._flat(u)
        k = self._k(uf)
        v = torch.linalg.solve_triangular(self.L, k.transpose(1, 2), upper=False)  # [P, N, M]
        return (self.s2[:, None] - (v**2).sum(1)).clamp_min(1e-10).reshape(u.shape)

    def dmu(self, u: torch.Tensor) -> torch.Tensor:
        uf = self._flat(u)
        d = uf[..., None] - self.X[:, None]
        dk = -self._k(uf) * d / self.ell[:, None, None] ** 2
        return (dk * self.alpha[:, None]).sum(-1).reshape(u.shape)

    def rows(self, idx) -> "GP":
        """A light copy for per-row means (no Cholesky factor; var() unavailable)."""
        return GP(self.X[idx], self.alpha[idx], self.L[:1].expand(len(idx), -1, -1), self.ell[idx], self.s2[idx],
                  self.m[idx], self.noise[idx])  # fmt: skip

    def at(self, idx) -> "GP":
        return GP(self.X[idx], self.alpha[idx], self.L[idx], self.ell[idx], self.s2[idx], self.m[idx], self.noise[idx])


@torch.no_grad()
def fit(xb: torch.Tensor, yb: torch.Tensor) -> GP:
    """xb, yb [P, N] -> the GP with the best (ℓ, noise) per task by marginal likelihood."""
    P, N = xb.shape
    dt = torch.float64
    X, y = xb.to(dt), yb.to(dt)
    m = y.mean(1)
    s2 = y.var(1).clamp_min(1e-4)
    yc = y - m[:, None]
    d2 = (X[:, :, None] - X[:, None]) ** 2
    best, arg = torch.full((P,), -math.inf, dtype=dt, device=xb.device), None
    eye = torch.eye(N, dtype=dt, device=xb.device)
    for ell in ELLS:
        Kb = s2[:, None, None] * torch.exp(-0.5 * d2 / ell**2)
        for nz in NOISES:
            L, info = torch.linalg.cholesky_ex(Kb + nz * eye)
            a = torch.cholesky_solve(yc[..., None], L)[..., 0]
            lml = (
                -0.5 * (yc * a).sum(1)
                - torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
                - 0.5 * N * math.log(2 * math.pi)
            )
            lml = torch.where(info == 0, lml, torch.full_like(lml, -math.inf))
            better = lml > best
            best = torch.where(better, lml, best)
            if arg is None:
                arg = (
                    torch.full((P,), ell, dtype=dt, device=xb.device),
                    torch.full((P,), nz, dtype=dt, device=xb.device),
                )
            arg = (torch.where(better, ell, arg[0]), torch.where(better, nz, arg[1]))
    ell, nz = arg
    K = s2[:, None, None] * torch.exp(-0.5 * d2 / ell[:, None, None] ** 2) + nz[:, None, None] * eye
    L = torch.linalg.cholesky(K)
    a = torch.cholesky_solve(yc[..., None], L)[..., 0]
    f = lambda v: v.to(xb.dtype)  # noqa: E731
    return GP(xb, f(a), f(L), f(ell), f(s2), f(m), f(nz))


@dataclass
class Exact:
    """A known function as the 'surrogate' (dense-B limit): μ = g, σ² = 0, μ′ by central differences."""

    g: object  # callable [P, M] -> [P, M]

    def mu(self, u):
        return self.g(u.reshape(u.shape[0], -1)).reshape(u.shape)

    def var(self, u):
        return torch.zeros_like(u)

    def dmu(self, u, eps: float = 1e-4):
        return (self.mu(u + eps) - self.mu(u - eps)) / (2 * eps)

    def at(self, idx):
        return Exact(self.g.at(idx))
