"""Batched task-pair prior for the A→B architecture's S3/S4 (issue #18): f_A(x) = h(f_B(T(x))), drawn on the GPU.

f ~ the BNN prior (pfn-repro, the frozen PFN's training prior and ECDF scale). B observes f at a uniform design.
A observes h(y) with y an observation of f at T(x):
- T: per-dim full-box monotone bijection of [0, 1], a Kumaraswamy CDF 1 - (1 - x^α)^β, log α, log β ~ N(0, t_sd²)
  (every A point has a correspondence; no sub-windows);
- h (`h_kind`): `id` (S3); `affine` h(u) = c + e u, e ~ U(e_min, 1), c ~ U(0, 1 - e); `kum` a Kumaraswamy CDF on y,
  log a, log b ~ N(0, h_sd²). h is applied to the noisy observation, so the oracle density is exact by change of
  variables: log p_A(y) = log p_B(h⁻¹(y)) + log |(h⁻¹)'(y)|.
Strata: `identity` (T = h = id), `warp` (T, h ~ the families), `unrelated` (B observes an independent function).
"""

from dataclasses import dataclass

import torch
from pfn_repro.checkpoints import make_prior

STRATA = ("identity", "warp", "unrelated")
_EPS = 1e-6


@dataclass
class PriorCfg:
    d: int = 1
    t_sd: float = 0.5
    h_kind: str = "id"  # id | affine | kum
    e_min: float = 0.5
    h_sd: float = 0.5


@dataclass
class Inst:
    xb: torch.Tensor  # [P, N_B, d]
    yb: torch.Tensor  # [P, N_B]
    xa: torch.Tensor  # [P, n, d]   A's train design (nested prefixes give smaller n_A)
    ya: torch.Tensor  # [P, n]
    xq: torch.Tensor  # [P, Q, d]   held-out A queries
    yq: torch.Tensor  # [P, Q]
    t_al: torch.Tensor  # [P, d]    true T (Kumaraswamy α, β per dim)
    t_be: torch.Tensor
    h_par: torch.Tensor  # [P, 2]   true h: (c, e) for affine, (a, b) for kum, (0, 1) for id
    h_kind: str
    related: torch.Tensor  # [P] bool


def kum(x: torch.Tensor, al: torch.Tensor, be: torch.Tensor) -> torch.Tensor:
    u = x.clamp(_EPS, 1 - _EPS)
    return 1 - (1 - u**al) ** be


def kum_inv(y: torch.Tensor, al: torch.Tensor, be: torch.Tensor) -> torch.Tensor:
    v = y.clamp(_EPS, 1 - _EPS)
    return (1 - (1 - v) ** (1 / be)) ** (1 / al)


def apply_t(x: torch.Tensor, al: torch.Tensor, be: torch.Tensor) -> torch.Tensor:
    """x [P, M, d], al/be [P, d] -> T(x) [P, M, d]."""
    return kum(x, al[:, None], be[:, None])


def h_fwd(u: torch.Tensor, kind: str, par: torch.Tensor) -> torch.Tensor:
    """u [P, M] -> h(u)."""
    if kind == "id":
        return u
    p0, p1 = par[:, :1], par[:, 1:]
    return p0 + p1 * u if kind == "affine" else kum(u, p0, p1)


def h_inv_logjac(y: torch.Tensor, kind: str, par: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y [P, M] -> (h⁻¹(y) clipped to [0, 1], log |(h⁻¹)'(y)|); outside h's range the density is 0 (-inf)."""
    if kind == "id":
        return y, torch.zeros_like(y)
    p0, p1 = par[:, :1], par[:, 1:]
    if kind == "affine":
        u = (y - p0) / p1
        lj = -p1.log().expand_as(y)
        return u.clamp(0, 1), torch.where((u >= 0) & (u <= 1), lj, torch.full_like(y, -torch.inf))
    a, b = p0, p1
    v = y.clamp(_EPS, 1 - _EPS)
    w = 1 - (1 - v) ** (1 / b)  # u = w^(1/a)
    u = w ** (1 / a)
    # du/dy = (1/a) w^(1/a - 1) * (1/b) (1 - v)^(1/b - 1)
    lj = (1 / a - 1) * w.log() - a.log() + (1 / b - 1) * (1 - v).log() - b.log()
    return u, lj


class PairPrior:
    """Fresh functions per `sample`; deterministic given the seed and the call sequence."""

    def __init__(self, cfg: PriorCfg, batch: int, seed: int, device: str):
        self.cfg, self.P, self.dev = cfg, batch, device
        self.f = make_prior(x_dim=cfg.d, batch_size=batch, seed=seed, device=device)
        self.f_other = make_prior(x_dim=cfg.d, batch_size=batch, seed=seed + 7_777_777, device=device)
        self.g = torch.Generator(device=device).manual_seed(seed + 1)

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.dev, generator=self.g)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, device=self.dev, generator=self.g)

    def sample(self, stratum: str, n_a: int, n_q: int, n_b: int) -> Inst:
        c, P, d = self.cfg, self.P, self.cfg.d
        self.f.reset()
        xb, xa, xq = self._rand(P, n_b, d), self._rand(P, n_a, d), self._rand(P, n_q, d)
        al, be = (self._randn(P, d) * c.t_sd).exp(), (self._randn(P, d) * c.t_sd).exp()
        if c.h_kind == "affine":
            e = c.e_min + (1 - c.e_min) * self._rand(P, 1)
            hp = torch.cat([(1 - e) * self._rand(P, 1), e], 1)
        elif c.h_kind == "kum":
            hp = (self._randn(P, 2) * c.h_sd).exp()
        else:
            hp = torch.tensor([[0.0, 1.0]], device=self.dev).expand(P, 2)
        if stratum == "identity":  # T = id; h = the identity member of its family
            al, be = torch.ones_like(al), torch.ones_like(be)
            hp = torch.tensor([[1.0, 1.0]] if c.h_kind == "kum" else [[0.0, 1.0]], device=self.dev).expand(P, 2)
        hk = c.h_kind
        with torch.no_grad():
            y_aq = self.f.evaluate(apply_t(torch.cat([xa, xq], 1), al, be), noise=True)
            if stratum == "unrelated":
                self.f_other.reset()
                yb = self.f_other.evaluate(xb, noise=True)
            else:
                yb = self.f.evaluate(xb, noise=True)
        y_aq = h_fwd(y_aq, hk, hp).clamp(0, 1)
        rel = torch.full((P,), stratum != "unrelated", device=self.dev)
        return Inst(xb, yb, xa, y_aq[:, :n_a], xq, y_aq[:, n_a:], al, be, hp.contiguous(), hk, rel)
