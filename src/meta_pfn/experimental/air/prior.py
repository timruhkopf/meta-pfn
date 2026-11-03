"""AIR prior (docs/a2b_registration_proposal_2026-11-02.md §1, §3.1; issue #19), d = 1, batched on any device.

- g: random 2-layer tanh MLP (width 32, random input and weight scales), normalized to [0, 1] by `g_norm`:
  "minmax" (pilot 1): each function min-max normalized on a 256-point grid, so every g attains exactly 0 and 1;
  "ecdf" (pilot 2, the repo's standard, as pfn-repro's BNN prior): one global monotone map, the ECDF of raw outputs
  pooled over 4000 prior draws at uniform x, applied identically to every function (`prior_ecdf`);
- T_θ: time-1 flow of v_θ(x) = x (1 - x) Σ_j θ_j cos((j - 1) π x), θ_j = s_T / j · θ̃_j, θ̃ ~ N(0, I_J) (we work in
  the standardized θ̃ throughout, so p(θ̃) = N(0, I)); RK4 with 16 steps; the inverse integrates -v. s_T is calibrated
  per J so that E|T(x) - x| ≈ 0.13 under the prior;
- h: H0 identity, H1 affine a + b v with b ~ U[0.3, 1], a ~ U[0, 1 - b];
- z ~ Bern(ρ̄); z = 0: A uses an independent g' through the same h ∘ · ∘ T (choice C1);
- designs: X_B 128 uniform, X_A n_A uniform, X* n_q uniform;
- noise C2': y_b = g(x_b) + ε_b, y_a = h(g•(T(x_a))) + ε_a, σ_A, σ_B ~ U[0.005, 0.03] independently.
"""

import functools
import math
from dataclasses import dataclass

import torch

WIDTH, GRID = 32, 256
T_MEAN_SHIFT = 0.13
RK_STEPS = 24  # spec: 16; 24 keeps the J = 16 inverse error below 1e-4 in float32


@dataclass
class PriorCfg:
    J: int = 2
    h_kind: str = "id"  # id (H0) | affine (H1)
    rho: float = 1.0  # P(z = 1); 1.0 = phase Z0
    n_b: int = 128
    n_q: int = 32
    sig_lo: float = 0.005
    sig_hi: float = 0.03
    g_norm: str = "minmax"  # minmax | ecdf


@dataclass
class Funcs:
    """A batch of base functions g (2-layer tanh MLPs). Raw output r(x) is mapped to [0, 1] either affinely,
    (r - lo) / (hi - lo) (min-max), or, if `ecdf` is set, by the global prior ECDF (sorted raw quantiles)."""

    W1: torch.Tensor  # [P, H]
    b1: torch.Tensor  # [P, H]
    W2: torch.Tensor  # [P, H]
    lo: torch.Tensor  # [P]
    hi: torch.Tensor  # [P]
    ecdf: torch.Tensor | None = None  # [S] sorted raw quantiles, shared by all functions

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x[..., None] * self.W1[:, None] + self.b1[:, None]) @ self.W2[..., None])[..., 0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [P, M] -> g(x) [P, M]."""
        r = self.raw(x)
        if self.ecdf is None:
            return (r - self.lo[:, None]) / (self.hi - self.lo)[:, None]
        return ecdf_map(r, self.ecdf)

    def at(self, idx) -> "Funcs":
        return Funcs(self.W1[idx], self.b1[idx], self.W2[idx], self.lo[idx], self.hi[idx], self.ecdf)


def ecdf_map(r: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear ECDF on the sorted quantiles `ref` (as pfn-repro's BNNPrior.evaluate), clamped to [0, 1]."""
    S = ref.shape[0]
    idx = torch.searchsorted(ref, r.contiguous(), right=True).clamp(1, S - 1)
    lo, hi = ref[idx - 1], ref[idx]
    frac = ((r - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return ((idx.to(r.dtype) - 1 + frac) / (S - 1)).clamp(0, 1)


def _weights(P: int, g: torch.Generator, device):
    r = lambda *s: torch.rand(*s, generator=g, device=device)  # noqa: E731
    n = lambda *s: torch.randn(*s, generator=g, device=device)  # noqa: E731
    a = 2 + 10 * r(P, 1)  # input scale: frequency content
    W1 = n(P, WIDTH) * a
    b1 = (2 * r(P, WIDTH) - 1) * a
    W2 = n(P, WIDTH) * (0.5 + r(P, 1))
    return W1, b1, W2


@functools.lru_cache(maxsize=4)
def prior_ecdf(device: str = "cpu", n_funcs: int = 4000, n_x: int = 256, n_q: int = 1001, seed: int = 20261103):
    """Global ECDF of raw MLP outputs, pooled over `n_funcs` prior draws at `n_x` uniform x each; fixed seed, CPU, so
    every device and run uses the same map. -> sorted quantiles [n_q]."""
    g = torch.Generator().manual_seed(seed)
    W1, b1, W2 = _weights(n_funcs, g, "cpu")
    z = torch.zeros(n_funcs)
    raw = Funcs(W1, b1, W2, z, z + 1).raw(torch.rand(n_funcs, n_x, generator=g)).flatten().sort().values
    return raw[torch.linspace(0, raw.numel() - 1, n_q).long()].to(device)


def draw_funcs(P: int, g: torch.Generator, device, norm: str = "minmax") -> Funcs:
    W1, b1, W2 = _weights(P, g, device)
    z = torch.zeros(P, device=device)
    if norm == "ecdf":
        return Funcs(W1, b1, W2, z, z + 1, prior_ecdf(str(device)))
    if norm != "minmax":
        raise ValueError(f"g_norm must be minmax or ecdf, got {norm}")
    v = Funcs(W1, b1, W2, z, z + 1).raw(torch.linspace(0, 1, GRID, device=device).expand(P, -1))
    lo, hi = v.min(1).values, v.max(1).values
    return Funcs(W1, b1, W2, lo, torch.maximum(hi, lo + 1e-6))


# ---- transform -------------------------------------------------------------------------------------------------------


def velocity(x: torch.Tensor, theta: torch.Tensor, s_t: float) -> torch.Tensor:
    """x [..., M], theta [..., J] standardized -> v(x) [..., M]."""
    J = theta.shape[-1]
    j = torch.arange(J, device=x.device, dtype=x.dtype)
    coef = theta * (s_t / (j + 1))  # [..., J]
    phi = torch.cos(j * math.pi * x[..., None])  # [..., M, J]
    return x * (1 - x) * (phi * coef[..., None, :]).sum(-1)


def flow(x: torch.Tensor, theta: torch.Tensor, s_t: float, inverse: bool = False, steps: int = RK_STEPS):
    """Time-1 flow (RK4) of ±v_θ: x [..., M], theta [..., J] -> T_θ(x) or T_θ⁻¹(x)."""
    sgn = -1.0 if inverse else 1.0
    h = 1.0 / steps
    f = lambda z: sgn * velocity(z, theta, s_t)  # noqa: E731
    for _ in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@functools.lru_cache(maxsize=16)
def s_t_for(J: int, target: float = T_MEAN_SHIFT, n: int = 4000) -> float:
    """Calibrate s_T so that E|T(x) - x| ≈ target under the prior (bisection, fixed seed, CPU)."""
    g = torch.Generator().manual_seed(J)
    th = torch.randn(n, J, generator=g, dtype=torch.float64)
    x = torch.rand(n, 32, generator=g, dtype=torch.float64)
    lo, hi = 0.01, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = float((flow(x, th, mid) - x).abs().mean())
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return 0.5 * (lo + hi)


def log_prior(theta: torch.Tensor) -> torch.Tensor:
    """log N(θ̃; 0, I) [...]."""
    J = theta.shape[-1]
    return -0.5 * (theta**2).sum(-1) - 0.5 * J * math.log(2 * math.pi)


def log_prior_s(theta: torch.Tensor, sig) -> torch.Tensor:
    """log N(θ_c; 0, (1 + Σ) I): the marginal of θ_c = θ̃ + √Σ ε (pilot 3, Rao-Blackwellised latent). sig broadcastable
    to theta[..., 0] (a float or a tensor); sig = 0 is `log_prior`."""
    v = 1.0 + torch.as_tensor(sig, dtype=theta.dtype, device=theta.device)
    J = theta.shape[-1]
    return log_prior(theta / v.sqrt()[..., None]) - 0.5 * J * torch.log(v)


# ---- tasks -----------------------------------------------------------------------------------------------------------


@dataclass
class Tasks:
    xb: torch.Tensor  # [P, N_B]
    yb: torch.Tensor
    xa: torch.Tensor  # [P, n]
    ya: torch.Tensor
    xq: torch.Tensor  # [P, Q]
    yq: torch.Tensor
    fa: torch.Tensor  # [P, n]  noise-free A values (soft targets)
    fq: torch.Tensor  # [P, Q]
    sig_a: torch.Tensor  # [P]
    theta: torch.Tensor  # [P, J] standardized
    s_t: float
    hp: torch.Tensor  # [P, 2] (a, b); identity (0, 1)
    z: torch.Tensor  # [P] bool
    g: Funcs  # B's function
    ga: Funcs  # A's base function (g if z else g')

    def sub(self, idx) -> "Tasks":
        f = lambda t: t[idx]  # noqa: E731
        return Tasks(f(self.xb), f(self.yb), f(self.xa), f(self.ya), f(self.xq), f(self.yq), f(self.fa), f(self.fq),
                     f(self.sig_a), f(self.theta), self.s_t, f(self.hp), f(self.z), self.g.at(idx),
                     self.ga.at(idx))  # fmt: skip

    def head(self, n: int) -> "Tasks":
        """A's first n points (nested designs)."""
        return Tasks(self.xb, self.yb, self.xa[:, :n], self.ya[:, :n], self.xq, self.yq, self.fa[:, :n], self.fq,
                     self.sig_a, self.theta, self.s_t, self.hp, self.z, self.g, self.ga)  # fmt: skip

    def b_in_a(self, theta: torch.Tensor) -> torch.Tensor:
        """T_θ⁻¹(x_b) for θ [P, J] (or [P, K, J] -> [P, K, N_B])."""
        xb = self.xb if theta.dim() == 2 else self.xb[:, None].expand(-1, theta.shape[1], -1)
        return flow(xb, theta, self.s_t, inverse=True)


def h_apply(v: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
    return hp[:, :1] + hp[:, 1:] * v


def sample_tasks(cfg: PriorCfg, P: int, n_a: int, g: torch.Generator, device) -> Tasks:
    r = lambda *s: torch.rand(*s, generator=g, device=device)  # noqa: E731
    s_t = s_t_for(cfg.J)
    f, f2 = draw_funcs(P, g, device, cfg.g_norm), draw_funcs(P, g, device, cfg.g_norm)
    theta = torch.randn(P, cfg.J, generator=g, device=device)
    z = r(P) < cfg.rho
    if cfg.h_kind == "affine":
        b = 0.3 + 0.7 * r(P, 1)
        hp = torch.cat([(1 - b) * r(P, 1), b], 1)
    else:
        hp = torch.tensor([[0.0, 1.0]], device=device).expand(P, 2).clone()
    sig_a = cfg.sig_lo + (cfg.sig_hi - cfg.sig_lo) * r(P)
    sig_b = cfg.sig_lo + (cfg.sig_hi - cfg.sig_lo) * r(P)
    xb, xa, xq = r(P, cfg.n_b), r(P, n_a), r(P, cfg.n_q)
    pick = [torch.where(z[:, None] if t.dim() == 2 else z, t, t2) for t, t2 in
            zip((f.W1, f.b1, f.W2, f.lo, f.hi), (f2.W1, f2.b1, f2.W2, f2.lo, f2.hi), strict=True)]  # fmt: skip
    ga = Funcs(*pick, f.ecdf)
    t_aq = flow(torch.cat([xa, xq], 1), theta, s_t)
    f_aq = h_apply(ga(t_aq), hp)
    yb = f(xb) + sig_b[:, None] * torch.randn(P, cfg.n_b, generator=g, device=device)
    y_aq = f_aq + sig_a[:, None] * torch.randn(P, n_a + cfg.n_q, generator=g, device=device)
    return Tasks(xb, yb, xa, y_aq[:, :n_a], xq, y_aq[:, n_a:], f_aq[:, :n_a], f_aq[:, n_a:], sig_a, theta, s_t, hp, z,
                 f, ga)  # fmt: skip
