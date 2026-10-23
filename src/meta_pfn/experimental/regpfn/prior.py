"""Phase-1 task-pair prior (docs/spec_phase1_experiments.md §4): B is a distorted copy of A, f_B(u) = h(f_A(T(u))).

f_A is a pfn-repro BNN draw, evaluated raw (unnormalized) so that it extends smoothly beyond [0, 1]^d: the
latent domain U is all of R^d and T may move B's window partly outside A's. Both tasks sample x ∈ [0, 1]^d.
- T (input, `t_on`):  T(u) = warp(scale(shift(u))): shift s ~ U[-s_max, s_max]^d; per-dim scale about 0.5,
  a ~ logU[a_min, a_max]; per-dim Kumaraswamy warp on [0, 1], identity outside (log α, log β ~ N(0, warp_sd²)).
- h (output, `h_on`): on A's normalized value v ∈ [0, 1]: t = 4 (v - 0.5), h(t) = α sinh(δ asinh(t) - ε) + β
  (affine + sinh-arcsinh warp; monotone, closed-form inverse). α only matters before B's normalization.
- Normalization per task (§4.4): an affine map of the noise-free function over its own window [0, 1]^d (min/max
  on a dense grid) onto [margin, 1 - margin]; observation noise σ_ε (per pair, logU) is added after it and y is
  clipped to [0, 1], the bar distribution's support.
- Oracle correspondence: B_inA = (T(u), norm_A(h⁻¹(norm_B⁻¹(y_B)))), `inside` = T(u) ∈ [0, 1]^d.
All tensors are batched [P, ...] (one function per pair), on `device`.
"""

from dataclasses import dataclass

import torch
from pfn_repro.checkpoints import make_prior

CELLS = {"C00": (False, False), "C10": (True, False), "C01": (False, True), "C11": (True, True)}


@dataclass
class PriorCfg:
    d: int = 1
    s_max: float = 0.3
    a_min: float = 0.7
    a_max: float = 1.4
    warp_sd: float = 0.35  # "moderate" warp
    h_delta: tuple[float, float] = (0.5, 2.0)  # logU range of the sinh-arcsinh tail parameter δ
    h_eps: float = 1.0  # ε ~ U[-h_eps, h_eps]
    margin: float = 0.1
    noise: tuple[float, float] = (1e-3, 3e-2)  # logU range of σ_ε on the normalized scale
    n_grid: tuple[int, ...] = (512, 64)  # normalization grid points per axis, for d = 1, 2


@dataclass
class Batch:
    xa: torch.Tensor  # [P, n_A + n_test, d]  A's design (context first, then queries)
    ya: torch.Tensor  # [P, n_A + n_test]
    ub: torch.Tensor  # [P, N_B, d]          B's design in B's own coordinates
    yb: torch.Tensor  # [P, N_B]
    xb_inA: torch.Tensor  # [P, N_B, d]      privileged: B's points in A's frame, T(u)
    yb_inA: torch.Tensor  # [P, N_B]         privileged: B's values on A's scale
    inside: torch.Tensor  # [P, N_B] bool    T(u) ∈ [0, 1]^d
    fa_test: torch.Tensor  # [P, n_A + n_test] noise-free f_A at xa (normalized)
    noise: torch.Tensor  # [P]
    n_a: int


def _logu(lo: float, hi: float, shape, g, dev) -> torch.Tensor:
    return torch.exp(
        torch.empty(shape, device=dev).uniform_(
            float(torch.log(torch.tensor(lo))), float(torch.log(torch.tensor(hi))), generator=g
        )
    )


def _kum(v: torch.Tensor, al: torch.Tensor, be: torch.Tensor) -> torch.Tensor:
    inside = (v > 0) & (v < 1)
    vc = v.clamp(1e-6, 1 - 1e-6)
    return torch.where(inside, 1 - (1 - vc**al) ** be, v)


def _grid(d: int, k: int, dev) -> torch.Tensor:
    """A k^d grid of [0, 1]^d -> [k^d, d]."""
    axes = torch.meshgrid(*[torch.linspace(0, 1, k, device=dev)] * d, indexing="ij")
    return torch.stack([a.reshape(-1) for a in axes], -1)


class PairPrior:
    """Draws batches of (A, B, privileged) for one factorial cell."""

    def __init__(self, cfg: PriorCfg, batch_size: int, seed: int, device: str = "cpu"):
        self.cfg, self.P, self.dev = cfg, batch_size, device
        self.bnn = make_prior(x_dim=cfg.d, batch_size=batch_size, seed=seed, device=device)
        self.g = torch.Generator(device=device).manual_seed(seed)

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.bnn._raw_forward(x, noise=False)

    def sample(self, cell: str, n_a: int, n_test: int, n_b: int) -> Batch:
        c, P, d, dev, g = self.cfg, self.P, self.cfg.d, self.dev, self.g
        t_on, h_on = CELLS[cell]
        self.bnn.reset()
        rand = lambda *s: torch.rand(*s, device=dev, generator=g)  # noqa: E731
        xa, ub = rand(P, n_a + n_test, d), rand(P, n_b, d)
        grid = _grid(d, c.n_grid[d - 1], dev)[None].expand(P, -1, -1)
        # T
        if t_on:
            s = (2 * rand(P, 1, d) - 1) * c.s_max
            a = _logu(c.a_min, c.a_max, (P, 1, d), g, dev)
            al = torch.exp(c.warp_sd * torch.randn(P, 1, d, device=dev, generator=g))
            be = torch.exp(c.warp_sd * torch.randn(P, 1, d, device=dev, generator=g))
            T = lambda u: _kum(0.5 + a * (u + s - 0.5), al, be)  # noqa: E731
        else:
            T = lambda u: u  # noqa: E731
        # h on A's normalized scale
        if h_on:
            dl = _logu(*c.h_delta, (P, 1), g, dev)
            ep = (2 * rand(P, 1) - 1) * c.h_eps
            alpha = _logu(0.5, 2.0, (P, 1), g, dev)
            beta = 2 * rand(P, 1) - 1
            h = lambda v: alpha * torch.sinh(dl * torch.asinh(4 * (v - 0.5)) - ep) + beta  # noqa: E731
            h_inv = lambda w: torch.sinh((torch.asinh((w - beta) / alpha) + ep) / dl) / 4 + 0.5  # noqa: E731
        else:
            h = h_inv = lambda v: v
        # A: normalized by its own window
        fa_grid = self._f(grid)
        lo_a, hi_a = fa_grid.min(1, keepdim=True).values, fa_grid.max(1, keepdim=True).values
        norm_a = lambda f: (f - lo_a) / (hi_a - lo_a)  # noqa: E731  -> [0, 1] on the window
        va = norm_a(self._f(xa))
        # B: f_B(u) = h(f_A(T(u))), normalized by its own window
        hb_grid = h(norm_a(self._f(T(grid))))
        lo_b, hi_b = hb_grid.min(1, keepdim=True).values, hb_grid.max(1, keepdim=True).values
        tb = T(ub)
        wb = (h(norm_a(self._f(tb))) - lo_b) / (hi_b - lo_b)
        m = c.margin
        noise = _logu(*c.noise, (P,), g, dev)
        ya = (m + (1 - 2 * m) * va + noise[:, None] * torch.randn(va.shape, device=dev, generator=g)).clamp(0, 1)
        yb = (m + (1 - 2 * m) * wb + noise[:, None] * torch.randn(wb.shape, device=dev, generator=g)).clamp(0, 1)
        # privileged: B in A's frame (with the noise of y_B pushed through the inverse maps)
        w_obs = lo_b + (yb - m) / (1 - 2 * m) * (hi_b - lo_b)
        yb_inA = m + (1 - 2 * m) * h_inv(w_obs)
        inside = ((tb >= 0) & (tb <= 1)).all(-1)
        return Batch(xa, ya, ub, yb, tb, yb_inA, inside, m + (1 - 2 * m) * va, noise, n_a)
