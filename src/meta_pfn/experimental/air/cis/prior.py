"""The AIR-CIS prior (spec §1 with deviations D1, D2, D5, D6 of the pre-run addendum), d = 1, batched on any device.

- g: tanh BNN with a sampled architecture, depth ∈ {1, 2, 3}, width ∈ {16, 32, 64} (the first w of 64 units);
  layer 1: input scale a ~ U[2, 12], bias U[−a, a]; later layers N(0, 1)/√fan-in weights, N(0, 1) biases; output
  N(0, 1)/√fan-in times 0.5 + U[0, 1]; then one global ECDF pooled over the architecture prior (`prior_ecdf`).
- T(x) = c + w Φ(x): Φ the time-1 flow (RK4, 24 steps) of v(x) = x(1 − x) Σ_{j<J} θ_j cos(jπx), θ_j = s_T θ̃_j/(j+1),
  J ∈ {0, 2, 4, 8, 16} with p(J) = (0.2, 0.3, 0.2, 0.15, 0.15) (J = 0: Φ = id), log s_T ~ N(log s₀(J), 0.5²) with s₀(J)
  calibrated so that E|Φ(x) − x| ≈ 0.13 (D5); window w = 0.5 + 0.5 Φ_N(z_w), c = (1 − w) Φ_N(z_c) (D1: no boundary
  leak). The transform latent is lat = (θ̃_0..θ̃_15, z_s, z_w, z_c) ~ N(0, I) on its active entries (θ̃_{<J}, z_s if
  J > 0, z_w, z_c); inactive entries are 0.
- h: identity with probability PI_ID, else a + b v, b ~ U[0.3, 1], a ~ U[0, 1 − b] (D2).
- y_b = g(x_b) + ε_b, y_a = h(g(T(x_a))) + ε_a, σ_A, σ_B ~ U[0.005, 0.03]; always related; designs uniform.
- B in A's frame: T⁻¹(x_b) = Φ̄⁻¹((x_b − c)/w), Φ̄ = Φ on [0, 1] and the identity outside; the decoder's T′ feature
  is log (T⁻¹)′(x_b) (the log-derivative integrated along the inverse flow, minus log w).
"""

import functools
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from ..prior import ecdf_map

J_VALUES = (0, 2, 4, 8, 16)
P_J = (0.2, 0.3, 0.2, 0.15, 0.15)
J_MAX = 16
ZS, ZW, ZC = J_MAX, J_MAX + 1, J_MAX + 2
N_LAT = J_MAX + 3
WIDTHS, DEPTHS, H_MAX = (16, 32, 64), (1, 2, 3), 64
T_MEAN_SHIFT, LOG_S_SD, W_LO = 0.13, 0.5, 0.5
PI_ID, B_LO = 0.5, 0.3
RK_STEPS = 24
N_A_MAX, N_A_EVAL = 10, (1, 2, 3, 5, 8, 10)
TABLES = Path("results/air_cis")


@dataclass
class CisCfg:
    n_b: int = 128
    n_q: int = 32
    sig_lo: float = 0.005
    sig_hi: float = 0.03
    pi_id: float = PI_ID


# ---- g: the BNN with a sampled architecture -------------------------------------------------------------------------


@dataclass
class BNN:
    a: torch.Tensor  # [P, 1] input scale
    W1: torch.Tensor  # [P, H]
    b1: torch.Tensor  # [P, H]
    W2: torch.Tensor  # [P, H, H]
    b2: torch.Tensor  # [P, H]
    W3: torch.Tensor  # [P, H, H]
    b3: torch.Tensor  # [P, H]
    Wo: torch.Tensor  # [P, H]
    mask: torch.Tensor  # [P, H] first `width` units
    depth: torch.Tensor  # [P] long
    ecdf: torch.Tensor  # [S]

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """x [P, M] (or [P, K, M]) -> raw output, same shape."""
        sh = x.shape
        xf = x.reshape(sh[0], -1)
        m = self.mask[:, None]
        h1 = torch.tanh(xf[..., None] * (self.a[:, None] * self.W1[:, None]) + self.b1[:, None]) * m
        h2 = torch.tanh(torch.einsum("pmh,phk->pmk", h1, self.W2) + self.b2[:, None]) * m
        h3 = torch.tanh(torch.einsum("pmh,phk->pmk", h2, self.W3) + self.b3[:, None]) * m
        d = self.depth[:, None, None]
        h = torch.where(d == 1, h1, torch.where(d == 2, h2, h3))
        return (h * self.Wo[:, None]).sum(-1).reshape(sh)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ecdf_map(self.raw(x), self.ecdf)

    def at(self, idx) -> "BNN":
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        return BNN(**{k: (v if k == "ecdf" else v[idx]) for k, v in kw.items()})

    def to(self, device) -> "BNN":
        return BNN(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def _bnn_weights(P: int, g: torch.Generator, device, ecdf: torch.Tensor) -> BNN:
    r = lambda *s: torch.rand(*s, generator=g, device=device)  # noqa: E731
    n = lambda *s: torch.randn(*s, generator=g, device=device)  # noqa: E731
    width = torch.tensor(WIDTHS, device=device)[torch.randint(0, 3, (P,), generator=g, device=device)]
    depth = torch.tensor(DEPTHS, device=device)[torch.randint(0, 3, (P,), generator=g, device=device)]
    mask = (torch.arange(H_MAX, device=device)[None] < width[:, None]).float()
    fan = width.float().sqrt()[:, None]
    a = 2 + 10 * r(P, 1)
    W1, b1 = n(P, H_MAX), (2 * r(P, H_MAX) - 1) * a
    W2, b2 = n(P, H_MAX, H_MAX) / fan[..., None], n(P, H_MAX)
    W3, b3 = n(P, H_MAX, H_MAX) / fan[..., None], n(P, H_MAX)
    Wo = n(P, H_MAX) / fan * (0.5 + r(P, 1))
    return BNN(a, W1, b1, W2, b2, W3, b3, Wo, mask, depth, ecdf.to(device))


@functools.lru_cache(maxsize=1)
def prior_ecdf(n_funcs: int = 4000, n_x: int = 256, n_q: int = 1001, seed: int = 20261104) -> torch.Tensor:
    """The ECDF of raw outputs pooled over the architecture prior (fixed seed, CPU); cached in
    results/air_cis/ecdf_bnnarch.pt (recomputed, and checked, if the file is missing)."""
    path = TABLES / "ecdf_bnnarch.pt"
    if path.exists():
        return torch.load(path)
    g = torch.Generator().manual_seed(seed)
    f = _bnn_weights(n_funcs, g, "cpu", torch.zeros(2))
    raw = f.raw(torch.rand(n_funcs, n_x, generator=g)).flatten().sort().values
    q = raw[torch.linspace(0, raw.numel() - 1, n_q).long()].contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q, path)
    return q


def draw_g(P: int, g: torch.Generator, device) -> BNN:
    return _bnn_weights(P, g, device, prior_ecdf())


def n_extrema(gv: np.ndarray, prom: float = 0.01) -> tuple[int, list[int]]:
    """Interior extrema of one function on a dense grid (values gv [G]), ignoring wiggles smaller than `prom`
    (zigzag with hysteresis). -> (count, grid indices of the extrema)."""
    ext: list[int] = []
    hi = lo = float(gv[0])
    hi_i = lo_i = 0
    direction = 0
    for i in range(1, len(gv)):
        v = float(gv[i])
        if v > hi:
            hi, hi_i = v, i
        if v < lo:
            lo, lo_i = v, i
        if direction == 0:
            if hi - gv[0] > prom and hi - v > prom:
                ext.append(hi_i)
                direction, lo, lo_i = -1, v, i
            elif gv[0] - lo > prom and v - lo > prom:
                ext.append(lo_i)
                direction, hi, hi_i = 1, v, i
            elif hi - gv[0] > prom:
                direction = 1
            elif gv[0] - lo > prom:
                direction = -1
        elif direction == 1 and hi - v > prom:
            ext.append(hi_i)
            direction, lo, lo_i = -1, v, i
        elif direction == -1 and v - lo > prom:
            ext.append(lo_i)
            direction, hi, hi_i = 1, v, i
    return len(ext), ext


def branch_label(ext_x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The monotone segment of g containing u: the number of extrema (positions ext_x [E]) left of u."""
    return (u[..., None] > ext_x).sum(-1)


# ---- the transform --------------------------------------------------------------------------------------------------


def _coef(th: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    j = torch.arange(th.shape[-1], device=th.device, dtype=th.dtype)
    return th * s[..., None] / (j + 1)


def _v_and_dv(x: torch.Tensor, coef: torch.Tensor):
    """v(x) and v'(x): x [..., M], coef [..., J]."""
    j = torch.arange(coef.shape[-1], device=x.device, dtype=x.dtype) * math.pi
    ang = j * x[..., None]
    c = coef[..., None, :]
    S = (torch.cos(ang) * c).sum(-1)
    dS = (-torch.sin(ang) * j * c).sum(-1)
    return x * (1 - x) * S, (1 - 2 * x) * S + x * (1 - x) * dS


def flow(x: torch.Tensor, th: torch.Tensor, s: torch.Tensor, inverse: bool = False, steps: int = RK_STEPS,
         logdet: bool = False):  # fmt: skip
    """Φ or Φ⁻¹: x [..., M], th [..., J], s [...] (s_T) -> x' (and log |dx'/dx| if logdet)."""
    sgn = -1.0 if inverse else 1.0
    coef = sgn * _coef(th, s)
    h = 1.0 / steps
    ld = torch.zeros_like(x) if logdet else None
    for _ in range(steps):
        k1, d1 = _v_and_dv(x, coef)
        k2, d2 = _v_and_dv(x + 0.5 * h * k1, coef)
        k3, d3 = _v_and_dv(x + 0.5 * h * k2, coef)
        k4, d4 = _v_and_dv(x + h * k3, coef)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if logdet:
            ld = ld + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
    return (x, ld) if logdet else x


@functools.lru_cache(maxsize=8)
def s0_for(J: int, target: float = T_MEAN_SHIFT, n: int = 4000) -> float:
    """s₀(J) with E|Φ(x) − x| ≈ target under θ̃ ~ N(0, I), log s_T ~ N(log s₀, 0.5²) (bisection, fixed seed, CPU)."""
    if J == 0:
        return 1.0
    g = torch.Generator().manual_seed(100 + J)
    th = torch.randn(n, J, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    x = torch.rand(n, 32, generator=g, dtype=torch.float64)
    lo, hi = 0.01, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = float((flow(x, th, mid * torch.exp(LOG_S_SD * z)) - x).abs().mean())
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return 0.5 * (lo + hi)


def s0_table(device) -> torch.Tensor:
    """s₀ indexed by J (0..16)."""
    t = torch.ones(J_MAX + 1, dtype=torch.float32)
    for J in J_VALUES:
        t[J] = s0_for(J)
    return t.to(device)


def active(J: torch.Tensor) -> torch.Tensor:
    """J [...] -> bool [..., N_LAT]: the latent entries the prior is defined on."""
    i = torch.arange(N_LAT, device=J.device)
    th = i < J[..., None]
    zs = (i == ZS) & (J[..., None] > 0)
    return th | zs | (i == ZW) | (i == ZC)


def window(lat: torch.Tensor):
    w = W_LO + (1 - W_LO) * torch.special.ndtr(lat[..., ZW])
    return w, (1 - w) * torch.special.ndtr(lat[..., ZC])


def s_t(lat: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    return s0_table(lat.device)[J] * torch.exp(LOG_S_SD * lat[..., ZS])


def transform(x: torch.Tensor, lat: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """T(x) = c + w Φ(x): x [..., M], lat [..., N_LAT], J [...] -> [..., M]."""
    w, c = window(lat)
    u = flow(x, lat[..., :J_MAX], s_t(lat, J))
    return c[..., None] + w[..., None] * u


def inverse(u: torch.Tensor, lat: torch.Tensor, J: torch.Tensor):
    """T⁻¹(u) with Φ̄ (identity outside [0, 1]) and log (T⁻¹)′(u): u [..., M] -> (x̃, log-derivative)."""
    w, c = window(lat)
    z = (u - c[..., None]) / w[..., None]
    inside = (z >= 0) & (z <= 1)
    x, ld = flow(z.clamp(0, 1), lat[..., :J_MAX], s_t(lat, J), inverse=True, logdet=True)
    return torch.where(inside, x, z), torch.where(inside, ld, torch.zeros_like(ld)) - torch.log(w)[..., None]


def log_prior(lat: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """log N(0, I) density of the active latent entries."""
    a = active(J).to(lat.dtype)
    return -0.5 * ((lat**2) * a).sum(-1) - 0.5 * a.sum(-1) * math.log(2 * math.pi)


# ---- tasks ----------------------------------------------------------------------------------------------------------


@dataclass
class CisTasks:
    xb: torch.Tensor  # [P, N_B]
    yb: torch.Tensor
    xa: torch.Tensor  # [P, n]
    ya: torch.Tensor
    xq: torch.Tensor  # [P, Q]
    yq: torch.Tensor
    fa: torch.Tensor  # noise-free A values (soft targets)
    fq: torch.Tensor
    sig_a: torch.Tensor  # [P]
    sig_b: torch.Tensor
    J: torch.Tensor  # [P] long
    lat: torch.Tensor  # [P, N_LAT]
    hid: torch.Tensor  # [P] bool
    hp: torch.Tensor  # [P, 2] (a, b); identity (0, 1)
    g: BNN

    def _map(self, f, g_idx=None) -> "CisTasks":
        kw = {k.name: getattr(self, k.name) for k in fields(self)}
        for k in kw:
            if k != "g":
                kw[k] = f(kw[k])
        if g_idx is not None:
            kw["g"] = self.g.at(g_idx)
        return CisTasks(**kw)

    def sub(self, idx) -> "CisTasks":
        return self._map(lambda t: t[idx], idx)

    def head(self, n: int) -> "CisTasks":
        return replace(self, xa=self.xa[:, :n], ya=self.ya[:, :n], fa=self.fa[:, :n])

    def to(self, device) -> "CisTasks":
        return replace(self._map(lambda t: t.to(device)), g=self.g.to(device))

    def f_a(self, x: torch.Tensor) -> torch.Tensor:
        """Noise-free f_A at x [P, M]."""
        return self.hp[:, :1] + self.hp[:, 1:] * self.g(transform(x, self.lat, self.J))

    def b_in_a(self, lat: torch.Tensor | None = None, J: torch.Tensor | None = None):
        """(T⁻¹(x_b), log (T⁻¹)′(x_b)) at the true (default) or the given latents ([P, N_LAT] or [P, K, N_LAT])."""
        lat = self.lat if lat is None else lat
        J = self.J if J is None else J
        xb = self.xb if lat.dim() == 2 else self.xb[:, None].expand(-1, lat.shape[1], -1)
        return inverse(xb, lat, J)


def sample_tasks(cfg: CisCfg, P: int, n_a: int, g: torch.Generator, device=None) -> CisTasks:
    """Drawn on the generator's device, returned on `device` (default: the same)."""
    gd = g.device
    r = lambda *s: torch.rand(*s, generator=g, device=gd)  # noqa: E731
    f = draw_g(P, g, gd)
    J = torch.tensor(J_VALUES, device=gd)[torch.multinomial(torch.tensor(P_J, device=gd), P, True, generator=g)]
    lat = torch.randn(P, N_LAT, generator=g, device=gd) * active(J)
    hid = r(P) < cfg.pi_id
    b = B_LO + (1 - B_LO) * r(P, 1)
    a = (1 - b) * r(P, 1)
    hp = torch.where(hid[:, None], torch.tensor([[0.0, 1.0]], device=gd), torch.cat([a, b], 1))
    sig_a = cfg.sig_lo + (cfg.sig_hi - cfg.sig_lo) * r(P)
    sig_b = cfg.sig_lo + (cfg.sig_hi - cfg.sig_lo) * r(P)
    xb, xa, xq = r(P, cfg.n_b), r(P, n_a), r(P, cfg.n_q)
    yb = f(xb) + sig_b[:, None] * torch.randn(P, cfg.n_b, generator=g, device=gd)
    f_aq = hp[:, :1] + hp[:, 1:] * f(transform(torch.cat([xa, xq], 1), lat, J))
    y_aq = f_aq + sig_a[:, None] * torch.randn(P, n_a + cfg.n_q, generator=g, device=gd)
    t = CisTasks(xb, yb, xa, y_aq[:, :n_a], xq, y_aq[:, n_a:], f_aq[:, :n_a], f_aq[:, n_a:], sig_a, sig_b, J, lat, hid,
                 hp, f)  # fmt: skip
    return t if device is None else t.to(device)


def eval_tasks(n_tasks: int, seed: int, device, cfg: CisCfg | None = None) -> CisTasks:
    """Fixed evaluation tasks, drawn on the CPU (identical on every machine), nested in n_A (first n of 10)."""
    return sample_tasks(cfg or CisCfg(), n_tasks, N_A_MAX, torch.Generator().manual_seed(seed), device)


def extrema(t: CisTasks, G: int = 1024, prom: float = 0.01):
    """Per task: (number of extrema of g on [0, 1], list of extremum positions)."""
    x = torch.linspace(0, 1, G)
    gv = t.g.to("cpu")(x[None].expand(t.xb.shape[0], -1)).numpy()
    out = [n_extrema(v, prom) for v in gv]
    return [c for c, _ in out], [x[e] for _, e in out]


# ---- prior tables (proposal side; spec §3.2) ------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def prior_tables(n_draws: int = 100_000, n_bins: int = 256, n_dx: int = 32, seed: int = 4242) -> dict:
    """π(u | x) [x-bin, u-bin] (density in u, smoothed, floored) and the log-stretch mean / sd per Δx bin, from
    `n_draws` prior transforms (pooled over J); cached in results/air_cis/prior_tables.pt."""
    path = TABLES / "prior_tables.pt"
    if path.exists():
        return torch.load(path)
    g = torch.Generator().manual_seed(seed)
    xs = (torch.arange(n_bins) + 0.5) / n_bins
    dxs = torch.logspace(-3, math.log10(0.5), n_dx)
    hist = torch.zeros(n_bins, n_bins, dtype=torch.float64)
    m_k, s_k = torch.zeros(n_dx), torch.zeros(n_dx)
    stretch = [[] for _ in range(n_dx)]
    for _ in range(n_draws // 5000):
        P = 5000
        J = torch.tensor(J_VALUES)[torch.multinomial(torch.tensor(P_J), P, True, generator=g)]
        lat = torch.randn(P, N_LAT, generator=g) * active(J)
        u = transform(xs[None].expand(P, -1), lat, J)  # [P, n_bins]
        ub = (u * n_bins).long().clamp(0, n_bins - 1)
        hist.scatter_add_(1, ub.T.contiguous(), torch.ones(n_bins, P, dtype=torch.float64))
        x0 = torch.rand(P, n_dx, generator=g) * (1 - dxs)
        t0, t1 = transform(x0, lat, J), transform(x0 + dxs, lat, J)
        r = torch.log((t1 - t0).clamp_min(1e-12) / dxs)
        for k in range(n_dx):
            stretch[k].append(r[:, k])
    k = torch.exp(-0.5 * (torch.arange(-3, 4, dtype=torch.float64) / 1.0) ** 2)
    k = k / k.sum()
    sm = torch.nn.functional.conv1d(hist[:, None], k[None, None], padding=3)[:, 0]
    sm = sm / sm.sum(1, keepdim=True)
    sm = 0.995 * sm + 0.005 / n_bins  # floor: every u stays possible
    for i in range(n_dx):
        s = torch.cat(stretch[i])
        m_k[i], s_k[i] = s.mean(), s.std()
    out = {"log_pi": torch.log(sm * n_bins).float(), "dx": dxs, "m_k": m_k, "s_k": s_k, "n_bins": n_bins}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def log_pi(x: torch.Tensor, u: torch.Tensor, tab: dict) -> torch.Tensor:
    """log π(u | x) (density in u): x [...] broadcast against u [...]; nearest x bin, linear in u."""
    nb = tab["n_bins"]
    lp = tab["log_pi"].to(u.device)
    xi = (x * nb).long().clamp(0, nb - 1)
    z = (u.clamp(0, 1) * nb - 0.5).clamp(0, nb - 1)
    lo = z.floor().long().clamp(0, nb - 2)
    fr = z - lo
    xi = xi.expand_as(lo) if xi.shape != lo.shape else xi
    a, b = lp[xi, lo], lp[xi, lo + 1]
    return a + fr * (b - a)


def stretch_stats(dx: torch.Tensor, tab: dict):
    """(mean, sd) of log((T(x + Δx) − T(x))/Δx) at Δx [...] (interpolated in log Δx)."""
    ld = torch.log(tab["dx"].to(dx.device))
    z = (torch.log(dx.clamp_min(1e-4)) - ld[0]) / (ld[1] - ld[0])
    z = z.clamp(0, len(ld) - 1)
    lo = z.floor().long().clamp(0, len(ld) - 2)
    fr = z - lo
    m, s = tab["m_k"].to(dx.device), tab["s_k"].to(dx.device)
    return m[lo] + fr * (m[lo + 1] - m[lo]), s[lo] + fr * (s[lo + 1] - s[lo])
