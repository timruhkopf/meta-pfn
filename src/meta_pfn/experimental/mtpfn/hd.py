"""MTPFN beyond d = 1 and h = id (issue #20, notebook 16): the AIR A→B prior generalised, with an exact oracle.

Prior (one task = a dense source B and a scarce target A), x ∈ [0, 1]^d, d ∈ {1, 2}:
- g: random 2-layer tanh MLP (width 32, random input scale, as experimental/air/prior.py) on [0, 1]^d, mapped to
  [0, 1] by one global ECDF of raw outputs per d (fixed seed, CPU);
- T_θ: box-preserving time-1 flow (RK4), θ̃ ~ N(0, I):
  d = 1: AIR's velocity x(1 − x) Σ_j θ_j cos((j − 1)πx) s_T / j with J coefficients (air.prior.flow, s_t_for);
  d = 2: v_1 = x_1(1 − x_1) s (θ_1 + θ_2 cos(π x_2) / 2), v_2 = x_2(1 − x_2) s (θ_3 + θ_4 cos(π x_1) / 2); 4
  coefficients, the second of each couples the coordinates; s calibrated so that E|T(x) − x| ≈ 0.13 per coordinate;
  with `window` (default) T(x) = c + w φ_θ(x) per coordinate, w ~ U[0.5, 1], c ~ U[0, 1 − w] (as Gaussian latents
  through Φ): A's box maps into a random sub-box of B's. Without it φ_θ fixes the faces (the corners at d = 2, both
  ends at d = 1), so with h = id B gives A's values there exactly (the boundary leak; notebooks 15-16);
- h latent: identity with probability pi_id, otherwise affine (H1 of the AIR spec) a + b v, b ~ U[0.3, 1],
  a ~ U[0, 1 − b];
- y_b = g(x_b) + ε_b, y_a = h(g(T(x_a))) + ε_a, σ_A, σ_B ~ U[0.005, 0.03]; always related (Z0); designs uniform.

Oracle: the exact posterior predictive given the true g, σ_A and the prior (g known exactly: the dense-B limit, a
bound for models that see 128 noisy B points). h is integrated analytically: for fixed (θ, b) the likelihood is
Gaussian in a, integrated in closed form over a ∈ [0, 1 − b] (normal CDFs); b on a fine midpoint grid; the
id/affine mixture exactly. θ: a 101² grid at d = 1, J = 2 without the window; otherwise adaptive importance
sampling (prior, then defensive KDE stages), with the ESS reported. The predictive is a mixture over M posterior
draws of (θ, h) of N(h(g(T_θ(x*))), σ_A²) integrated over the bar buckets; its MC error is reported
(two independent halves).
"""

import functools
import math
from dataclasses import dataclass, fields, replace

import torch

from ..air import reference as R
from ..air.prior import ecdf_map, s_t_for
from ..air.prior import flow as flow1

WIDTH = 32
B_LO = 0.3
T_MEAN_SHIFT = 0.13
RK_STEPS = 24
W_LO = 0.5  # the window's smallest width per coordinate


@dataclass
class HDCfg:
    d: int = 1
    J: int = 2  # d = 1 only (d = 2 has 4 coefficients)
    pi_id: float = 0.5
    n_b: int = 128
    n_q: int = 32
    sig_lo: float = 0.005
    sig_hi: float = 0.03
    window: bool = True  # False: the box-preserving warp of the first notebook-16 runs (leaks the boundary values)


@dataclass
class Funcs:
    W1: torch.Tensor  # [P, H, d]
    b1: torch.Tensor  # [P, H]
    W2: torch.Tensor  # [P, H]
    ecdf: torch.Tensor  # [S]

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """x [P, M, d] -> [P, M]."""
        return (torch.tanh(torch.einsum("pmd,phd->pmh", x, self.W1) + self.b1[:, None]) @ self.W2[..., None])[..., 0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ecdf_map(self.raw(x), self.ecdf)

    def at(self, idx) -> "Funcs":
        return Funcs(self.W1[idx], self.b1[idx], self.W2[idx], self.ecdf)

    def to(self, device) -> "Funcs":
        return Funcs(self.W1.to(device), self.b1.to(device), self.W2.to(device), self.ecdf.to(device))


def _weights(P: int, d: int, g: torch.Generator, device):
    r = lambda *s: torch.rand(*s, generator=g, device=device)  # noqa: E731
    n = lambda *s: torch.randn(*s, generator=g, device=device)  # noqa: E731
    a = 2 + 10 * r(P, 1)
    W1 = n(P, WIDTH, d) * a[..., None]
    b1 = (2 * r(P, WIDTH) - 1) * a
    W2 = n(P, WIDTH) * (0.5 + r(P, 1))
    return W1, b1, W2


@functools.lru_cache(maxsize=4)
def prior_ecdf(d: int, n_funcs: int = 4000, n_x: int = 256, n_q: int = 1001, seed: int = 20261104) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + d)
    W1, b1, W2 = _weights(n_funcs, d, g, "cpu")
    f = Funcs(W1, b1, W2, torch.zeros(2))
    raw = f.raw(torch.rand(n_funcs, n_x, d, generator=g)).flatten().sort().values
    return raw[torch.linspace(0, raw.numel() - 1, n_q).long()]


def draw_funcs(P: int, d: int, g: torch.Generator, device) -> Funcs:
    return Funcs(*_weights(P, d, g, device), prior_ecdf(d).to(device))


# ---- transform -------------------------------------------------------------------------------------------------------


def _vel2(x: torch.Tensor, th: torch.Tensor, s: float) -> torch.Tensor:
    """x [..., M, 2], th [..., 4] -> v [..., M, 2]."""
    x1, x2 = x[..., 0], x[..., 1]
    t = th[..., None, :]
    v1 = x1 * (1 - x1) * s * (t[..., 0] + 0.5 * t[..., 1] * torch.cos(math.pi * x2))
    v2 = x2 * (1 - x2) * s * (t[..., 2] + 0.5 * t[..., 3] * torch.cos(math.pi * x1))
    return torch.stack([v1, v2], -1)


def flow2(x: torch.Tensor, th: torch.Tensor, s: float, steps: int = RK_STEPS) -> torch.Tensor:
    h = 1.0 / steps
    for _ in range(steps):
        k1 = _vel2(x, th, s)
        k2 = _vel2(x + 0.5 * h * k1, th, s)
        k3 = _vel2(x + 0.5 * h * k2, th, s)
        k4 = _vel2(x + h * k3, th, s)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@functools.lru_cache(maxsize=4)
def s_t2(target: float = T_MEAN_SHIFT, n: int = 4000) -> float:
    g = torch.Generator().manual_seed(2)
    th = torch.randn(n, 4, generator=g, dtype=torch.float64)
    x = torch.rand(n, 32, 2, generator=g, dtype=torch.float64)
    lo, hi = 0.01, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = float((flow2(x, th, mid) - x).abs().mean())
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return 0.5 * (lo + hi)


def n_theta(cfg: HDCfg) -> int:
    return (cfg.J if cfg.d == 1 else 4) + (2 * cfg.d if cfg.window else 0)


def s_t(cfg: HDCfg) -> float:
    return s_t_for(cfg.J) if cfg.d == 1 else s_t2()


def warp(x: torch.Tensor, th: torch.Tensor, d: int, s: float, window: bool = False) -> torch.Tensor:
    """x [P, M, d], th [P, J] -> T_θ(x) [P, M, d]; with window the last 2d entries of θ are (z_w, z_c) ~ N(0, I)."""
    k = th.shape[-1] - (2 * d if window else 0)
    u = flow1(x[..., 0], th[..., :k], s)[..., None] if d == 1 else flow2(x, th[..., :k], s)
    if not window:
        return u
    w = W_LO + (1 - W_LO) * torch.special.ndtr(th[..., k : k + d])
    c = (1 - w) * torch.special.ndtr(th[..., k + d :])
    return c[..., None, :] + w[..., None, :] * u


# ---- tasks -----------------------------------------------------------------------------------------------------------


@dataclass
class HDTasks:
    xb: torch.Tensor  # [P, N_B, d]
    yb: torch.Tensor  # [P, N_B]
    xa: torch.Tensor  # [P, n, d]
    ya: torch.Tensor
    xq: torch.Tensor  # [P, Q, d]
    yq: torch.Tensor
    sig_a: torch.Tensor  # [P]
    theta: torch.Tensor  # [P, J]
    hid: torch.Tensor  # [P] bool: h = id
    hp: torch.Tensor  # [P, 2] (a, b)
    g: Funcs
    d: int
    s: float
    window: bool = False

    def _map(self, f, g_idx=None) -> "HDTasks":
        kw = {k.name: getattr(self, k.name) for k in fields(self)}
        for k in ("xb", "yb", "xa", "ya", "xq", "yq", "sig_a", "theta", "hid", "hp"):
            kw[k] = f(kw[k])
        if g_idx is not None:
            kw["g"] = self.g.at(g_idx)
        return HDTasks(**kw)

    def sub(self, idx) -> "HDTasks":
        return self._map(lambda t: t[idx], idx)

    def head(self, n: int) -> "HDTasks":
        return replace(self, xa=self.xa[:, :n], ya=self.ya[:, :n])

    def to(self, device) -> "HDTasks":
        return replace(self._map(lambda t: t.to(device)), g=self.g.to(device))

    def f_a(self, x: torch.Tensor) -> torch.Tensor:
        """The true noise-free f_A at x [P, M, d]."""
        v = self.g(warp(x, self.theta, self.d, self.s, self.window))
        return self.hp[:, :1] + self.hp[:, 1:] * v


def sample_tasks(cfg: HDCfg, P: int, n_a: int, g: torch.Generator, device=None) -> HDTasks:
    """Drawn on the generator's device, returned on `device` (default: the same)."""
    gd = g.device
    r = lambda *s: torch.rand(*s, generator=g, device=gd)  # noqa: E731
    d, s = cfg.d, s_t(cfg)
    f = draw_funcs(P, d, g, gd)
    theta = torch.randn(P, n_theta(cfg), generator=g, device=gd)
    hid = r(P) < cfg.pi_id
    b = B_LO + (1 - B_LO) * r(P, 1)
    a = (1 - b) * r(P, 1)
    hp = torch.where(hid[:, None], torch.tensor([[0.0, 1.0]], device=gd), torch.cat([a, b], 1))
    sig_a = cfg.sig_lo + (cfg.sig_hi - cfg.sig_lo) * r(P)
    sig_b = cfg.sig_lo + (cfg.sig_hi - cfg.sig_lo) * r(P)
    xb, xa, xq = r(P, cfg.n_b, d), r(P, n_a, d), r(P, cfg.n_q, d)
    yb = f(xb) + sig_b[:, None] * torch.randn(P, cfg.n_b, generator=g, device=gd)
    v = f(warp(torch.cat([xa, xq], 1), theta, d, s, cfg.window))
    y = hp[:, :1] + hp[:, 1:] * v + sig_a[:, None] * torch.randn(P, n_a + cfg.n_q, generator=g, device=gd)
    t = HDTasks(xb, yb, xa, y[:, :n_a], xq, y[:, n_a:], sig_a, theta, hid, hp, f, d, s, cfg.window)
    return t if device is None else t.to(device)


N_A_EVAL = {1: (1, 2, 3, 5, 10), 2: (1, 2, 3, 5, 10, 20)}
N_A_TRAIN = {1: 10, 2: 20}


def eval_tasks(cfg: HDCfg, n_tasks: int, seed: int, device) -> HDTasks:
    """Fixed evaluation tasks, drawn on the CPU (identical on every machine), nested in n_A."""
    g = torch.Generator().manual_seed(seed + 1009 * cfg.d)
    return sample_tasks(cfg, n_tasks, N_A_TRAIN[cfg.d], g, device)


# ---- model batches ---------------------------------------------------------------------------------------------------


def mt_batch(t: HDTasks):
    """MTPFN input: task 0 = A (target), task 1 = B."""
    P, n, n_b, d = t.xa.shape[0], t.xa.shape[1], t.xb.shape[1], t.d
    N = max(n, n_b)
    x = t.xa.new_zeros(P, 2, N, d)
    y = t.xa.new_zeros(P, 2, N)
    mask = torch.zeros(P, 2, N, dtype=torch.bool, device=t.xa.device)
    x[:, 0, :n], y[:, 0, :n], mask[:, 0, :n] = t.xa, t.ya, True
    x[:, 1, :n_b], y[:, 1, :n_b], mask[:, 1, :n_b] = t.xb, t.yb, True
    return x, y, mask, t.xq, t.yq, torch.full((P,), d, device=t.xa.device, dtype=torch.long)


def ao_batch(t: HDTasks):
    """A-only input: the same architecture with A as the only task (B never enters)."""
    P, n, d = t.xa.shape[0], t.xa.shape[1], t.d
    mask = torch.ones(P, 1, n, dtype=torch.bool, device=t.xa.device)
    return t.xa[:, None], t.ya[:, None], mask, t.xq, t.yq, torch.full((P,), d, device=t.xa.device, dtype=torch.long)


def sampler(cfg: HDCfg, aonly: bool):
    """Training draws (on the loop's generator device): n_A ~ U{1..N_A_TRAIN[d]}."""

    def draw(rng, g: torch.Generator, P: int):
        n = int(rng.integers(1, N_A_TRAIN[cfg.d] + 1))
        t = sample_tasks(cfg, P, n, g)
        return ao_batch(t) if aonly else mt_batch(t)

    return draw


# ---- oracle ----------------------------------------------------------------------------------------------------------

LOG2PI = math.log(2 * math.pi)


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 − e^x) for x ≤ 0."""
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def log_phi_diff(u: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """log(Φ(u) − Φ(lo)) for u ≥ lo, stable in both tails."""
    ln = torch.special.log_ndtr
    pos = lo > 0
    a, b = torch.where(pos, -lo, u), torch.where(pos, -u, lo)
    return ln(a) + _log1mexp((ln(b) - ln(a)).clamp(max=0))


def b_grid(nb: int, device) -> torch.Tensor:
    return B_LO + (1 - B_LO) * (torch.arange(nb, device=device, dtype=torch.float32) + 0.5) / nb


def h_terms(ya: torch.Tensor, m: torch.Tensor, sig: torch.Tensor, bg: torch.Tensor):
    """ya [n], m [K, n] (g(T_θ(x_a))), sig scalar, bg [nb] -> (L_id [K], lb [K, nb], rbar [K, nb], s): lb is the log
    likelihood of A for (θ, b) with a integrated over its prior U[0, 1 − b] (closed form); L_aff = logmeanexp_b lb."""
    n = ya.shape[0]
    L_id = (-0.5 * ((ya - m) / sig) ** 2).sum(-1) - n * (torch.log(sig) + 0.5 * LOG2PI)
    yb, mb = ya.mean(), m.mean(-1, keepdim=True)
    yc, mc = ya - yb, m - mb
    syy, sym, smm = (yc**2).sum(), (yc * mc).sum(-1, keepdim=True), (mc**2).sum(-1, keepdim=True)
    ss = (syy - 2 * bg * sym + bg**2 * smm).clamp_min(0)
    rbar = yb - bg * mb
    s = sig / math.sqrt(n)
    la = 0.5 * LOG2PI + torch.log(s) + log_phi_diff((1 - bg - rbar) / s, -rbar / s) - torch.log(1 - bg)
    lb = -ss / (2 * sig**2) - n * (torch.log(sig) + 0.5 * LOG2PI) + la
    return L_id, lb, rbar, s


def log_marg(L_id: torch.Tensor, lb: torch.Tensor, pi_id: float) -> torch.Tensor:
    if pi_id >= 1:
        return L_id
    L_aff = torch.logsumexp(lb, -1) - math.log(lb.shape[-1])
    return torch.logaddexp(math.log(pi_id) + L_id, math.log(1 - pi_id) + L_aff)


def _m_of(t: HDTasks, j: int, th: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """g_j(T_θ(x)) for θ [K, J], x [M, d] -> [K, M]."""
    K = th.shape[0]
    u = warp(x[None].expand(K, -1, -1), th, t.d, t.s, t.window)
    return t.g.at(torch.full((K,), j, device=th.device, dtype=torch.long))(u)


def _lik(t: HDTasks, j: int, th: torch.Tensor, pi_id: float, bg: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    out = []
    for c in range(0, th.shape[0], chunk):
        m = _m_of(t, j, th[c : c + chunk], t.xa[j])
        L_id, lb, _, _ = h_terms(t.ya[j], m, t.sig_a[j], bg)
        out.append(log_marg(L_id, lb, pi_id))
    return torch.cat(out)


def _log_n01(th: torch.Tensor) -> torch.Tensor:
    return -0.5 * (th**2).sum(-1) - 0.5 * th.shape[-1] * LOG2PI


@torch.no_grad()
def smc_theta(t: HDTasks, j: int, pi_id: float, bg: torch.Tensor, K: int, g: torch.Generator, n_mh: int = 5,
              target: float = 0.5, max_steps: int = 200) -> tuple[torch.Tensor, dict]:  # fmt: skip
    """Likelihood-tempered SMC from the N(0, I) prior: adaptive temperatures (each reweight keeps ESS = target · K),
    multinomial resampling, n_mh random-walk MH moves per temperature (proposal from the particles' covariance,
    scale adapted to the acceptance rate), 2 n_mh at β = 1. -> (θ [K, J] equally weighted, diagnostics)."""
    dev, J = t.xa.device, t.theta.shape[1]
    th = torch.randn(K, J, generator=g, device=dev)
    L = torch.nan_to_num(_lik(t, j, th, pi_id, bg), nan=-math.inf)
    beta, sc, steps, ess_min, accs = 0.0, 1.0, 0, float(K), []

    def ess_of(db: float) -> float:
        w = db * L
        return float(torch.exp(-torch.logsumexp(2 * (w - torch.logsumexp(w, 0)), 0)))

    while beta < 1 and steps < max_steps:
        lo, hi = 0.0, 1.0 - beta
        if ess_of(hi) < target * K:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ess_of(mid) >= target * K else (lo, mid)
            hi = max(lo, 1e-6)
        ess_min = min(ess_min, ess_of(hi))
        beta = 1.0 if beta + hi >= 1 - 1e-9 else beta + hi
        w = hi * L
        idx = torch.multinomial((w - torch.logsumexp(w, 0)).exp(), K, replacement=True, generator=g)
        th, L = th[idx], L[idx]
        ch = torch.linalg.cholesky(torch.cov(th.T).reshape(J, J) + 1e-8 * torch.eye(J, device=dev))
        for _ in range(n_mh * (2 if beta >= 1 else 1)):
            prop = th + (2.38 / math.sqrt(J)) * sc * torch.randn(K, J, generator=g, device=dev) @ ch.T
            Lp = torch.nan_to_num(_lik(t, j, prop, pi_id, bg), nan=-math.inf)
            la = _log_n01(prop) + beta * Lp - _log_n01(th) - beta * L
            acc = torch.log(torch.rand(K, generator=g, device=dev)) < la
            th, L = torch.where(acc[:, None], prop, th), torch.where(acc, Lp, L)
            a = float(acc.float().mean())
            accs.append(a)
            sc *= 0.7 if a < 0.15 else (1.3 if a > 0.4 else 1.0)
        steps += 1
    return th, {"ess_min": ess_min, "steps": steps, "acc": sum(accs) / max(len(accs), 1), "beta": beta}


@torch.no_grad()
def posterior_theta(t: HDTasks, j: int, pi_id: float, *, nb: int = 2048, k_is: int = 16384, stages: int = 4,
                    g: torch.Generator | None = None, force_is: bool = False,
                    method: str = "auto") -> tuple[torch.Tensor, torch.Tensor, float]:  # fmt: skip
    """-> (θ particles [K, J], normalized log weights [K], ESS). method: auto = the 101² grid at d = 1, J = 2 without
    the window, else SMC; "is" = adaptive importance sampling (the first runs' d = 2 oracle; fails in 8 dims)."""
    dev = t.xa.device
    bg = b_grid(nb, dev)
    J = t.theta.shape[1]
    if method == "auto" and t.d == 1 and J == 2 and not t.window and not force_is:
        th = R.grid(101, device=str(dev))
        lw = _log_n01(th) + _lik(t, j, th, pi_id, bg)
    elif method == "auto" and not force_is:
        g = g or torch.Generator(device=dev).manual_seed(12345 + j)
        th, dg = smc_theta(t, j, pi_id, bg, k_is // 4, g)
        return th, th.new_full((th.shape[0],), -math.log(th.shape[0])), dg["ess_min"]  # ESS: the smallest reweight's
    else:
        g = g or torch.Generator(device=dev).manual_seed(12345 + j)
        th = torch.randn(k_is, J, generator=g, device=dev)
        lw = _lik(t, j, th, pi_id, bg)  # prior proposal: weight = likelihood
        for _ in range(stages):
            w = (lw - torch.logsumexp(lw, 0)).exp()
            idx = torch.multinomial(w, 2048, replacement=True, generator=g)
            c = th[idx]
            bw = 0.4 * c.std(0).clamp_min(1e-3)
            comp = torch.randint(0, 2048, (k_is,), generator=g, device=dev)
            prior_pick = torch.rand(k_is, generator=g, device=dev) < 0.1
            new = torch.where(prior_pick[:, None], torch.randn(k_is, J, generator=g, device=dev),
                              c[comp] + bw * torch.randn(k_is, J, generator=g, device=dev))  # fmt: skip
            lk = []
            for s in range(0, k_is, 4096):
                z = (new[s : s + 4096, None] - c[None]) / bw
                lk.append(torch.logsumexp(-0.5 * (z**2).sum(-1), 1) - torch.log(bw).sum() - 0.5 * J * LOG2PI
                          - math.log(2048))  # fmt: skip
            lq = torch.logaddexp(math.log(0.1) + _log_n01(new), math.log(0.9) + torch.cat(lk))
            th, lw = new, _log_n01(new) + _lik(t, j, new, pi_id, bg) - lq
            ess = float(torch.exp(-torch.logsumexp(2 * (lw - torch.logsumexp(lw, 0)), 0)))
            if ess > k_is / 4:
                break
    lw = lw - torch.logsumexp(lw, 0)
    return th, lw, float(torch.exp(-torch.logsumexp(2 * lw, 0)))


def _uses_smc(t: HDTasks) -> bool:
    return not (t.d == 1 and t.theta.shape[1] == 2 and not t.window)


@torch.no_grad()
def draw_h(t: HDTasks, j: int, th: torch.Tensor, pi_id: float, g: torch.Generator, nb: int = 2048):
    """Posterior draws of (a, b) given each θ [M, J] -> (a [M], b [M])."""
    bg = b_grid(nb, th.device)
    m = _m_of(t, j, th, t.xa[j])
    L_id, lb, rbar, s = h_terms(t.ya[j], m, t.sig_a[j], bg)
    L_aff = torch.logsumexp(lb, -1) - math.log(nb)
    p_id = torch.ones_like(L_id) if pi_id >= 1 else torch.sigmoid(math.log(pi_id) + L_id - math.log(1 - pi_id) - L_aff)
    is_id = torch.rand(th.shape[0], generator=g, device=th.device) < p_id
    k = torch.multinomial((lb - torch.logsumexp(lb, -1, keepdim=True)).exp(), 1, generator=g)[:, 0]
    b = bg[k]
    rb = rbar.gather(1, k[:, None])[:, 0]
    lo, hi = torch.special.ndtr(-rb / s), torch.special.ndtr((1 - b - rb) / s)
    u = lo + (hi - lo) * torch.rand(th.shape[0], generator=g, device=th.device)
    a = (rb + s * torch.special.ndtri(u.clamp(1e-12, 1 - 1e-12))).clamp(0, 1).minimum(1 - b)
    return torch.where(is_id, torch.zeros_like(a), a), torch.where(is_id, torch.ones_like(b), b)


@torch.no_grad()
def oracle_draws(t: HDTasks, j: int, pi_id: float, M: int, g: torch.Generator, post=None):
    th, lw, ess = post if post is not None else posterior_theta(t, j, pi_id, g=g)
    idx = torch.multinomial(lw.exp(), M, replacement=True, generator=g)
    a, b = draw_h(t, j, th[idx], pi_id, g)
    return th[idx], a, b, ess


@torch.no_grad()
def oracle_lp(t: HDTasks, j: int, pi_id: float, M: int = 16384, seed: int = 0, post=None) -> dict:
    """log predictive density of task j's queries [Q] (bucket scale), ESS, MC error (|half A − half B| mean over Q)."""
    g = torch.Generator(device=t.xa.device).manual_seed(seed + 7919 * j)
    if post is None and _uses_smc(t):  # two independent SMC runs: the halves' difference is an honest MC error
        p1, p2 = posterior_theta(t, j, pi_id, g=g), posterior_theta(t, j, pi_id, g=g)
        d1, d2 = oracle_draws(t, j, pi_id, M // 2, g, p1), oracle_draws(t, j, pi_id, M - M // 2, g, p2)
        th, a, b = (torch.cat([x, y]) for x, y in zip(d1[:3], d2[:3], strict=True))
        ess = min(d1[3], d2[3])
    else:
        th, a, b, ess = oracle_draws(t, j, pi_id, M, g, post)
    mq = _m_of(t, j, th, t.xq[j])
    lp = R.bucket_lp(a[:, None] + b[:, None] * mq, t.sig_a[j], t.yq[j][None])  # [M, Q]
    half = M // 2
    l1 = torch.logsumexp(lp[:half], 0) - math.log(half)
    l2 = torch.logsumexp(lp[half:], 0) - math.log(M - half)
    return {"lp": torch.logsumexp(lp, 0) - math.log(M), "ess": ess, "mc": float((l1 - l2).mean().abs())}


@torch.no_grad()
def oracle_ppd(t: HDTasks, j: int, pi_id: float, xs: torch.Tensor, borders: torch.Tensor, M: int = 4096,
               seed: int = 0) -> torch.Tensor:  # fmt: skip
    """Predictive bucket probabilities on points xs [G, d] -> [G, bins]."""
    g = torch.Generator(device=t.xa.device).manual_seed(seed + 7919 * j)
    th, a, b, _ = oracle_draws(t, j, pi_id, M, g)
    s = t.sig_a[j]
    p = torch.zeros(xs.shape[0], borders.shape[0] - 1, device=xs.device)
    for c in range(0, M, 256):
        mu = a[c : c + 256, None] + b[c : c + 256, None] * _m_of(t, j, th[c : c + 256], xs)  # [m, G]
        cdf = torch.special.ndtr((borders - mu[..., None]) / s)  # [m, G, bins + 1]
        p += (cdf[..., 1:] - cdf[..., :-1]).sum(0)
    return p / M


@torch.no_grad()
def evaluate(models: dict, cfg: HDCfg, n_tasks: int, seed: int, device, log=print):
    """models: {"mtpfn": MTPFN (two tasks), "aonly": MTPFN (A only)}; -> (nll rows, task rows): per (n_A, task) the NLL
    per query of each model and of the oracle, with the oracle's ESS and MC error."""
    from .train import nll_per_task

    base = eval_tasks(cfg, n_tasks, seed, device)
    shift = (warp(base.xq, base.theta, cfg.d, base.s, cfg.window) - base.xq).abs().mean((1, 2))
    task_rows = [{"d": cfg.d, "task": i, "h_id": bool(base.hid[i]), "a": float(base.hp[i, 0]),
                  "b": float(base.hp[i, 1]), "sig_a": float(base.sig_a[i]), "shift": float(shift[i])}
                 for i in range(n_tasks)]  # fmt: skip
    rows = []
    for n in N_A_EVAL[cfg.d]:
        t = base.head(n)
        for name, m in models.items():
            m.eval()
            v = nll_per_task(m, ao_batch(t) if name == "aonly" else mt_batch(t))
            rows += [{"d": cfg.d, "n_a": n, "task": i, "method": name, "nll": float(x)} for i, x in enumerate(v)]
        for i in range(n_tasks):
            o = oracle_lp(t, i, cfg.pi_id)
            rows.append({"d": cfg.d, "n_a": n, "task": i, "method": "oracle", "nll": float(-o["lp"].mean()),
                         "ess": o["ess"], "mc": o["mc"]})  # fmt: skip
        log(f"d={cfg.d} n_a={n} done")
    import pandas as pd

    return pd.DataFrame(rows), pd.DataFrame(task_rows)
