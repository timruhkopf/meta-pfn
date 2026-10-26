"""Task pairs: a scarce target task A and a dense source task B whose inputs are related by a warp.

One latent function f ~ BNN prior (pfn-repro, the frozen PFNs' training prior and ECDF scale) lives on
[0, 1]^d. B observes it directly, y_B = f(x) + noise. A observes it through T: y_A = f(T(x)) + noise, with
T: [0, 1]^d -> [0, 1]^d per axis (so A covers a sub-window of B, possibly non-uniformly stretched).
Strata (`kind`):
- `identity`:  T = id (B is A's own task);
- `window`:    T_i(x) = a_i + b_i x, b_i ~ U(b_min, 1), a_i ~ U(0, 1 - b_i);
- `warp`:      T_i(x) = a_i + b_i K(x; α_i, β_i), Kumaraswamy CDF with log α, log β ~ N(0, s²);
- `unrelated`: B observes an independent function; T is meaningless.
Every T is a forward map from A's coordinates into B's: nothing is ever inverted.
A pair is identified by (split, d, kind, i) alone (`pair_seed`), drawn on the CPU.
"""

from dataclasses import dataclass

import torch
from pfn_repro.checkpoints import make_prior

KINDS = ("identity", "window", "warp", "unrelated")
_EPS = 1e-6


def pair_seed(split: int, d: int, kind: str, i: int) -> int:
    """Disjoint seed ranges per split (0 pilot, 1 confirm, ...), d and kind."""
    return split * 100_000_007 + d * 1_000_003 + KINDS.index(kind) * 100_019 + i


@dataclass
class WarpFamily:
    b_min: float = 0.4  # smallest window width: A sees at least 40% of each axis of B
    kum_log_sd: float = 0.5

    def sample(self, kind: str, n: int, d: int, g: torch.Generator) -> torch.Tensor:
        """θ [n, d, 4] = (a, b, α, β) per axis."""
        if kind == "identity" or kind == "unrelated":
            th = torch.zeros(n, d, 4)
            th[..., 1:] = 1.0
            return th
        b = self.b_min + (1 - self.b_min) * torch.rand(n, d, generator=g)
        a = (1 - b) * torch.rand(n, d, generator=g)
        if kind == "window":
            al = be = torch.ones(n, d)
        elif kind == "warp":
            al = torch.exp(self.kum_log_sd * torch.randn(n, d, generator=g))
            be = torch.exp(self.kum_log_sd * torch.randn(n, d, generator=g))
        else:
            raise ValueError(kind)
        return torch.stack([a, b, al, be], -1)


def apply_warp(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """theta [L, d, 4], x [L, N, d] in [0, 1] -> T(x) [L, N, d] in [0, 1]."""
    a, b, al, be = (theta[:, None, :, k] for k in range(4))
    u = x.clamp(_EPS, 1 - _EPS)
    k = 1 - (1 - u**al) ** be
    return a + b * k


@dataclass
class Pairs:
    d: int
    kind: str
    seeds: list[int]
    rho: float
    theta: torch.Tensor  # [P, d, 4] the true T (identity for `identity` / `unrelated`)
    XB: torch.Tensor  # [P, n_B, d]
    YB: torch.Tensor  # [P, n_B]
    XA: torch.Tensor  # [P, n_A_max, d]: nested designs, prefixes give smaller n_A
    YA: torch.Tensor
    XQ: torch.Tensor  # [P, n_Q, d]: held-out A queries
    YQ: torch.Tensor

    @property
    def P(self) -> int:
        return len(self.seeds)


WEIGHTS = ("W_in", "b_in", "W_h", "b_h", "W_out", "b_out")


def function(f_seed: int, d: int, rho: float = 0.0, pert_seed: int = 0):
    """The BNN function `f_seed`; with rho > 0 a relative of it: every nonzero weight becomes
    sqrt(1 - ρ²) w + ρ rms(w) ε. This is variance-preserving with the same architecture and masks, so it is
    still a prior draw; ρ = 1 gives independent weights."""
    prior = make_prior(x_dim=d, batch_size=1, seed=f_seed)
    if rho > 0:
        g = torch.Generator().manual_seed(pert_seed)
        for name in WEIGHTS:
            w = getattr(prior, name)
            eps = torch.randn(w.shape, generator=g)
            nz = w[w != 0]
            scale = nz.pow(2).mean().sqrt() if len(nz) else w.new_zeros(())  # rms over the live weights
            setattr(prior, name, (1 - rho**2) ** 0.5 * w + rho * scale * eps * (w != 0))
    return prior


def observe(prior, noise_seed: int, x: torch.Tensor) -> torch.Tensor:
    """Noisy observations (the PFN's training noise) of a one-function prior at x [N, d]."""
    torch.manual_seed(noise_seed)  # BNNPrior draws its observation noise from the global RNG
    with torch.no_grad():
        return prior.evaluate(x[None], noise=True)[0]


def draw_pairs(
    d: int,
    kind: str,
    split: int,
    n_pairs: int,
    n_b: int,
    n_a_max: int,
    n_q: int,
    family: WarpFamily,
    rho: float = 0.0,
) -> Pairs:
    """rho: A's function is a weight-space relative of B's (`function`); 0 = the same function."""
    seeds = [pair_seed(split, d, kind, i) for i in range(n_pairs)]
    cols: dict[str, list[torch.Tensor]] = {k: [] for k in ("th", "XB", "YB", "XA", "YA", "XQ", "YQ")}
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        theta = family.sample(kind, 1, d, g)
        xb = torch.rand(n_b, d, generator=g)
        xa = torch.rand(n_a_max, d, generator=g)
        xq = torch.rand(n_q, d, generator=g)
        f_a = function(s, d, rho, pert_seed=s + 3)
        f_b = function(s + 7_777_777, d) if kind == "unrelated" else function(s, d)
        y_aq = observe(f_a, s + 1, apply_warp(theta, torch.cat([xa, xq])[None])[0])
        y_b = observe(f_b, s + 2, xb)
        for k, v in zip(cols, (theta[0], xb, y_b, xa, y_aq[:n_a_max], xq, y_aq[n_a_max:]), strict=True):
            cols[k].append(v)
    st = {k: torch.stack(v) for k, v in cols.items()}
    return Pairs(d, kind, seeds, rho, st["th"], st["XB"], st["YB"], st["XA"], st["YA"], st["XQ"], st["YQ"])
