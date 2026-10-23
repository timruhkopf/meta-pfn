"""E0b pairs and predictors: A differs from B only by a monotone output map g of [0, 1] onto itself; x is shared.

One latent function f ~ BNN prior (pfn-repro; y on the prior's ECDF scale [0, 1]). B observes it directly,
y_B = f(x) + noise. A observes it through g: y_A = g(f(x) + noise), g a Kumaraswamy CDF
g(u) = 1 - (1 - u^α)^β with log α, log β ~ N(0, s²). The prior adds its noise before the ECDF, so
g⁻¹(y_A) is exactly a prior observation of f, and the frozen PFN gives every predictor below exactly,
by change of variables: log p(y_A) = log PFN(g⁻¹(y_A)) + log |(g⁻¹)'(y_A)|. g has a closed-form inverse.
Strata (`kind`):
- `identity`:  g = id (B is A's own task);
- `outmap`:    g ~ the family;
- `unrelated`: B observes an independent function; g = id with probability 1 - p_map, else ~ the family.

Predictors of A's held-out queries, all with one frozen PFN:
- `lower`:       PFN(A) on A's raw y                     A alone, as a plain prior draw (g = id assumed)
- `lower_bayes`: A alone, g inferred (mixture over g)    A alone under the pair prior: what B must beat
- `naive`:       PFN(B ∪ A), g = id assumed              pooling without the output map
- `oracle_g`:    PFN(B ∪ g⁻¹(A)), the true g             relation known (upper bound)
- `bayes`:       mixture over {B used, B ignored} × {g = id, g_1..K}, each weighted by p(h) times the
                 PFN chain-rule likelihood of A's points (with the Jacobian). g_1..K is a randomized
                 quantile grid of the family prior (equal prior mass per point).
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..oracle import logp
from ..pairs import function, observe

KINDS = ("identity", "outmap", "unrelated")
PREDICTORS = ("lower", "lower_bayes", "naive", "oracle_g", "bayes")
_EPS = 1e-6


def pair_seed(split: int, d: int, kind: str, i: int) -> int:
    """Disjoint from E0's seeds (pairs.pair_seed uses kind indices 0-3; these use 4-6)."""
    return split * 100_000_007 + d * 1_000_003 + (4 + KINDS.index(kind)) * 100_019 + i


@dataclass
class OutputFamily:
    log_sd: float = 0.5

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        """ab [n, 2] = (α, β)."""
        return torch.exp(self.log_sd * torch.randn(n, 2, generator=g))

    def grid(self, k: int, offset: float) -> torch.Tensor:
        """A k x k quantile grid of the prior, ab [k², 2]: every point carries prior mass 1/k²."""
        z = torch.special.ndtri((torch.arange(k, dtype=torch.float64) + offset) / k).float() * self.log_sd
        za, zb = torch.meshgrid(z, z, indexing="ij")
        return torch.stack([za.reshape(-1), zb.reshape(-1)], -1).exp()


def _clip(y: torch.Tensor) -> torch.Tensor:
    return y.clamp(_EPS, 1 - _EPS)


def kum(u: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """g(u) for u [N] and one ab [2] -> [N]; or for ab [H, 2] -> [H, N]."""
    a, b = (ab[..., k, None] for k in range(2))
    return 1 - (1 - _clip(u) ** a) ** b


def kum_inv(y: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """g⁻¹(y) = (1 - (1 - y)^{1/β})^{1/α}; y [N], ab [H, 2] -> [H, N]."""
    a, b = (ab[..., k, None] for k in range(2))
    w = -torch.expm1(torch.log1p(-_clip(y)) / b)
    return w.clamp_min(1e-30) ** (1 / a)


def log_jac_inv(y: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """log |(g⁻¹)'(y)|; y [N], ab [H, 2] -> [H, N]."""
    a, b = (ab[..., k, None] for k in range(2))
    lv = torch.log1p(-_clip(y))
    lw = torch.log((-torch.expm1(lv / b)).clamp_min(1e-30))
    return -torch.log(a) + (1 / a - 1) * lw - torch.log(b) + (1 / b - 1) * lv


@dataclass
class OutPairs:
    d: int
    kind: str
    seeds: list[int]
    ab: torch.Tensor  # [P, 2] the true g ((1, 1) = identity)
    XB: torch.Tensor  # [P, n_B, d]
    YB: torch.Tensor  # [P, n_B]
    XA: torch.Tensor  # [P, n_A_max, d]: nested designs, prefixes give smaller n_A
    YA: torch.Tensor
    XQ: torch.Tensor  # [P, n_Q, d]: held-out A queries
    YQ: torch.Tensor

    @property
    def P(self) -> int:
        return len(self.seeds)


def draw_pairs(
    d: int, kind: str, split: int, n_pairs: int, n_b: int, n_a_max: int, n_q: int, family: OutputFamily, p_map: float
) -> OutPairs:
    seeds = [pair_seed(split, d, kind, i) for i in range(n_pairs)]
    cols: dict[str, list[torch.Tensor]] = {k: [] for k in ("ab", "XB", "YB", "XA", "YA", "XQ", "YQ")}
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        ab = family.sample(1, g)[0]
        if kind == "identity" or (kind == "unrelated" and float(torch.rand(1, generator=g)) >= p_map):
            ab = torch.ones(2)
        xb = torch.rand(n_b, d, generator=g)
        xa = torch.rand(n_a_max, d, generator=g)
        xq = torch.rand(n_q, d, generator=g)
        f_a = function(s, d)
        f_b = function(s + 7_777_777, d) if kind == "unrelated" else f_a
        y_aq = kum(observe(f_a, s + 1, torch.cat([xa, xq])), ab)
        y_b = observe(f_b, s + 2, xb)
        for k, v in zip(cols, (ab, xb, y_b, xa, y_aq[:n_a_max], xq, y_aq[n_a_max:]), strict=True):
            cols[k].append(v)
    st = {k: torch.stack(v) for k, v in cols.items()}
    return OutPairs(d, kind, seeds, st["ab"], st["XB"], st["YB"], st["XA"], st["YA"], st["XQ"], st["YQ"])


def chain(pfn, xb, yb, xa, ya, xq, yq, ab, use_b: bool, n_grid: list[int], chunk: int):
    """For hypotheses ab [H, 2] that all use B or all ignore it: the chain-rule log likelihood of A's points
    ll [H, n_max], ll[h, i] = log p(y_i | ctx_h(A_<i)), and the query log densities lq [len(n_grid), H, Q]."""
    H, n_max = len(ab), max(n_grid)
    ua, ja = kum_inv(ya, ab), log_jac_inv(ya, ab)  # [H, n_max]
    uq, jq = kum_inv(yq, ab), log_jac_inv(yq, ab)  # [H, Q]

    def ctx(n: int):
        x = torch.cat([xb, xa[:n]]) if use_b else xa[:n]
        y = torch.cat([yb[None].expand(H, -1), ua[:, :n]], 1) if use_b else ua[:, :n]
        return x[None].expand(H, -1, -1), y, torch.ones(H, len(x), dtype=torch.bool)

    ll = torch.zeros(H, n_max)
    for i in range(n_max):
        X, Y, M = ctx(i)
        ll[:, i] = logp(pfn, X, Y, M, xa[None, i : i + 1].expand(H, -1, -1), ua[:, i : i + 1], chunk)[:, 0] + ja[:, i]
    lq = []
    for n in n_grid:
        X, Y, M = ctx(n)
        lq.append(logp(pfn, X, Y, M, xq[None].expand(H, -1, -1), uq, chunk) + jq)
    return ll, torch.stack(lq)


@torch.no_grad()
def evaluate_pair(
    pfn, pairs: OutPairs, j: int, grid_ab: torch.Tensor, mass: dict[str, float], p_map: float, n_grid, chunk=512
) -> dict[str, np.ndarray]:
    """-> {"nll": [len(n_grid), 5] mean query NLL of PREDICTORS;
    "post": [len(n_grid), 3] posterior mass of (identity, outmap, unrelated);
    "ess": [len(n_grid)] ESS within the outmap grid (B used); "g_err": [len(n_grid)] posterior RMS of
    g(u) - g_true(u) over u ∈ [0, 1] within the outmap grid}."""
    xb, yb, xa, ya, xq, yq = (t[j] for t in (pairs.XB, pairs.YB, pairs.XA, pairs.YA, pairs.XQ, pairs.YQ))
    K, one = len(grid_ab), torch.ones(1, 2)
    m = {k: v / sum(mass.values()) for k, v in mass.items()}
    # rows with B: [identity, truth, grid...]; rows without B: [identity, grid...]
    ab_b = torch.cat([one, pairs.ab[j][None], grid_ab])
    ab_0 = torch.cat([one, grid_ab])
    ll_b, lq_b = chain(pfn, xb, yb, xa, ya, xq, yq, ab_b, True, n_grid, chunk)
    ll_0, lq_0 = chain(pfn, xb, yb, xa, ya, xq, yq, ab_0, False, n_grid, chunk)
    lp_b = torch.tensor([np.log(m["identity"])] + [np.log(m["outmap"]) - np.log(K)] * K)
    lp_0 = torch.tensor([np.log(m["unrelated"] * (1 - p_map))] + [np.log(m["unrelated"] * p_map) - np.log(K)] * K)
    keep_b = torch.ones(len(ab_b), dtype=torch.bool)
    keep_b[1] = False  # the truth is the oracle, not a hypothesis
    u = torch.linspace(0, 1, 101)
    g_grid, g_true = kum(u, grid_ab), kum(u, pairs.ab[j])
    nll, post, ess, gerr = [], [], [], []
    for a, n in enumerate(n_grid):
        lw_b = lp_b + ll_b[keep_b, :n].sum(1)
        lw_0 = lp_0 + ll_0[:, :n].sum(1)
        lw = torch.cat([lw_b, lw_0])
        w = lw - torch.logsumexp(lw, 0)
        lq = torch.cat([lq_b[a][keep_b], lq_0[a]])
        bayes = torch.logsumexp(w[:, None] + lq, 0)
        w0 = lw_0 - torch.logsumexp(lw_0, 0)
        lower_bayes = torch.logsumexp(w0[:, None] + lq_0[a], 0)
        nll.append(
            [
                -float(lq_0[a, 0].mean()),
                -float(lower_bayes.mean()),
                -float(lq_b[a, 0].mean()),
                -float(lq_b[a, 1].mean()),
                -float(bayes.mean()),
            ]
        )
        pw = w.exp()
        post.append([float(pw[0]), float(pw[1 : K + 1].sum()), float(pw[K + 1 :].sum())])
        wk = torch.softmax(lw_b[1:], 0)
        ess.append(float(1 / (wk**2).sum()))
        gerr.append(float((((g_grid - g_true) ** 2).mean(1) * wk).sum().sqrt()))
    return {"nll": np.asarray(nll), "post": np.asarray(post), "ess": np.asarray(ess), "g_err": np.asarray(gerr)}
