"""Sanity ladder S0/S1 of the A→B query architecture (issue #16, notebook 08,
docs/design_2026-10-28_A_to_B_architecture.md §8). Frozen pfn_large only, no training.

Pairs: f ~ BNN prior on [0, 1]^d (pfn-repro's training prior and ECDF scale). B observes f at a uniform design;
A observes f(T(x)) with T(x) = x + s, s ~ U(-s_max, s_max)^d, h = id. A's inputs (train and queries) are drawn
so that T(x) stays in [0, 1]^d. Strata (`kind`): `shift`; `identity` (s = 0); `unrelated` (B observes an
independent function, s is meaningless).

Q_B is the frozen PFN's test-token path over B's cached per-layer K/V: it only ever sees B, and A's labels are
compared with its output in the same y-scale (h = id).

S0 predictors (per query x*, A's first n points):
- `a_only`:    PFN(A_<n) at x*
- `oracle_q`:  PFN(B) at T(x*), the oracle particle (true s; A unused)
- `teacher_a`: PFN(A_<n ∪ B_inA) at x*, B_inA = {(x_b - s, y_b)} restricted to [0, 1]^d
- `teacher_b`: PFN(B ∪ A_inB,<n) at T(x*), A_inB = {(x_a + s, y_a)} (the B-space teacher, §7.2)
- `naive`:     PFN(B ∪ A_<n) at x* (B pooled without registration)

S1 (grid over s, uniform prior on the grid; x + s is clipped to [0, 1]^d):
- `pseudo`: log L(s) = Σ_{a<n} log Q_B(y_a | x_a + s), the architecture's likelihood (A's points independent given B);
  predictive Σ_s w_s Q_B(y | x* + s) (`grid_pseudo`) and the MAP plug-in (`map_pseudo`, one particle).
- `exact`: the chain rule log L(s) = Σ_a log PFN(y_a | B ∪ A_inB,<a(s), x_a + s), exact for a Bayes-optimal PFN;
  predictive Σ_s w_s PFN(y | B ∪ A_inB,<n(s), x* + s) (`grid_exact`), and its mixture with A-only at
  prior 1/2 (`bayes_null`, the Bayes learner that also allows "unrelated").
- gradient descent (Adam) on s from 0 through Q_B on the pseudo log-likelihood.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.ndimage
import torch

from ..oracle import logp
from ..pairs import function, observe

KINDS = ("shift", "identity", "unrelated")
PREDICTORS = ("a_only", "oracle_q", "teacher_a", "teacher_b", "naive", "grid_pseudo", "map_pseudo", "grid_exact",
              "bayes_null")  # fmt: skip
MODE_FRAC = 0.05  # a mode: a connected grid region where the posterior is >= 5% of its peak
NEAR = 0.05  # posterior mass within NEAR (L∞) of the true s


def ladder_seed(split: int, d: int, kind: str, i: int) -> int:
    """Disjoint from `pairs.pair_seed` (offset 9e8) and across split / d / kind."""
    return 900_000_000 + split * 100_000_007 + d * 1_000_003 + KINDS.index(kind) * 100_019 + i


@dataclass
class ShiftPair:
    seed: int
    s: torch.Tensor  # [d] the true shift
    xb: torch.Tensor  # [N_B, d]
    yb: torch.Tensor
    xa: torch.Tensor  # [n_max, d] nested design: prefixes give smaller n_A
    ya: torch.Tensor
    xq: torch.Tensor  # [Q, d] held-out A queries
    yq: torch.Tensor


def draw_pair(d: int, kind: str, split: int, i: int, n_b: int, n_a_max: int, n_q: int, s_max: float) -> ShiftPair:
    seed = ladder_seed(split, d, kind, i)
    g = torch.Generator().manual_seed(seed)
    s = (2 * torch.rand(d, generator=g) - 1) * s_max
    if kind == "identity":
        s = torch.zeros(d)
    lo, hi = (-s).clamp(min=0), (1 - s).clamp(max=1)  # A's box: T(x) = x + s stays in [0, 1]^d
    xb = torch.rand(n_b, d, generator=g)
    xa = lo + (hi - lo) * torch.rand(n_a_max, d, generator=g)
    xq = lo + (hi - lo) * torch.rand(n_q, d, generator=g)
    f = function(seed, d)
    f_b = function(seed + 7_777_777, d) if kind == "unrelated" else f
    y_aq = observe(f, seed + 1, torch.cat([xa, xq]) + s)
    yb = observe(f_b, seed + 2, xb)
    return ShiftPair(seed, s, xb, yb, xa, y_aq[:n_a_max], xq, y_aq[n_a_max:])


def shift_grid(d: int, s_max: float, n1: int, n2: int) -> torch.Tensor:
    """[G, d]: n1 points for d = 1, n2 per axis for d = 2."""
    ax = torch.linspace(-s_max, s_max, n1 if d == 1 else n2)
    return torch.cartesian_prod(*[ax] * d).reshape(-1, d)


def _ragged(pfn, ctxs, qs, chunk: int) -> torch.Tensor:
    """Contexts of different sizes, padded with masked tokens (an all-masked row = no data). -> [L, Q]."""
    d = qs[0][0].shape[-1]
    n = max(len(x) for x, _ in ctxs)
    L = len(ctxs)
    X, Y, M = torch.zeros(L, n, d), torch.zeros(L, n), torch.zeros(L, n, dtype=torch.bool)
    dev = qs[0][0].device
    X, Y, M = X.to(dev), Y.to(dev), M.to(dev)
    for i, (x, y) in enumerate(ctxs):
        X[i, : len(x)], Y[i, : len(x)], M[i, : len(x)] = x, y, True
    return logp(pfn, X, Y, M, torch.stack([q for q, _ in qs]), torch.stack([y for _, y in qs]), chunk)


def _kv(pfn, x, y):
    nf = torch.full((1,), x.shape[-1], dtype=torch.long, device=x.device)
    return pfn.context_kv(x[None], y[None], n_features=nf), nf


def _q_b(pfn, kv, nf, xq: torch.Tensor, yq: torch.Tensor, qchunk: int = 8192) -> torch.Tensor:
    """log Q_B(yq | xq) for a flat batch of queries [M, d] on one cached context. -> [M]."""
    out = []
    for s in range(0, len(xq), qchunk):
        lg = pfn.logits_from_context_kv(kv, xq[None, s : s + qchunk], n_features=nf).float()
        out.append(-pfn.bar_dist(lg, yq[None, s : s + qchunk])[0])
    return torch.cat(out)


def _shifted(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """x [N, d], grid [G, d] -> clip(x + s) [G, N, d]."""
    return (x[None] + grid[:, None]).clamp(0, 1)


def n_modes(w: np.ndarray, d: int) -> int:
    shape = (len(w),) if d == 1 else (round(len(w) ** 0.5),) * 2
    return int(scipy.ndimage.label((w >= MODE_FRAC * w.max()).reshape(shape), structure=np.ones((3,) * d))[1])  # type: ignore[index]


def post_stats(lw: torch.Tensor, grid: torch.Tensor, s: torch.Tensor) -> dict:
    """Posterior summaries from unnormalized log weights lw [G]."""
    w = (lw - torch.logsumexp(lw, 0)).exp()
    G = len(w)
    mean = (w[:, None] * grid).sum(0)
    m = int(w.argmax())
    wn = w.cpu().numpy()
    return {
        "s_map": grid[m].tolist(),
        "err_map": float((grid[m] - s).norm()),
        "err_mean": float((mean - s).norm()),
        "post_sd": float((w * ((grid - mean) ** 2).sum(1)).sum().sqrt()),
        "entropy": float(-(w * w.clamp_min(1e-300).log()).sum() / np.log(G)),
        "n_modes": n_modes(wn, grid.shape[1]),
        "mass_true": float(w[((grid - s).abs().max(1).values <= NEAR)].sum()),
        "w_map": float(w[m]),
    }


@torch.no_grad()
def evaluate_pair(pfn, p: ShiftPair, grid: torch.Tensor, n_grid: list[int], chunk: int) -> dict:
    """-> {"nll": rows (n_a, predictor, nll), "post": rows (n_a, method, stats), "lw": {(method, n): [G]},
    "ll_pseudo": [G, n_max] per-point pseudo log-likelihoods}."""
    dev = grid.device
    xb, yb, xa, ya, xq, yq, s = (t.to(dev) for t in (p.xb, p.yb, p.xa, p.ya, p.xq, p.yq, p.s))
    G, Q, n_max = len(grid), len(xq), max(n_grid)
    kv, nf = _kv(pfn, xb, yb)
    xq_t = (xq + s).clamp(0, 1)
    lq = {"oracle_q": _q_b(pfn, kv, nf, xq_t, yq)}
    keep = ((xb - s >= 0) & (xb - s <= 1)).all(1)
    b_in_a = (xb[keep] - s, yb[keep])
    ctxs = {
        "a_only": lambda n: ((xa[:n], ya[:n]), (xq, yq)),
        "teacher_a": lambda n: ((torch.cat([xa[:n], b_in_a[0]]), torch.cat([ya[:n], b_in_a[1]])), (xq, yq)),
        "teacher_b": lambda n: ((torch.cat([xb, (xa[:n] + s).clamp(0, 1)]), torch.cat([yb, ya[:n]])), (xq_t, yq)),
        "naive": lambda n: ((torch.cat([xb, xa[:n]]), torch.cat([yb, ya[:n]])), (xq, yq)),
    }
    names = list(ctxs)
    batch = [ctxs[k](n) for n in n_grid for k in names]
    out = _ragged(pfn, [c for c, _ in batch], [q for _, q in batch], chunk).reshape(len(n_grid), len(names), Q)

    # S1, pseudo: one cached B, all grid shifts as queries
    ll_p = _q_b(pfn, kv, nf, _shifted(xa, grid).reshape(-1, xa.shape[1]), ya.repeat(G)).reshape(G, n_max)
    lq_p = _q_b(pfn, kv, nf, _shifted(xq, grid).reshape(-1, xq.shape[1]), yq.repeat(G)).reshape(G, Q)
    # S1, exact chain rule: context B ∪ A_inB,<i(s) per grid point
    ta, tq = _shifted(xa, grid), _shifted(xq, grid)

    def ctx(n):
        X = torch.cat([xb[None].expand(G, -1, -1), ta[:, :n]], 1)
        Y = torch.cat([yb[None].expand(G, -1), ya[None, :n].expand(G, -1)], 1)
        return X, Y, torch.ones(G, X.shape[1], dtype=torch.bool, device=dev)

    ll_e = torch.stack([logp(pfn, *ctx(i), ta[:, i : i + 1], ya[i : i + 1][None].expand(G, -1), chunk)[:, 0]
                        for i in range(n_max)], 1)  # fmt: skip
    # null: A-only chain rule
    prev = [(xa[:i], ya[:i]) for i in range(n_max)]
    ll_0 = _ragged(pfn, prev, [(xa[i : i + 1], ya[i : i + 1]) for i in range(n_max)], chunk)[:, 0]

    rows, prow, lws = [], [], {}
    for j, n in enumerate(n_grid):
        q = {k: out[j, i] for i, k in enumerate(names)} | lq
        lw_p, lw_e = ll_p[:, :n].sum(1), ll_e[:, :n].sum(1)
        wp, we = lw_p - torch.logsumexp(lw_p, 0), lw_e - torch.logsumexp(lw_e, 0)
        q["grid_pseudo"] = torch.logsumexp(wp[:, None] + lq_p, 0)
        q["map_pseudo"] = lq_p[int(lw_p.argmax())]
        lq_e = logp(pfn, *ctx(n), tq, yq[None].expand(G, -1), chunk)
        q["grid_exact"] = torch.logsumexp(we[:, None] + lq_e, 0)
        log_zs, log_z0 = torch.logsumexp(lw_e, 0) - np.log(G), ll_0[:n].sum()
        ps = torch.sigmoid(log_zs - log_z0)
        q["bayes_null"] = torch.logaddexp(ps.log() + q["grid_exact"], (1 - ps).clamp_min(1e-30).log() + q["a_only"])
        for k in PREDICTORS:
            rows.append({"n_a": n, "predictor": k, "nll": -float(q[k].mean())})
        for meth, lw in (("pseudo", lw_p), ("exact", lw_e)):
            prow.append({"n_a": n, "method": meth, **post_stats(lw, grid, s)})
            lws[(meth, n)] = lw.cpu()
        prow[-1]["p_related"] = float(ps)
    return {"nll": rows, "post": prow, "lw": lws, "ll_pseudo": ll_p, "n_b_in_a": int(keep.sum())}


def gd_shift(pfn, p: ShiftPair, n_list: list[int], s_max: float, steps: int, lr: float, dev, every: int = 0):
    """Adam on s from 0 through Q_B, maximizing the pseudo log-likelihood of A's first n points (one run per n,
    batched). -> (s_gd [len(n_list), d], ll_gd [len(n_list)], trajectory [steps // every, len(n_list), d] or None)."""
    xb, yb, xa, ya = (t.to(dev) for t in (p.xb, p.yb, p.xa, p.ya))
    kv, nf = _kv(pfn, xb, yb)
    L, n_max, d = len(n_list), len(xa), xa.shape[1]
    mask = (torch.arange(n_max, device=dev)[None] < torch.tensor(n_list, device=dev)[:, None]).float()
    s = torch.zeros(L, d, device=dev, requires_grad=True)
    opt = torch.optim.Adam([s], lr=lr)
    traj = []

    def ll(sv):
        x = (xa[None] + sv[:, None]).clamp(0, 1).reshape(1, -1, d)
        lg = pfn.logits_from_context_kv(kv, x, n_features=nf).float()
        return -(pfn.bar_dist(lg, ya.repeat(L)[None])[0].reshape(L, n_max) * mask).sum(1)

    for t in range(steps):
        opt.zero_grad()
        (-ll(s).sum()).backward()
        opt.step()
        with torch.no_grad():
            s.clamp_(-s_max, s_max)
        if every and t % every == 0:
            traj.append(s.detach().cpu().clone())
    with torch.no_grad():
        return s.detach().cpu(), ll(s).cpu(), (torch.stack(traj) if traj else None)


@torch.no_grad()
def curves(pfn, p: ShiftPair, n_show: int, n_x: int = 200) -> pd.DataFrame:
    """1d example: true f_A (noiseless), the oracle particle Q_B(T(x)) and the A-only PFN at n_show, on a grid."""
    dev = next(pfn.parameters()).device
    x = torch.linspace(0, 1, n_x)[:, None]
    f = function(p.seed, 1)
    f_a = f.evaluate((x + p.s).clamp(0, 1)[None], noise=False)[0]
    f_b = f.evaluate(x[None], noise=False)[0]
    xs, rows = x.to(dev), []
    for name, (cx, cy, qx) in {
        "oracle_q": (p.xb, p.yb, (xs + p.s.to(dev)).clamp(0, 1)),
        "a_only": (p.xa[:n_show], p.ya[:n_show], xs),
    }.items():
        cx, cy = cx.to(dev), cy.to(dev)
        if len(cx) == 0:
            cx, cy = cx.new_zeros(1, 1), cy.new_zeros(1)
            m = torch.zeros(1, 1, dtype=torch.bool, device=dev)
        else:
            m = torch.ones(1, len(cx), dtype=torch.bool, device=dev)
        nf = torch.ones(1, dtype=torch.long, device=dev)
        kv = pfn.context_kv(cx[None], cy[None], n_features=nf, train_key_padding_mask=m)
        lg = pfn.logits_from_context_kv(kv, qx[None], n_features=nf, train_key_padding_mask=m).float()[0]
        bd = pfn.bar_dist
        q10, q90 = bd.icdf(lg, 0.1).cpu().numpy(), bd.icdf(lg, 0.9).cpu().numpy()
        rows.append(pd.DataFrame({"x": x[:, 0].numpy(), "model": name, "mean": bd.mean(lg).cpu().numpy(),
                                  "q10": q10, "q90": q90}))  # fmt: skip
    rows.append(pd.DataFrame({"x": x[:, 0].numpy(), "model": "f_A", "mean": f_a.numpy()}))
    rows.append(pd.DataFrame({"x": x[:, 0].numpy(), "model": "f_B", "mean": f_b.numpy()}))
    return pd.concat(rows, ignore_index=True)
