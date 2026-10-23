"""Predictors for A's held-out queries, all computed with one frozen PFN (pfn_large).

For a pair and each n_A in the grid (A's first n_A points):
- `lower`:    PFN(A) at x*                              A alone: what B must beat
- `naive`:    PFN(B ∪ A) at x*                          B pooled without registration
- `oracle_T`: PFN(B ∪ T(A)) at T(x*), the true T        registration known (upper bound)
- `bayes`:    the posterior predictive of a learner that knows the pair prior but not the pair: a mixture
  over hypotheses h ∈ {unrelated, identity, window θ_1..K, warp θ_1..K} (θ from the prior), each
  predicting like `lower` (unrelated) or like `oracle_T` with its own T, weighted by
  p(h) · p(D_A | D_B, h). The likelihood uses the PFN's chain rule,
  log p(D_A | D_B, h) = Σ_i log PFN(y_i | B ∪ T_h(A_<i), T_h(x_i)), which is exact for a Bayes-optimal
  PFN (in the order of A's design).
The gap lower → bayes is what B is worth to a Bayes learner that must infer the relation: an amortized
Meta-PFN can at best realize it. bayes → oracle_T is the price of not knowing T.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .pairs import WarpFamily, apply_warp


def logp(
    pfn, X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, Xq: torch.Tensor, Yq: torch.Tensor, chunk: int
) -> torch.Tensor:
    """log PFN density of Yq at Xq given contexts (X, Y, mask M); empty contexts allowed. -> [L, Q]."""
    out = []
    d = X.shape[-1]
    for s in range(0, len(X), chunk):
        sl = slice(s, s + chunk)
        x, y, m = X[sl], Y[sl], M[sl]
        if x.shape[1] == 0:  # one masked dummy token: attention to nothing = the PFN's no-data prediction
            x, y, m = x.new_zeros(len(x), 1, d), y.new_zeros(len(y), 1), m.new_zeros(len(m), 1)
        nf = torch.full((len(x),), d, dtype=torch.long, device=x.device)
        kv = pfn.context_kv(x, y, n_features=nf, train_key_padding_mask=m)
        lg = pfn.logits_from_context_kv(kv, Xq[sl], n_features=nf, train_key_padding_mask=m).float()
        out.append(-pfn.bar_dist(lg, Yq[sl]))
    return torch.cat(out)


@dataclass
class Hypotheses:
    kinds: list[str]  # per hypothesis: unrelated | identity | window | warp
    theta: torch.Tensor  # [H, d, 4]
    log_prior: torch.Tensor  # [H]
    uses_b: torch.Tensor  # [H] bool


def hypotheses(d: int, family: WarpFamily, k: int, prior_mass: dict[str, float], g: torch.Generator) -> Hypotheses:
    kinds, th, lp = [], [], []
    for kind, mass in prior_mass.items():
        n = 1 if kind in ("unrelated", "identity") else k
        kinds += [kind] * n
        th.append(family.sample(kind, n, d, g))
        lp += [np.log(mass) - np.log(n)] * n
    kinds_t = np.asarray(kinds)
    return Hypotheses(
        kinds, torch.cat(th), torch.tensor(lp, dtype=torch.float32), torch.from_numpy(kinds_t != "unrelated")
    )


def _contexts(xb, yb, xa, ya, theta, uses_b, n):
    """Contexts [H, n_B + n, d] (B first, then T_h(A_<n)), with masks; B masked out where uses_b is False."""
    H = len(theta)
    ta = apply_warp(theta, xa[None, :n].expand(H, -1, -1)) if n else xa.new_zeros(H, 0, xa.shape[-1])
    X = torch.cat([xb[None].expand(H, -1, -1), ta], 1)
    Y = torch.cat([yb[None].expand(H, -1), ya[None, :n].expand(H, -1)], 1)
    M = torch.cat([uses_b[:, None].expand(-1, len(xb)), torch.ones(H, n, dtype=torch.bool)], 1)
    return X, Y, M


@torch.no_grad()
def evaluate_pair(pfn, pair, j: int, hyp: Hypotheses, n_grid: list[int], chunk: int = 512) -> dict[str, np.ndarray]:
    """-> {"nll": [len(n_grid), 4] mean query NLL of (lower, naive, oracle_T, bayes);
    "post": [len(n_grid), 4] posterior mass of (unrelated, identity, window, warp);
    "ess": [len(n_grid), 2] effective sample size within the window and warp particles;
    "kind_logz": [len(n_grid), 4] log p(A_<n | D_B, kind) per kind (unrelated, identity, window, warp);
    "kind_logq": [len(n_grid), 4, Q] log predictive density of each query given the kind}."""
    xb, yb, xa, ya, xq, yq = pair.XB[j], pair.YB[j], pair.XA[j], pair.YA[j], pair.XQ[j], pair.YQ[j]
    n_max = max(n_grid)
    true = Hypotheses(
        ["truth", "naive", "lower"],
        torch.stack([pair.theta[j], hyp.theta[hyp.kinds.index("identity")], hyp.theta[0]]),
        torch.zeros(3),
        torch.tensor([True, True, False]),
    )

    # chain-rule log likelihood of A's points, for every hypothesis: ll[h, i] = log p(y_i | ctx_h(A_<i))
    ll = torch.zeros(len(hyp.kinds), n_max)
    for i in range(n_max):
        X, Y, M = _contexts(xb, yb, xa, ya, hyp.theta, hyp.uses_b, i)
        q = apply_warp(hyp.theta, xa[None, i : i + 1].expand(len(hyp.kinds), -1, -1))
        ll[:, i] = logp(pfn, X, Y, M, q, ya[i : i + 1][None].expand(len(hyp.kinds), -1), chunk)[:, 0]

    nll, post, ess, kz, kq = [], [], [], [], []
    kinds = np.asarray(hyp.kinds)
    for n in n_grid:
        X, Y, M = _contexts(xb, yb, xa, ya, hyp.theta, hyp.uses_b, n)
        q = apply_warp(hyp.theta, xq[None].expand(len(kinds), -1, -1))
        lq = logp(pfn, X, Y, M, q, yq[None].expand(len(kinds), -1), chunk)  # [H, Q]
        lw = hyp.log_prior + ll[:, :n].sum(1)
        w = lw - torch.logsumexp(lw, 0)
        bayes = torch.logsumexp(w[:, None] + lq, 0)  # [Q]
        Xt, Yt, Mt = _contexts(xb, yb, xa, ya, true.theta, true.uses_b, n)
        qt = apply_warp(true.theta, xq[None].expand(3, -1, -1))
        lt = logp(pfn, Xt, Yt, Mt, qt, yq[None].expand(3, -1), chunk)  # truth, naive, lower
        nll.append([-float(lt[2].mean()), -float(lt[1].mean()), -float(lt[0].mean()), -float(bayes.mean())])
        pw = w.exp().numpy()
        post.append([float(pw[kinds == k].sum()) for k in ("unrelated", "identity", "window", "warp")])
        e = []
        for k in ("window", "warp"):
            wk = pw[kinds == k]
            e.append(float(wk.sum() ** 2 / max((wk**2).sum(), 1e-300)))
        ess.append(e)
        z, qk = [], []
        for k in KINDS4:  # within-kind posterior: drop the kind's prior mass, keep the 1/K particle weights
            m = torch.from_numpy(kinds == k)
            lk = lw[m] - torch.logsumexp(hyp.log_prior[m], 0)
            z.append(float(torch.logsumexp(lk, 0)))
            qk.append(torch.logsumexp((lk - torch.logsumexp(lk, 0))[:, None] + lq[m], 0).numpy())
        kz.append(z)
        kq.append(np.stack(qk))
    return {
        "nll": np.asarray(nll),
        "post": np.asarray(post),
        "ess": np.asarray(ess),
        "kind_logz": np.asarray(kz),
        "kind_logq": np.stack(kq),
    }


PREDICTORS = ("lower", "naive", "oracle_T", "bayes")
KINDS4 = ("unrelated", "identity", "window", "warp")
