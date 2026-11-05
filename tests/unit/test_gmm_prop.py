"""GMM-token propagation (issue #23): prior, closed-form oracle, model invariances."""

import math

import torch

from meta_pfn.experimental.gmmprop import oracle as O
from meta_pfn.experimental.gmmprop import prior as P
from meta_pfn.experimental.gmmprop.model import GMMPFN


def _batch(B=32, n=12, d=2, Q=16, seed=0):
    return P.sample(B, n, d, Q, torch.Generator().manual_seed(seed))


def test_prior_shapes_and_shuffle():
    b = _batch()
    assert b.x.shape == (32, 12, 2) and b.mu.shape == (32, 12, P.M_MAX, 2) and b.yq.shape == (32, 16, 2)
    # weights sum to one per token, absent components carry zero weight
    assert torch.allclose(b.w.sum(-1), torch.ones(32, 12), atol=1e-5)
    assert (b.w[~b.cmask] == 0).all() and (b.cmask.sum(-1) == b.m[:, None]).all()
    # shuffling permutes each token's components: the multiset of means is preserved
    srt = lambda t: t.sum(-1).sort(-1).values  # noqa: E731
    assert torch.allclose(srt(b.mu), srt(b.mu_lab), atol=1e-6)
    # the query's hypothesis is a present one
    assert (b.mstar < b.m[:, None]).all()


def test_oracle_matches_explicit_gp():
    """M = 1 task, d = 1: the oracle's component for the true length scale equals the textbook GP posterior."""
    b = _batch(B=64, n=10, d=1, Q=5, seed=1)
    i = int(torch.nonzero(b.m == 1)[0])
    t = b.task(i)
    p = O.predictive(t)
    ell = P.ells(1)
    x, xq, y, s = t.x[0].double(), t.xq[0].double(), t.mu_lab[0, :, 0].double(), t.s[0].double()
    for li, e in enumerate(ell):
        k = lambda a, c, e=e: P.ALPHA**2 * torch.exp(-0.5 * (a - c.T) ** 2 / e**2)  # noqa: E731
        Kxx = k(x, x) + torch.diag(s**2)
        kq = k(xq, x)
        mean = kq @ torch.linalg.solve(Kxx, y)
        var = P.ALPHA**2 * (1 + P.JIT) - (kq @ torch.linalg.solve(Kxx, kq.T)).diagonal()
        c = li * P.M_MAX  # hypothesis 0 of length scale li
        assert torch.allclose(p["mean"][0, :, c], mean, atol=1e-6)
        assert torch.allclose(p["var"][0, :, c], var, atol=1e-8)
    # hypotheses 1, 2 are absent
    assert torch.isinf(p["logw"][0, :, 1]).all() and torch.isinf(p["logw"][0, :, 2]).all()


def test_oracle_is_calibrated_and_best():
    """On prior draws the oracle beats its factorised version and the prior (Bayes optimality, in expectation),
    and for M = 1 the standardised errors of its dominant component are ~ N(0, 1)."""
    b = _batch(B=512, n=24, d=3, Q=32, seed=2)
    r = O.references(b)
    nll = {k: -float(v.mean()) for k, v in r.items()}
    assert nll["oracle"] < nll["oracle_fact"] - 0.05 < nll["prior"]
    one = b.m == 1
    p = O.predictive(b)
    top = p["logw"].argmax(-1)  # [B, Q]
    mean = torch.gather(p["mean"], 2, top[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    var = torch.gather(p["var"], 2, top[..., None])
    z = ((b.yq.double() - mean) / var.sqrt())[one]
    assert abs(float(z.mean())) < 0.05 and abs(float(z.var()) - 1) < 0.15


def test_oracle_n0_is_prior():
    b = _batch(n=0)
    r = O.references(b)
    assert torch.allclose(r["oracle"], r["prior"], atol=1e-9) and torch.allclose(
        r["oracle_fact"], r["prior"], atol=1e-9
    )


def test_model_invariances():
    torch.manual_seed(0)
    m = GMMPFN(2, D=32, layers=2, K=3).eval()
    b = _batch(B=4, n=7, d=2, Q=6)
    with torch.no_grad():
        p = m(b)
        lp = m.log_density(p, b.yq)
        assert lp.shape == (4, 6) and torch.isfinite(lp).all()
        assert torch.allclose(p["logw"].exp().sum(-1), torch.ones(4, 6), atol=1e-5)
        # permuting context tokens, or the components within the tokens, changes nothing
        pc = torch.randperm(7)
        b2 = P.Batch(
            **{k: (v[:, pc] if k in ("x", "w", "mu", "s", "cmask", "mu_lab") else v) for k, v in b.__dict__.items()}
        )
        pm = torch.randperm(P.M_MAX)
        b2.w, b2.mu, b2.cmask = b2.w[..., pm], b2.mu[:, :, pm], b2.cmask[..., pm]
        assert torch.allclose(m.log_density(m(b2), b.yq), lp, atol=1e-4)
        # queries do not see each other: dropping queries leaves the others unchanged
        b3 = P.Batch(**{k: (v[:, :2] if k in ("xq", "yq", "mstar") else v) for k, v in b.__dict__.items()})
        assert torch.allclose(m.log_density(m(b3), b3.yq), lp[:, :2], atol=1e-4)
        # n = 0 works (null token only)
        assert torch.isfinite(m.log_density(m(b.head(0)), b.yq)).all()
        # marginals of one Gaussian per dimension sum to the joint for K = 1
    m1 = GMMPFN(2, D=16, layers=1, K=1).eval()
    with torch.no_grad():
        p1 = m1(b)
        assert torch.allclose(m1.log_marginals(p1, b.yq).sum(-1), m1.log_density(p1, b.yq), atol=1e-4)
    assert math.isfinite(float(lp.mean()))
