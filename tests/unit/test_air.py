"""AIR invariants (issue #19): the transform family (V0), the decoder's masks (no leakage into the chain-rule evidence,
null head independent of B), the grid reference, and the estimator's plumbing."""

import pytest
import torch

from meta_pfn.experimental.air import prior as Pr
from meta_pfn.experimental.air import reference as R
from meta_pfn.experimental.air.decoder import Decoder
from meta_pfn.experimental.air.estimator import decode, importance
from meta_pfn.experimental.air.proposal import Proposal


@pytest.mark.parametrize("J", [2, 16])
def test_flow_is_a_monotone_box_bijection_with_exact_inverse(J: int) -> None:
    s = Pr.s_t_for(J)
    th = torch.randn(200, J, generator=torch.Generator().manual_seed(0))
    x = torch.linspace(0, 1, 101).expand(200, -1)
    t = Pr.flow(x, th, s)
    assert bool((t.diff(dim=1) >= 0).all())
    assert float(t[:, 0].abs().max()) < 1e-6 and float((t[:, -1] - 1).abs().max()) < 1e-6
    assert float((Pr.flow(t, th, s, inverse=True) - x).abs().max()) < 1e-4
    assert 0.10 < float((t - x).abs().mean()) < 0.16


def test_evidence_queries_do_not_see_their_own_or_later_points() -> None:
    torch.manual_seed(0)
    dec = Decoder(D=32, layers=2).eval()
    t = Pr.sample_tasks(Pr.PriorCfg(J=2, n_b=16, n_q=4), 3, 5, torch.Generator().manual_seed(1), "cpu")
    rank = torch.arange(5).expand(3, -1)  # identity order
    xb_t = t.b_in_a(t.theta)
    _, le = dec(xb_t, t.yb, t.xa, t.ya, t.xq, rank=rank)
    ya2 = t.ya.clone()
    ya2[:, 3:] += 0.5  # change y_3, y_4 and x_4 (x_3 is the evidence query's own conditioning input)
    xa2 = t.xa.clone()
    xa2[:, 4] = torch.rand(3)
    _, le2 = dec(xb_t, t.yb, xa2, ya2, t.xq, rank=rank)
    torch.testing.assert_close(le[:, :4], le2[:, :4])  # evidence of points 0..3 sees only points < itself
    assert not torch.allclose(le[:, 4], le2[:, 4])


def test_null_head_ignores_b() -> None:
    torch.manual_seed(0)
    dec = Decoder(D=32, layers=2).eval()
    t = Pr.sample_tasks(Pr.PriorCfg(J=2, n_b=16, n_q=4), 2, 3, torch.Generator().manual_seed(2), "cpu")
    rank = torch.arange(3).expand(2, -1)
    a = dec(t.b_in_a(t.theta), t.yb, t.xa, t.ya, t.xq, rank=rank, use_b=False)
    b = dec(torch.rand(2, 16), torch.rand(2, 16), t.xa, t.ya, t.xq, rank=rank, use_b=False)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v)


def test_grid_reference_concentrates_near_the_truth() -> None:
    t = Pr.sample_tasks(Pr.PriorCfg(J=2), 4, 10, torch.Generator().manual_seed(3), "cpu")
    pts = R.grid(41)
    errs = []
    for j in range(4):
        gr = R.grid_ref(t, j, pts)
        mean = (gr["logpost"].exp()[:, None] * pts).sum(0)
        errs.append(float((mean - t.theta[j]).norm()))
        assert torch.isfinite(gr["lp"]).all()
    assert sum(e < 1.5 for e in errs) >= 3  # 10 points pin a 2-d transform in most tasks


def test_estimator_shapes_and_proposal_mixture() -> None:
    torch.manual_seed(0)
    dec = Decoder(D=32, layers=2).eval()
    q = Proposal(2, D=16, hidden=32, transforms=2).eval()
    t = Pr.sample_tasks(Pr.PriorCfg(J=2, n_b=16, n_q=4), 3, 4, torch.Generator().manual_seed(4), "cpu")
    th, lq = q.sample_mix(t.xb, t.yb, t.xa, t.ya, 6)
    r = importance(dec, t, th, lq)
    assert r["lp1"].shape == (3, 4) and torch.isfinite(r["lp1"]).all()
    assert bool(((r["ess"] > 0) & (r["ess"] <= 1 + 1e-6)).all())
    ev, _, sd = decode(dec, t, t.theta[:, None], n_perm=3)
    assert ev.shape == (3, 1) and sd.shape == (3, 1)


def test_ecdf_prior_is_global_and_does_not_force_extremes() -> None:
    """g_norm = ecdf (pilot 2): one fixed monotone map for all functions; values in [0, 1]; extremes not forced."""
    g = torch.Generator().manual_seed(7)
    f = Pr.draw_funcs(500, g, "cpu", "ecdf")
    x = torch.linspace(0, 1, 256).expand(500, -1)
    v = f(x)
    assert float(v.min()) >= 0 and float(v.max()) <= 1
    assert float((v.max(1).values > 0.99).float().mean()) < 0.2  # min-max would give 1.0
    assert torch.equal(Pr.prior_ecdf("cpu"), Pr.prior_ecdf("cpu"))  # cached, deterministic
    r = f.raw(x)
    assert bool(((v.diff(dim=1) >= -1e-6) == (r.diff(dim=1) >= -1e-6)).float().mean() > 0.99)  # monotone in raw
    t = Pr.sample_tasks(Pr.PriorCfg(J=2, n_b=16, n_q=4, g_norm="ecdf"), 3, 4, torch.Generator().manual_seed(8), "cpu")
    assert t.g.ecdf is not None and t.g.at(torch.tensor([0, 1])).ecdf is not None
