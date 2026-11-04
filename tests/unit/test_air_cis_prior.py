"""AIR-CIS S0 (issue #22; spec §2 tests, plus the leak check of deviation D1)."""

import numpy as np
import torch
from scipy.stats import chisquare, kstest

from meta_pfn.experimental.air.cis import decoder as CD
from meta_pfn.experimental.air.cis import mt
from meta_pfn.experimental.air.cis import prior as C


def _tasks(P=2000, n=5, seed=0):
    return C.sample_tasks(C.CisCfg(), P, n, torch.Generator().manual_seed(seed))


def test_p_j_frequencies():
    t = _tasks(5000)
    obs = torch.bincount(t.J, minlength=17)[list(C.J_VALUES)].numpy()
    assert chisquare(obs, np.array(C.P_J) * 5000).pvalue > 0.01


def test_ecdf_output_is_marginally_uniform():
    t = _tasks(2000, seed=1)
    v = t.g(torch.rand(2000, 8, generator=torch.Generator().manual_seed(2))).flatten().numpy()
    assert kstest(v, "uniform").statistic < 0.02


def test_j0_is_identity_flow_and_latents_follow_activity():
    x = torch.rand(64, 50)
    lat = torch.randn(64, C.N_LAT) * C.active(torch.zeros(64, dtype=torch.long))
    J = torch.zeros(64, dtype=torch.long)
    assert torch.equal(C.flow(x, lat[:, : C.J_MAX], C.s_t(lat, J)), x)
    w, c = C.window(lat)
    assert torch.allclose(C.transform(x, lat, J), c[:, None] + w[:, None] * x)
    t = _tasks(500, seed=3)
    assert torch.all(t.lat[~C.active(t.J)] == 0)


def test_transform_maps_into_the_box_and_inverts():
    t = _tasks(500, seed=4)
    x = torch.rand(500, 64)
    u = C.transform(x, t.lat, t.J)
    assert float(u.min()) >= 0 and float(u.max()) <= 1
    xb, ld = t.b_in_a()
    ins = (xb >= 0) & (xb <= 1)
    assert float((C.transform(xb.clamp(0, 1), t.lat, t.J) - t.xb).abs()[ins].quantile(0.99)) < 1e-3
    x1, _ = C.inverse(t.xb + 1e-3, t.lat, t.J)
    x0, _ = C.inverse(t.xb - 1e-3, t.lat, t.J)
    ok = ins & (t.xb > 0.01) & (t.xb < 0.99)
    assert float((torch.log((x1 - x0) / 2e-3) - ld).abs()[ok].quantile(0.99)) < 0.02


def test_no_boundary_leak():
    """D1: with h = id, B does not reveal A's values at the box ends (the box-preserving flow made them equal)."""
    t = _tasks(2000, seed=5)

    def gap(e):
        x = torch.full((2000, 1), e)
        return (t.f_a(x) - t.g(x)).abs()[t.hid]

    mid = gap(0.5)
    for e in (0.0, 1.0):  # pinned ends would give d = 0 on every task; here the ends look like the middle
        d = gap(e)
        assert float((d < 1e-3).float().mean()) < 2 * float((mid < 1e-3).float().mean()) + 0.1
        assert float(d.mean()) > 0.5 * float(mid.mean())


def test_prior_tables_are_normalised():
    tab = C.prior_tables()
    nb = tab["n_bins"]
    dens = tab["log_pi"].exp()
    assert torch.allclose(dens.sum(1) / nb, torch.ones(nb), atol=1e-4)
    assert torch.all(tab["s_k"] > 0)
    m, s = C.stretch_stats(torch.tensor([0.01, 0.1]), tab)
    assert m.shape == (2,) and torch.all(s > 0)


def test_decoder_at_n_a_zero_and_mt_batches():
    t = _tasks(4, n=0, seed=6)
    dec = CD.CisDecoder(D=32, layers=1)
    xb_t, ld = t.b_in_a()
    lq, le = dec(xb_t, t.yb, t.xa, t.ya, t.xq, ld=ld)
    assert lq.shape == (4, 32, 400) and le.shape == (4, 0, 400) and torch.isfinite(lq).all()
    assert torch.isfinite(CD.loss(dec, t, xb_t, ld))
    t3 = _tasks(4, n=3, seed=7)
    x, _, mask, *_ = mt.mt_batch(t3)
    assert x.shape == (4, 2, 128, 1) and int(mask[:, 0].sum(-1).max()) == 3
    assert mt.ao_batch(t3)[0].shape == (4, 1, 3, 1)


def test_eval_tasks_reproducible():
    a, b = C.eval_tasks(8, 777, "cpu"), C.eval_tasks(8, 777, "cpu")
    assert torch.equal(a.xa, b.xa) and torch.equal(a.lat, b.lat) and torch.equal(a.yq, b.yq)
