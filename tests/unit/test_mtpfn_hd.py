"""The generalised MTPFN prior and its exact oracle (experimental/mtpfn/hd.py, notebook 16)."""

import math

import torch

from meta_pfn.experimental.air import reference as R
from meta_pfn.experimental.mtpfn import hd


def test_h_marginal_matches_brute_force():
    torch.manual_seed(0)
    for n in (1, 3):
        ya, m, sig = torch.rand(n), torch.rand(3, n), torch.tensor(0.05)
        _, lb, _, _ = hd.h_terms(ya, m, sig, hd.b_grid(2048, "cpu"))
        an = torch.logsumexp(lb, -1) - math.log(2048)
        b = torch.linspace(0.3, 1, 801)[1:-1].double()
        u = torch.linspace(0, 1, 801)[1:-1].double()
        bb, uu = torch.meshgrid(b, u, indexing="ij")
        aa = uu * (1 - bb)  # the prior is uniform in (b, u)
        for k in range(3):
            r = ya.double() - aa[..., None] - bb[..., None] * m[k].double()
            ll = (-0.5 * (r / 0.05) ** 2).sum(-1) - n * (math.log(0.05) + 0.5 * math.log(2 * math.pi))
            assert abs(float(torch.logsumexp(ll.flatten(), 0) - math.log(ll.numel())) - float(an[k])) < 0.02


def test_oracle_with_h_id_is_the_exact_grid_predictive():
    t = hd.eval_tasks(hd.HDCfg(d=1, pi_id=1.0), 2, 777, "cpu").head(3)
    th = R.grid(101)
    for j in range(2):
        ll = (-0.5 * ((t.ya[j] - hd._m_of(t, j, th, t.xa[j])) / t.sig_a[j]) ** 2).sum(-1)
        lpost = hd._log_n01(th) + ll
        lpost = lpost - torch.logsumexp(lpost, 0)
        ex = torch.logsumexp(lpost[:, None] + R.bucket_lp(hd._m_of(t, j, th, t.xq[j]), t.sig_a[j], t.yq[j][None]), 0)
        o = hd.oracle_lp(t, j, 1.0)
        assert abs(float(o["lp"].mean() - ex.mean())) < 0.03


def test_tasks_and_batches():
    for d in (1, 2):
        t = hd.eval_tasks(hd.HDCfg(d=d), 4, 1, "cpu")
        assert t.xa.shape == (4, hd.N_A_TRAIN[d], d) and t.xb.shape == (4, 128, d)
        assert float(t.xq.min()) >= 0 and float(t.xq.max()) <= 1
        w = hd.warp(t.xq, t.theta, d, t.s)
        assert float(w.min()) >= -1e-5 and float(w.max()) <= 1 + 1e-5  # box-preserving
        assert torch.allclose(t.f_a(t.xq), t.yq, atol=0.15)  # noise-free f_A vs noisy y
        x, _, mask, *_ = hd.mt_batch(t.head(3))
        assert x.shape == (4, 2, 128, d) and int(mask[:, 0].sum(-1).max()) == 3
        x, *_ = hd.ao_batch(t.head(3))
        assert x.shape == (4, 1, 3, d)
    t2 = hd.eval_tasks(hd.HDCfg(d=1), 4, 1, "cpu")
    assert torch.equal(t2.xa, hd.eval_tasks(hd.HDCfg(d=1), 4, 1, "cpu").xa)  # CPU-drawn: reproducible
