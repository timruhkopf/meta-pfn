"""MTPFN (issue #20): model invariances, the robust ICM prior and the exact references."""

import torch

from meta_pfn.experimental.mtpfn import icm as I
from meta_pfn.experimental.mtpfn import train as Tr
from meta_pfn.experimental.mtpfn.model import MTPFN


def _model(head="bar"):
    torch.manual_seed(0)
    return MTPFN(3, torch.linspace(-5, 5, 41), d_model=32, heads=4, n_pairs=2, d_ff=64, head=head).eval()


def _batch(P=3, T=3, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Tr.icm_batch(I.sample_icm(P, T, d, 5, g, "cpu", n_tot=30))


def test_shapes_and_heads():
    x, y, m, xq, yq, nf = _batch()
    assert _model()(x, y, m, xq, nf).shape == (3, 5, 40)
    mg = _model("gauss")
    out = mg(x, y, m, xq, nf)
    assert out.shape == (3, 5, 2) and torch.isfinite(mg.log_density(out, yq)).all()


def test_point_and_source_permutation_invariance():
    x, y, m, xq, _, nf = _batch()
    md = _model()
    ref = md(x, y, m, xq, nf)
    perm = torch.randperm(x.shape[2])
    out = md(x[:, :, perm], y[:, :, perm], m[:, :, perm], xq, nf)
    assert torch.allclose(ref, out, atol=1e-5)
    src = torch.tensor([0, 2, 1])  # swap the two sources; the target stays task 0
    assert torch.allclose(ref, md(x[:, src], y[:, src], m[:, src], xq, nf), atol=1e-5)


def test_padding_is_invisible_and_queries_independent():
    x, y, m, xq, _, nf = _batch()
    md = _model()
    ref = md(x, y, m, xq, nf)
    x2, y2 = x.clone(), y.clone()
    x2[~m] = 0.77
    y2[~m] = 9.0
    assert torch.allclose(ref, md(x2, y2, m, xq, nf), atol=1e-5)
    xq2 = xq.clone()
    xq2[:, 0] += 0.3
    assert torch.allclose(ref[:, 1:], md(x, y, m, xq2, nf)[:, 1:], atol=1e-5)


def test_sources_matter_only_through_task_tokens():
    """Changing a source's y changes the target's prediction (the inter-task route exists)."""
    x, y, m, xq, _, nf = _batch()
    md = _model()
    y2 = y.clone()
    y2[:, 1] += 1.0
    assert not torch.allclose(md(x, y, m, xq, nf), md(x, y2, m, xq, nf))


def test_icm_prior_marginals_and_unrelated_sources():
    g = torch.Generator().manual_seed(1)
    b = I.sample_icm(2000, 3, 1, 4, g, "cpu", counts=[1, 1, 1], p=0.0)
    assert abs(float(b.y[:, 0, 0].var()) - 1.0 - float((b.sig**2).mean())) < 0.1  # unit signal variance + noise
    # p = 0: corr(y_0, y_1) at nearby x follows K_T; p = 1: sources independent of the target
    b1 = I.sample_icm(4000, 2, 1, 1, g, "cpu", counts=[1, 1], p=1.0)
    assert b1.unrel[:, 1].all() and abs(float(torch.corrcoef(b1.y[:, :, 0].T)[0, 1])) < 0.06
    e = I.sample_icm(50, 4, 2, 3, g, "cpu", counts=[5, 20, 20, 20], force_unrel=2)
    assert (e.unrel.sum(1) == 2).all() and not e.unrel[:, 0].any()
    assert e.mask.sum((1, 2)).eq(65).all()
    ell = I._gamma((20000,), g, "cpu")
    assert abs(float(ell.mean()) - 0.5) < 0.02  # Gamma(3, rate 6)


def test_references_order():
    """The oracle ICM has the most information; on average it beats the Bayes predictive, which beats target-only."""
    g = torch.Generator().manual_seed(2)
    b = I.sample_icm(6, 3, 1, 20, g, "cpu", counts=[3, 15, 15], force_unrel=0)
    tot = {}
    for i in range(6):
        lp, ess = I.references(b, i, 0.2, 256, g)
        for k, v in lp.items():
            tot[k] = tot.get(k, 0.0) - float(v.mean()) / 6
        assert ess["ess_bayes"] >= 1.0
    assert tot["oracle_icm"] <= tot["bayes"] + 0.05
    assert tot["bayes"] <= tot["gp_target_bayes"] + 0.05
    assert tot["oracle_icm"] < tot["gp_target_oracle"]
