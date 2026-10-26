import pytest
import torch

from meta_pfn.experimental import fieldonly as F
from meta_pfn.experimental.regpfn.train import eval_set


def test_truth_field_has_zero_error_and_low_energy():
    b = eval_set(1, "C10", 8, 8, 64, 0, "cpu")
    ex, ey = F.field_rms(b.xb_inA, b.yb_inA, b)
    assert ex.max() < 1e-6 and ey.max() < 1e-6
    x0, y0 = F.init_field(b, torch.Generator().manual_seed(0))
    n = 10
    e_true = F.energy(b.xa[:, :n], b.ya[:, :n], b.xb_inA, b.yb_inA, 0.02)
    e_start = F.energy(b.xa[:, :n], b.ya[:, :n], x0, y0, 0.02)
    assert e_true.median() < e_start.median()


def test_identity_start_is_the_truth_in_c00():
    b = eval_set(1, "C00", 8, 8, 64, 0, "cpu")
    x0, _ = F.init_field(b, torch.Generator().manual_seed(0), identity=True)
    assert F.field_rms(x0, b.yb_inA, b)[0].max() < 1e-6


def test_monotone_map_starts_affine_and_stays_increasing():
    u = torch.rand(3, 50, 1)
    c, a = torch.tensor([0.1, -0.2, 0.0]), torch.tensor([0.8, 1.3, 1.0])
    theta = F._softplus_inv(a)[:, None].repeat(1, 8)
    x = F.monotone_map(u, c, theta)
    assert torch.allclose(x, c[:, None, None] + a[:, None, None] * u, atol=1e-5)
    theta = torch.randn(3, 8)
    us = torch.linspace(-0.2, 1.2, 100)[None, :, None].expand(3, -1, -1)
    assert (F.monotone_map(us, c, theta).diff(dim=1) > 0).all()


def test_gd_recovers_a_shift_in_c10():
    # a pure shift is in gd's model class: from a start off by the shift it should move most pairs closer
    b = eval_set(1, "C10", 16, 8, 64, 0, "cpu")
    x0, _ = F.init_field(b, torch.Generator().manual_seed(1))
    xh, yh, _, _ = F.gd_fit(b.xa[:, :20], b.ya[:, :20], b, x0, steps=200)
    assert F.field_rms(xh, yh, b)[0].median() < F.field_rms(x0, b.yb, b)[0].median()


def test_grad_feats_shape_and_constant():
    b = eval_set(1, "C10", 2, 8, 12, 0, "cpu")
    x = b.ub.clone().requires_grad_()
    g = F.grad_feats(b.xa[:, :5], b.ya[:, :5], x, b.yb)
    assert g.shape == (2, 12, 3 * len(F.LENS)) and not g.requires_grad


@pytest.mark.parametrize("method", ["operator", "optimizer"])
def test_learned_smoke(method, monkeypatch):
    model = F.make_model(method, n_iter=2, dm=16)
    logged = []
    hist = F.train_learned(model, method, 4, "cpu", batch=2, n_b=12, every=2, metrics=lambda v, s: logged.append(s))
    assert logged == [2, 4] and "val/frac_005_n10" in hist
    monkeypatch.setattr(F, "N_A_EVAL", (2,))
    ev = F.evaluate(method, model, "cpu", n_pairs=3, n_b=12)
    assert set(ev.method) == {method} and set(ev.init) == set(F.INITS) and ev.err_x.notna().all()
    pts, trs = F.examples(method, model, "cpu", n_pairs=2, n_b=12)
    assert {"A", "B", "A_test"} <= set(pts.role) and trs.t.max() == 2


def test_gd_eval_has_references(monkeypatch):
    monkeypatch.setattr(F, "N_A_EVAL", (2,))
    ev = F.evaluate("gd", None, "cpu", n_pairs=3, n_b=12, restarts=2, gd_steps=5)
    assert {"gd", "gd_best8", "gd_oracle8", "start", "truth"} == set(ev.method)
    t = ev.loc[ev.method == "truth"]
    assert t.err_x.max() < 1e-6
