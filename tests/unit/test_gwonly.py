import pytest
import torch

from meta_pfn.experimental import gwonly as G
from meta_pfn.experimental.regpfn.train import eval_set


def test_gw_grad_matches_autograd():
    b = eval_set(1, "C10", 4, 8, 12, 0, "cpu")
    n = 3
    sa, sb = G.relations(b.xa[:, :n], b.ya[:, :n]), G.relations(b.ub, b.yb)
    pi = torch.softmax(torch.randn(4, n, 12), -1).requires_grad_()
    diff = ((sa[:, :, None, :, None, :] - sb[:, None, :, None, :, :]) ** 2).sum(-1)
    e = (diff * pi[:, :, :, None, None] * pi[:, None, None, :, :]).sum((1, 2, 3, 4)) / n
    (g,) = torch.autograd.grad(e.sum(), pi)
    assert torch.allclose(g, G.gw_grad(sa, sb, pi.detach()), atol=1e-4)


def test_truth_is_identity_in_c00():
    b = eval_set(1, "C00", 8, 8, 256, 0, "cpu")
    u, cov = G.truth(b, 5)
    assert (u - b.xa[:, :5, 0])[cov].abs().max() < 0.01


def test_peyre_finds_a_planted_copy():
    # B = A's points plus distractor-free copies: the order-consistent matching is the identity permutation
    x = torch.rand(2, 6, 1)
    y = torch.rand(2, 6)
    pi, _ = G.peyre(G.relations(x, y), G.relations(x, y), n_iter=300)
    assert (pi.argmax(-1) == torch.arange(6)).float().mean() > 0.9


@pytest.mark.parametrize("method", ["unrolled", "gru"])
def test_learned_smoke(method, monkeypatch):
    model = G.LearnedGW(n_iter=3, gru=method == "gru")
    logged = []
    hist = G.train_learned(model, 4, "cpu", batch=2, n_b=12, every=2, metrics=lambda v, s: logged.append(s))
    assert logged == [2, 4] and "val/frac_005_n5" in hist
    monkeypatch.setattr(G, "N_A_EVAL", (2,))
    ev = G.evaluate(method, model, "cpu", n_pairs=3, n_b=12)
    assert set(ev.method) == {method} and ev.err_x.notna().all()
    pts, cps = G.examples(method, model, "cpu", n_pairs=2, n_b=12)
    assert {"A", "B", "A_test"} <= set(pts.role) and abs(
        cps.groupby(["cell", "n_a", "pair", "row"]).pi.sum() - 1
    ).max() < 1e-4


def test_peyre_eval_has_baselines(monkeypatch):
    monkeypatch.setattr(G, "N_A_EVAL", (2,))
    ev = G.evaluate("peyre", None, "cpu", n_pairs=3, n_b=12, restarts=2)
    assert {"peyre", "peyre_oracle_k", "identity", "value"} == set(ev.method)
