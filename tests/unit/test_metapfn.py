import pytest
import torch

from meta_pfn.experimental.metapfn import train as T
from meta_pfn.experimental.metapfn.models import RegE2E
from meta_pfn.experimental.metapfn.prior import MixPrior, eval_batch, shuffle_b
from meta_pfn.experimental.regpfn.prior import PriorCfg


def _batch(stratum="C11", n=4, d=1):
    return MixPrior(PriorCfg(d=d), 3, 0, "cpu").sample(stratum, n, 5, 16)


def test_unrelated_has_no_correspondence_and_shuffle_keeps_values():
    b = _batch("U")
    assert not b.inside.any()
    s = shuffle_b(_batch("C00"), torch.Generator().manual_seed(0))
    assert torch.equal(s.yb.sort(1).values, _batch("C00").yb.sort(1).values)


def test_m0_context_roles_and_masks():
    b = _batch(n=4)
    xc, _, role, m = T.m0_context(b, 4, True, True)
    assert xc.shape[1] == 4 + 16 + 16 and role[0, 0] == 0 and role[0, 4] == 1 and role[0, -1] == 2
    assert torch.equal(m[:, -16:], b.inside)
    xc, _, role, m = T.m0_context(b, 0, False, False)  # nothing: one masked dummy
    assert xc.shape[1] == 1 and not m.any()


def test_m3_identity_at_n0_and_loo_evidence():
    torch.manual_seed(0)
    model = RegE2E(1, T.load_pfn(None, 1, "cpu"), n_iter=3).eval()
    b = _batch(n=0)
    _, _, hist = model(b.xa, b.ya, 0, b.ub, b.yb, b.xa, b.ya)
    assert torch.equal(hist[-1][0], b.ub) and torch.equal(hist[-1][1], b.yb)  # no A evidence: the field stays
    b = _batch(n=3)
    ev = model.evidence(b.xa, b.ya, 3, [None])[:, 0]
    ref = torch.zeros(3)
    for j in range(3):
        keep = [i for i in range(3) if i != j]
        ref += model.logp(b.xa[:, keep], b.ya[:, keep], torch.ones(3, 2, dtype=torch.bool), None, None,
                          b.xa[:, j : j + 1], b.ya[:, j : j + 1])[:, 0]  # fmt: skip
    assert torch.allclose(ev, ref, atol=1e-4)


@pytest.mark.parametrize(("mode", "priv"), [("m0", "none"), ("m0", "dropout"), ("m0", "rb"), ("m3", "none")])
def test_train_and_evaluate_smoke(mode, priv, monkeypatch):
    model, _ = T.train(mode, 1, 3, "cpu", batch=2, n_b=16, n_test=4, priv=priv, p_priv=1.0, n_iter=2)
    monkeypatch.setattr(T, "STRATA", ("C11", "U"))
    ev, slots = T.evaluate(model, mode, 1, "cpu", T.load_pfn(None, 1, "cpu"), n_pairs=2, n_b=16)
    assert ev.nll.notna().all() and {"model", "model_aonly", "model_shuffled", "pfn_aonly", "pfn_pool"} <= set(
        ev.variant
    )
    assert (len(slots) > 0) == (mode == "m3")


def test_eval_batch_u_shares_a_with_c11():
    a, u = eval_batch(1, "C11", 2, 16, 0, "cpu"), eval_batch(1, "U", 2, 16, 0, "cpu")
    assert torch.equal(a.xa, u.xa) and not torch.equal(a.yb, u.yb)
