"""A→B S3/S4 invariants (issue #18): the prior's T is a monotone full-box bijection, h⁻¹ and its Jacobian are exact,
the model starts at the identity (T_k = id, h_k = id), so on identity pairs it equals the oracle particle, and the
interpolated bar density matches the PFN's at bin centres."""

import pytest
import torch
from pfn_repro.checkpoints import load

from meta_pfn.experimental.a2b.model import A2B, MonotoneH, mix_eps, mixture_lp
from meta_pfn.experimental.a2b.prior import PairPrior, PriorCfg, apply_t, h_fwd, h_inv_logjac
from meta_pfn.experimental.a2b.train import oracle_lp


@pytest.fixture(scope="module")
def pfn():
    return load("pfn_smoke")[0]


def test_t_is_a_monotone_bijection_of_the_box() -> None:
    inst = PairPrior(PriorCfg(d=2), 8, 0, "cpu").sample("warp", 4, 4, 8)
    x = torch.linspace(0, 1, 101)[None, :, None].expand(8, -1, 2)
    t = apply_t(x, inst.t_al, inst.t_be)
    assert bool((t.diff(dim=1) >= 0).all()) and float(t[:, 0].max()) < 0.02 and float(t[:, -1].min()) > 0.98


@pytest.mark.parametrize("kind", ["affine", "kum"])
def test_h_inverse_and_jacobian(kind: str) -> None:
    inst = PairPrior(PriorCfg(d=1, h_kind=kind), 6, 1, "cpu").sample("warp", 2, 2, 4)
    u = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)[None].expand(6, -1)
    par = inst.h_par.double()
    y = h_fwd(u, kind, par)
    ui, lj = h_inv_logjac(y, kind, par)
    torch.testing.assert_close(ui, u, atol=1e-6, rtol=0)
    du = 1e-6
    num = (h_inv_logjac(y + du, kind, par)[0] - h_inv_logjac(y - du, kind, par)[0]) / (2 * du)
    torch.testing.assert_close(lj, num.log(), atol=1e-4, rtol=0)


def test_monotone_h_starts_at_the_identity() -> None:
    h = MonotoneH(16, J=8)
    y = torch.rand(3, 2, 20)
    u, lj = h.inv(torch.randn(3, 2, 16), y)
    torch.testing.assert_close(u, y, atol=1e-3, rtol=0)
    assert float(lj.detach().abs().max()) < 1e-2


def test_model_starts_at_the_oracle_on_identity_pairs(pfn) -> None:
    inst = PairPrior(PriorCfg(d=1), 4, 2, "cpu").sample("identity", 5, 6, 12)
    model = A2B(pfn, 1, D=32, M=4, K=3, R=1, null=False)
    outs = model(inst.xb, inst.yb, inst.xa, inst.ya, inst.xq, inst.yq)
    torch.testing.assert_close(outs[0]["ta"][:, 0], inst.xa, atol=1e-4, rtol=0)
    kv, nf, _ = model.qb.encode(inst.xb, inst.yb)
    torch.testing.assert_close(
        mixture_lp(outs[0]), oracle_lp(model.qb, kv, nf, inst, inst.xq, inst.yq), atol=2e-3, rtol=0
    )


def test_interpolated_density_matches_bar_distribution_at_centres(pfn) -> None:
    model = A2B(pfn, 1, D=32, M=4, K=2, R=0)
    lg = torch.randn(3, 5, pfn.bar_dist.borders.numel() - 1)
    c = model.qb.centers[[3, 10, 20, 30, 40]].expand(3, 5)
    torch.testing.assert_close(model.qb.logpdf(lg, c), -pfn.bar_dist(lg, c), atol=1e-4, rtol=0)


def test_null_particle_is_the_a_only_pfn_chain_rule(pfn) -> None:
    inst = PairPrior(PriorCfg(d=2), 3, 4, "cpu").sample("warp", 4, 5, 6)
    model = A2B(pfn, 2, D=32, M=4, K=2, R=0)
    nul = model.null_particle(inst.xa, inst.ya, inst.xq, inst.yq)

    def ref(i, xq, yq):  # PFN(A_<i) at xq via the PFN's own forward; empty context = one masked (0, 0) token
        x, y = inst.xa[:, :i], inst.ya[:, :i]
        m = torch.ones_like(y, dtype=torch.bool)
        if i == 0:
            x, y, m = x.new_zeros(3, 1, 2), y.new_zeros(3, 1), torch.zeros(3, 1, dtype=torch.bool)
        with torch.no_grad():
            lg = pfn(x, y, xq, n_features=torch.full((3,), 2), train_key_padding_mask=m).float()
        return mix_eps(model.qb.logpdf(lg, yq))

    ev = sum(ref(i, inst.xa[:, i : i + 1], inst.ya[:, i : i + 1])[:, 0] for i in range(4))
    torch.testing.assert_close(nul["ev"], ev, atol=1e-4, rtol=0)
    torch.testing.assert_close(nul["lq"], ref(4, inst.xq, inst.yq), atol=1e-4, rtol=0)
