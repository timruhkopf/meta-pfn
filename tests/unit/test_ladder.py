"""S0/S1 plumbing identities (issue #16): A's inputs stay in T's preimage of the box, the B-space teacher with
no A points is the oracle particle, the exact chain rule's first point is the pseudo-likelihood, and padding is
invisible."""

import pandas as pd
import pytest
import torch
from pfn_repro.checkpoints import load

from meta_pfn.experimental import ladder as L


@pytest.fixture(scope="module")
def pfn():
    return load("pfn_smoke")[0]


@pytest.mark.parametrize("d", [1, 2])
def test_pairs_are_reproducible_and_shift_stays_in_the_box(d: int) -> None:
    a = L.draw_pair(d, "shift", 0, 3, 8, 5, 6, 0.2)
    b = L.draw_pair(d, "shift", 0, 3, 8, 5, 6, 0.2)
    torch.testing.assert_close(a.ya, b.ya)
    assert bool((a.s.abs() <= 0.2).all())
    t = torch.cat([a.xa, a.xq]) + a.s
    assert bool((t >= 0).all() and (t <= 1).all())
    assert bool((L.draw_pair(d, "identity", 0, 3, 8, 5, 6, 0.2).s == 0).all())


def test_s0_s1_identities(pfn) -> None:
    p = L.draw_pair(1, "shift", 0, 0, 12, 4, 6, 0.2)
    grid = L.shift_grid(1, 0.2, 21, 5)
    r = L.evaluate_pair(pfn, p, grid, [0, 1, 4], chunk=64)
    nll = pd.DataFrame(r["nll"]).pivot_table(index="n_a", columns="predictor", values="nll")
    assert abs(nll.loc[0, "teacher_b"] - nll.loc[0, "oracle_q"]) < 1e-5  # B ∪ ∅ at T(x*) is the oracle particle
    assert abs(nll.loc[0, "grid_exact"] - nll.loc[0, "grid_pseudo"]) < 1e-5  # no A points: same mixture
    post = pd.DataFrame(r["post"]).set_index(["n_a", "method"])
    assert post.loc[(0, "pseudo"), "entropy"] == pytest.approx(1.0, abs=1e-5)  # n_A = 0: the uniform prior
    # with one A point, the chain rule's context is B alone: exact = pseudo
    torch.testing.assert_close(r["lw"][("exact", 1)], r["lw"][("pseudo", 1)], atol=1e-4, rtol=0)


def test_ragged_padding_is_invisible(pfn) -> None:
    g = torch.Generator().manual_seed(0)
    x, y, q = torch.rand(7, 2, generator=g), torch.rand(7, generator=g), torch.rand(3, 2, generator=g)
    yq = torch.rand(3, generator=g)
    both = L._ragged(pfn, [(x[:3], y[:3]), (x, y)], [(q, yq), (q, yq)], chunk=8)
    alone = L._ragged(pfn, [(x[:3], y[:3])], [(q, yq)], chunk=8)
    torch.testing.assert_close(both[0], alone[0], atol=1e-5, rtol=0)


def test_gd_moves_toward_higher_likelihood(pfn) -> None:
    p = L.draw_pair(1, "shift", 0, 1, 12, 5, 4, 0.2)
    _, ll, traj = L.gd_shift(pfn, p, [5], 0.2, steps=30, lr=0.01, dev="cpu", every=10)
    _, ll0, _ = L.gd_shift(pfn, p, [5], 0.2, steps=0, lr=0.01, dev="cpu")
    assert float(ll[0]) >= float(ll0[0]) - 1e-4 and traj is not None and traj.shape == (3, 1, 1, 1)


def test_post_stats_are_finite_for_sharp_posteriors() -> None:
    grid = L.shift_grid(1, 0.2, 201, 5)
    lw = torch.full((201,), -1e4)
    lw[37] = 0.0
    st = L.post_stats(lw, grid, grid[37])
    assert st["entropy"] == pytest.approx(0.0, abs=1e-9) and st["n_modes"] == 1 and st["err_map"] == 0.0
