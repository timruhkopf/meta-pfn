import torch

from meta_pfn.experimental.regpfn import train as T
from meta_pfn.experimental.regpfn.model import RegBack
from meta_pfn.experimental.regpfn.prior import CELLS, PairPrior, PriorCfg, _grid


def test_prior_b_in_a_is_exact_without_noise() -> None:
    for d in (1, 2):
        for cell in CELLS:
            pr = PairPrior(PriorCfg(d=d, noise=(1e-12, 1e-12)), 8, 0)
            b = pr.sample(cell, 3, 5, 64)
            fg = pr._f(_grid(d, (512, 64)[d - 1], "cpu")[None].expand(8, -1, -1))
            lo, hi = fg.min(1, keepdim=True).values, fg.max(1, keepdim=True).values
            ref = 0.1 + 0.8 * (pr._f(b.xb_inA) - lo) / (hi - lo)
            assert float((b.yb_inA - ref).abs()[b.inside].max()) < 1e-5
            if cell in ("C00", "C01"):
                assert b.inside.all()


def _batch(n_a=4):
    pr = PairPrior(PriorCfg(d=1), 4, 0)
    return pr.sample("C11", n_a, 8, 32)


def test_no_a_points_leave_the_field_unchanged() -> None:
    torch.manual_seed(0)
    m = RegBack(1, dm=32, layers=1, n_iter=3)
    for p in m.delta.parameters():  # make the update non-trivial
        torch.nn.init.normal_(p, std=0.5)
    b = _batch(0)
    xs, ys = T.perturb(
        b,
        torch.full((4,), 0.1),
        torch.full((4,), 0.1),
        torch.Generator().manual_seed(0),
        torch.zeros(4, dtype=torch.bool),
    )
    _, _, _, hist = T._back_step(m, b, 0, xs, ys, 3)
    assert torch.equal(hist[-1][0], hist[0][0]) and torch.equal(hist[-1][1], hist[0][1])


def test_lupi_loss_decreases_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    m = RegBack(1, dm=32, layers=1, n_iter=3)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    b = _batch(8)
    g = torch.Generator().manual_seed(0)
    xs, ys = T.perturb(b, torch.full((4,), 0.05), torch.full((4,), 0.05), g, torch.zeros(4, dtype=torch.bool))
    losses = []
    for _ in range(60):
        _, lp, _, _ = T._back_step(m, b, 8, xs, ys, 3)
        opt.zero_grad()
        lp.mean().backward()
        opt.step()
        losses.append(float(lp.mean().detach()))
    assert losses[-1] < 0.5 * losses[0]


def test_train_and_eval_smoke() -> None:
    pfn, _ = T.train_aonly(1, 0.05, "cpu", batch=4, n_b=8, log=lambda *_: None)
    assert len(T.eval_aonly(pfn, 1, "cpu", n_pairs=4, n_perm=1)) > 0
    model, _ = T.train_back(1, 0.05, "cpu", "perturb", 2, batch=4, n_b=16, log=lambda *_: None)
    e = T.eval_back(model, 1, "cpu", "perturb", n_pairs=4)
    assert set(e.n_iter) == {0, 2} and e.nll.notna().all()
