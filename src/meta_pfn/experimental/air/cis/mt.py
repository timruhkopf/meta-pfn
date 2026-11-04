"""MTPFN (#20's architecture) and the clean A-only PFN on the AIR-CIS prior: batch adapters and the training draw."""

import torch

from . import prior as C


def mt_batch(t: C.CisTasks):
    """MTPFN input: task 0 = A (target), task 1 = B; x [P, 2, N, 1]."""
    P, n, n_b = t.xa.shape[0], t.xa.shape[1], t.xb.shape[1]
    N = max(n, n_b)
    x = t.xa.new_zeros(P, 2, N, 1)
    y = t.xa.new_zeros(P, 2, N)
    mask = torch.zeros(P, 2, N, dtype=torch.bool, device=t.xa.device)
    x[:, 0, :n, 0], y[:, 0, :n], mask[:, 0, :n] = t.xa, t.ya, True
    x[:, 1, :n_b, 0], y[:, 1, :n_b], mask[:, 1, :n_b] = t.xb, t.yb, True
    return x, y, mask, t.xq[..., None], t.yq, torch.ones(P, device=t.xa.device, dtype=torch.long)


def ao_batch(t: C.CisTasks):
    """A-only input: the same architecture with A as the only task (B never enters)."""
    P, n = t.xa.shape
    mask = torch.ones(P, 1, n, dtype=torch.bool, device=t.xa.device)
    return (t.xa[:, None, :, None], t.ya[:, None], mask, t.xq[..., None], t.yq,
            torch.ones(P, device=t.xa.device, dtype=torch.long))  # fmt: skip


def sampler(aonly: bool):
    """Training draws: n_A ~ U{1..10} (as notebook 16), fresh prior tasks."""
    cfg = C.CisCfg()

    def draw(rng, g: torch.Generator, P: int):
        t = C.sample_tasks(cfg, P, int(rng.integers(1, C.N_A_MAX + 1)), g)
        return ao_batch(t) if aonly else mt_batch(t)

    return draw
