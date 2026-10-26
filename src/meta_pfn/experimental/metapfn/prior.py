"""The phase-1 pair prior (experimental/regpfn/prior.py) as a training mixture, plus an unrelated stratum U.

- Cells C00 / C10 / C01 / C11: B = h(f_A(T(u))) with T and/or h on (regpfn.prior.CELLS).
- U: A as in C11, B a C11 draw of an *independent* BNN function; no B point has a correspondence (`inside` False).
Training mixes the strata with `TRAIN_P` (one stratum per batch). Evaluation uses regpfn.train.eval_set for the four
cells, so numbers are comparable with notebook 04; U's A is exactly C11's evaluation A (only B differs).
"""

import numpy as np
import torch

from ..regpfn.prior import CELLS, Batch, PairPrior, PriorCfg
from ..regpfn.train import N_A_EVAL, eval_set

STRATA = (*CELLS, "U")
TRAIN_P = {"C00": 0.1, "C10": 0.7 / 3, "C01": 0.7 / 3, "C11": 0.7 / 3, "U": 0.2}


def unrelated(a: Batch, b: Batch) -> Batch:
    """A from `a`, B from `b` (an independent function); B has no correspondence in A."""
    return Batch(a.xa, a.ya, b.ub, b.yb, b.xb_inA, b.yb_inA, torch.zeros_like(b.inside), a.fa_test, a.noise, a.n_a)


class MixPrior:
    def __init__(self, cfg: PriorCfg, batch: int, seed: int, device: str):
        self.priors = {c: PairPrior(cfg, batch, seed + 31 * i, device) for i, c in enumerate(CELLS)}
        self.other = PairPrior(cfg, batch, seed + 997, device)
        self.g = np.random.default_rng(seed + 5)

    def draw_stratum(self) -> str:
        return str(self.g.choice(STRATA, p=[TRAIN_P[s] for s in STRATA]))

    def sample(self, stratum: str, n_a: int, n_test: int, n_b: int) -> Batch:
        if stratum == "U":
            return unrelated(self.priors["C11"].sample("C11", n_a, n_test, n_b), self.other.sample("C11", 0, 1, n_b))
        return self.priors[stratum].sample(stratum, n_a, n_test, n_b)


def eval_batch(d: int, stratum: str, n_pairs: int, n_b: int, seed: int, device: str) -> Batch:
    """Fixed evaluation pairs (nested in n_A: the first max(N_A_EVAL) A points are the context pool)."""
    if stratum != "U":
        return eval_set(d, stratum, n_pairs, 128, n_b, seed, device)
    a = eval_set(d, "C11", n_pairs, 128, n_b, seed, device)
    b = PairPrior(PriorCfg(d=d), n_pairs, seed + 4242 + 7 * d, device).sample("C11", max(N_A_EVAL), 1, n_b)
    return unrelated(a, b)


def shuffle_b(b: Batch, g: torch.Generator) -> Batch:
    """Shuffled-B control: B's values at permuted locations (per pair); correspondences destroyed."""
    perm = torch.rand(b.yb.shape, generator=g, device="cpu").argsort(1).to(b.yb.device)
    yb = torch.gather(b.yb, 1, perm)
    return Batch(b.xa, b.ya, b.ub, yb, b.xb_inA, b.yb_inA, torch.zeros_like(b.inside), b.fa_test, b.noise, b.n_a)
