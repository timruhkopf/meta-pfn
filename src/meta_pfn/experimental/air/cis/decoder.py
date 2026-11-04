"""The transported decoder D for AIR-CIS (spec §2): AIR's Decoder (experimental/air/decoder.py, unchanged) plus the
moved-B feature log (T⁻¹)′(x_b), Fourier-featured and summed into each B token's embedding; n_A ∈ {0..10} in
training; gradient clip 20; plateau rule (pilot 3's `Plateau`: < 0.01 improvement over 5k steps, then a 5k
cosine anneal). `tfeat=False` is the C8 ablation (the plain AIR decoder on this prior).

Batches carry B already moved into A's frame by the true transform (teacher forcing on privileged θ), and every
step also trains the null head (B masked out), as AIR."""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..decoder import Decoder, build_mask, fourier, soft_targets
from ..train_rb import Plateau, _set_lr
from . import prior as C

TFEAT_SCALE = 0.25  # log-derivatives are O(1); keep the Fourier features' lowest frequency below one period


class CisDecoder(Decoder):
    def __init__(self, D: int = 128, layers: int = 6, heads: int = 4, tfeat: bool = True):
        super().__init__(D, layers, heads)
        self.tfeat = tfeat
        if tfeat:
            self.emb_t = nn.Linear(fourier(torch.zeros(1)).shape[-1], D)

    def forward(self, xb_t, yb, xa, ya, xq, rank=None, use_b: bool = True, ld=None):
        """xb_t [P, N_B] B in A's frame, ld [P, N_B] log (T⁻¹)′(x_b) (ignored if tfeat=False); A [P, n]; x* [P, Q].
        -> (logits at x* [P, Q, G], evidence logits at x_a [P, n, G])."""
        P, n_b, n, Q = xb_t.shape[0], xb_t.shape[1], xa.shape[1], xq.shape[1]
        if rank is None:
            rank = torch.rand(P, n, device=xa.device).argsort(-1).argsort(-1)
        tb = self.emb_x(fourier(xb_t)) + self.emb_y(fourier(yb)) + self.src[0]
        if self.tfeat:
            tb = tb + self.emb_t(fourier(ld * TFEAT_SCALE))
        tok = torch.cat([
            self.sink.expand(P, -1, -1),
            tb,
            self.emb_x(fourier(xa)) + self.emb_y(fourier(ya)) + self.src[1],
            self.emb_x(fourier(xq)) + self.src[2],
            self.emb_x(fourier(xa)) + self.src[3],
        ], 1)  # fmt: skip
        mask = build_mask(n_b, n, Q, rank, use_b)
        for blk in self.blocks:
            tok = blk(tok, mask)
        out = self.head(self.ln(tok))
        s = 1 + n_b + n
        return out[:, s : s + Q], out[:, s + Q :]


def loss(dec: CisDecoder, t: C.CisTasks, xb_t, ld, use_b: bool = True) -> torch.Tensor:
    """Soft-target CE (as AIR's decoder_loss) on x* and on the evidence queries, in NLL density units."""
    from ..decoder import N_BINS, Y_HI, Y_LO

    lq, le = dec(xb_t, t.yb, t.xa, t.ya, t.xq, use_b=use_b, ld=ld)
    out = -(soft_targets(t.fq, t.sig_a) * lq.log_softmax(-1)).sum(-1).mean()
    if t.xa.shape[1]:
        out = out - (soft_targets(t.fa, t.sig_a) * le.log_softmax(-1)).sum(-1).mean()
    return out + float(np.log((Y_HI - Y_LO) / N_BINS)) * (1 + (t.xa.shape[1] > 0))


@torch.no_grad()
def val_nll(dec: CisDecoder, t: C.CisTasks) -> dict:
    from ..decoder import bar_logpdf

    dec.eval()
    xb_t, ld = t.b_in_a()
    lq, _ = dec(xb_t, t.yb, t.xa, t.ya, t.xq, ld=ld)
    lq0, _ = dec(xb_t, t.yb, t.xa, t.ya, t.xq, use_b=False, ld=ld)
    return {"val_teacher_n5": float(-bar_logpdf(lq, t.yq).mean()), "val_null_n5": float(-bar_logpdf(lq0, t.yq).mean())}


def train(dec: CisDecoder, max_steps: int, device, *, seed: int, batch: int, lr: float, clip: float = 20.0,
          every: int = 250, min_steps: int = 10000, metrics=None, log=print) -> pd.DataFrame:  # fmt: skip
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = C.CisCfg()
    val = C.sample_tasks(cfg, 256, 5, torch.Generator().manual_seed(seed + 7), device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    pl = Plateau(lr, max_steps, min_steps=min_steps)
    hist, t0, step = [], time.time(), 0
    while True:
        _set_lr(opt, pl.rate(step))
        n = int(rng.integers(0, C.N_A_MAX + 1))
        t = C.sample_tasks(cfg, batch, n, g)
        xb_t, ld = t.b_in_a()
        dec.train()
        lt = loss(dec, t, xb_t, ld)
        l0 = loss(dec, t, xb_t, ld, use_b=False)
        tot = lt + l0
        opt.zero_grad(set_to_none=True)
        tot.backward()
        gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), clip)
        opt.step()
        if step % every == 0:
            v = val_nll(dec, val)
            pl.observe(step, v["val_teacher_n5"])
            row = {"step": step, "loss_teacher": float(lt), "loss_null": float(l0), "grad_norm": float(gn), "n_a": n,
                   "lr": pl.rate(step), "s": time.time() - t0, **v}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({f"train/{k}": float(row[k]) for k in ("loss_teacher", "loss_null", "grad_norm", "lr")}
                        | {f"val/{k}": float(x) for k, x in v.items()}, step=step)  # fmt: skip
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
            if not np.isfinite(row["loss_teacher"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
        if pl.done(step):
            break
        step += 1
    return pd.DataFrame(hist)
