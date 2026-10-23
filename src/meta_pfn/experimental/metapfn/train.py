"""Training and evaluation of the end-to-end models (issue #12, notebook 05).

Training (fixed step budget, cosine LR, one stratum per batch from `prior.TRAIN_P`, n_A from the phase-1 curriculum):
- `m0`: TwoCtxPFN, sample NLL on A's queries; B dropped with prob. `p_drop_b` (A-only mode). LUPI via `priv`:
  `none`; `dropout`: B_inA (inside points) added as role-2 tokens with prob. `p_priv`; `rb`: every batch with B
  is run twice, with and without privileged tokens; loss = NLL(priv) + λ NLL(plain) + (1 - λ) CE(sg p_priv, p_plain)
  over the bar bins (the Rao-Blackwellized target, brainstorm §5.2).
- `m3`: RegE2E, mixture NLL + `lupi` · Σ_t γ^(T-t) field error vs B_inA (inside points), as in phase 1.
Evaluation: fixed pairs per stratum (prior.eval_batch), n_A ∈ N_A_EVAL, per-pair NLL for the variants
`model`, `model_aonly` (B dropped / null slot), `model_shuffled` (shuffled-B control), `pfn_aonly` and
`pfn_pool` (the frozen A-only PFN with B pooled at face value, (u, y_B), as if it were A).
"""

import math
import time

import numpy as np
import pandas as pd
import torch
from pfn_repro.surrogates.pfn import PFN

from ..regpfn.prior import Batch, PriorCfg
from ..regpfn.train import N_A_EVAL, _pad_ctx, field_err, field_mse, n_a_curriculum
from .models import RegE2E, TwoCtxPFN
from .prior import STRATA, MixPrior, eval_batch, shuffle_b


def load_pfn(path: str | None, d: int, device: str) -> PFN:
    pfn = PFN(max_x_dim=d, d_model=256, n_heads=4, n_layers=6, d_ff=512, n_bins=64)
    if path:
        pfn.load_state_dict(torch.load(path, map_location="cpu"))
    return pfn.to(device).eval()


def m0_context(b: Batch, n: int, use_b: bool, priv: bool):
    """-> (xc, yc, role, mask): A's first n points, B's points (role 1), B_inA inside points (role 2)."""
    P, d, dev = b.xa.shape[0], b.xa.shape[2], b.xa.device
    parts = [(b.xa[:, :n], b.ya[:, :n], 0, torch.ones(P, n, dtype=torch.bool, device=dev))]
    if use_b:
        parts.append((b.ub, b.yb, 1, torch.ones_like(b.yb, dtype=torch.bool)))
    if priv:
        parts.append((b.xb_inA.clamp(0, 1), b.yb_inA.clamp(0, 1), 2, b.inside))
    xc, yc = torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
    role = torch.cat([torch.full_like(p[1], p[2], dtype=torch.long) for p in parts], 1)
    m = torch.cat([p[3] for p in parts], 1)
    if xc.shape[1] == 0:
        xc, yc, m = _pad_ctx(xc.new_zeros(P, 0, d), yc.new_zeros(P, 0))
        role = torch.zeros(P, 1, dtype=torch.long, device=dev)
    return xc, yc, role, m


def _lr(opt, step: int, steps: int, lr: float, warm: int = 500):
    for pg in opt.param_groups:
        pg["lr"] = lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))


def train(
    mode: str,
    d: int,
    steps: int,
    device: str,
    seed: int = 0,
    batch: int = 32,
    n_b: int = 256,
    n_test: int = 64,
    priv: str = "none",
    p_priv: float = 0.3,
    p_drop_b: float = 0.15,
    lam: float = 0.5,
    lupi: float = 1.0,
    n_iter: int = 8,
    gn: bool = True,
    pfn_path: str | None = None,
    lr: float = 3e-4,
    log=print,
):
    if mode == "m0":
        model = TwoCtxPFN(d).to(device)
    elif mode == "m3":
        model = RegE2E(d, load_pfn(pfn_path, d, device), n_iter=n_iter, gn=gn).to(device)
    else:
        raise ValueError(f"mode must be m0 or m3, got {mode}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    pr = MixPrior(PriorCfg(d=d), batch, seed, device)
    g = np.random.default_rng(seed)
    tg = torch.Generator(device="cpu").manual_seed(seed)
    t0, hist = time.time(), []
    model.train()
    for step in range(steps):
        _lr(opt, step, steps, lr)
        stratum = pr.draw_stratum()
        n = n_a_curriculum(g)
        b = pr.sample(stratum, n, n_test, n_b)
        xq, yq = b.xa[:, n:], b.ya[:, n:]
        extra = 0.0
        if mode == "m0":
            use_b = g.random() >= p_drop_b
            if priv == "rb" and use_b:
                lg_p = model(*m0_context(b, n, True, True), xq)
                lg = model(*m0_context(b, n, True, False), xq)
                ce = -(torch.softmax(lg_p.detach(), -1) * torch.log_softmax(lg, -1)).sum(-1).mean()
                nll = model.nll(lg, yq).mean()
                extra = float(ce.detach())
                loss = model.nll(lg_p, yq).mean() + lam * nll + (1 - lam) * ce
            else:
                use_p = priv == "dropout" and use_b and g.random() < p_priv
                nll = model.nll(model(*m0_context(b, n, use_b, use_p), xq), yq).mean()
                loss = nll
        else:
            lp, log_w, fh = model(b.xa, b.ya, n, b.ub, b.yb, xq, yq, k_max=8, g=tg)
            nll = model.mixture_nll(lp, log_w).mean()
            T = len(fh) - 1
            fl = sum(0.8 ** (T - t) * field_mse(x, y, b) for t, (x, y) in enumerate(fh) if t > 0)
            extra = float(fl.mean()) if torch.is_tensor(fl) else 0.0
            loss = nll + (lupi * fl.mean() if torch.is_tensor(fl) and lupi > 0 else 0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            hist.append((step + 1, time.time() - t0, stratum, n, float(nll), extra))
            if (step + 1) % 500 == 0:
                log(f"{mode} step {step + 1} nll {float(nll):.3f} extra {extra:.4f} {time.time() - t0:.0f}s")
    return model, pd.DataFrame(hist, columns=["step", "seconds", "stratum", "n_a", "nll", "extra"])


@torch.no_grad()
def evaluate(model, mode: str, d: int, device: str, pfn: PFN, n_pairs=128, n_b=256, seed=12345):
    """-> (per-pair NLL rows (stratum, n_a, variant, pair, noise, nll), m3 slot weights and field errors per pair)."""
    model.eval()
    rows, extra_rows = [], []
    tg = torch.Generator(device="cpu").manual_seed(seed + 3)
    n0 = max(N_A_EVAL)
    for stratum in STRATA:
        b = eval_batch(d, stratum, n_pairs, n_b, seed, device)
        bs = shuffle_b(b, tg)
        xq, yq = b.xa[:, n0:], b.ya[:, n0:]
        noise = b.noise.cpu().numpy()
        for n in N_A_EVAL:
            out = {}
            xa = torch.cat([b.xa[:, :n], xq], 1)
            ya = torch.cat([b.ya[:, :n], yq], 1)
            if mode == "m0":
                out["model"] = model.nll(model(*m0_context(b, n, True, False), xq), yq).mean(1)
                out["model_aonly"] = model.nll(model(*m0_context(b, n, False, False), xq), yq).mean(1)
                out["model_shuffled"] = model.nll(model(*m0_context(bs, n, True, False), xq), yq).mean(1)
                out["model_priv"] = model.nll(model(*m0_context(b, n, True, True), xq), yq).mean(1)
            else:
                lp, log_w, fh = model(xa, ya, n, b.ub, b.yb, xq, yq)
                out["model"] = model.mixture_nll(lp, log_w).mean(1)
                out["model_aonly"] = -lp[..., 2].mean(1)
                out["model_refined_only"] = -lp[..., 0].mean(1)
                out["model_raw_only"] = -lp[..., 1].mean(1)
                lps, log_ws, _ = model(xa, ya, n, bs.ub, bs.yb, xq, yq)
                out["model_shuffled"] = model.mixture_nll(lps, log_ws).mean(1)
                w = log_w.exp().cpu().numpy()
                e0, e1 = field_err(*fh[0], b).cpu().numpy(), field_err(*fh[-1], b).cpu().numpy()
                for p in range(n_pairs):
                    extra_rows.append((stratum, n, p, *w[p], e0[p], e1[p], float(b.inside[p].float().mean())))
            xa0, ya0, ma0 = _pad_ctx(b.xa[:, :n], b.ya[:, :n])
            out["pfn_aonly"] = pfn.bar_dist(pfn(xa0, ya0, xq, train_key_padding_mask=ma0), yq).mean(1)
            xc, yc = torch.cat([b.xa[:, :n], b.ub], 1), torch.cat([b.ya[:, :n], b.yb], 1)
            out["pfn_pool"] = pfn.bar_dist(pfn(xc, yc, xq), yq).mean(1)
            for v, t in out.items():
                for p, val in enumerate(t.cpu().numpy()):
                    rows.append((stratum, n, v, p, noise[p], float(val)))
    ev = pd.DataFrame(rows, columns=["stratum", "n_a", "variant", "pair", "noise", "nll"])
    cols = ["stratum", "n_a", "pair", "w_refined", "w_raw", "w_null", "err_seed", "err_final", "inside"]
    return ev, pd.DataFrame(extra_rows, columns=cols)
