"""Phase-1 training and evaluation (docs/spec_phase1_experiments.md §5–§7).

- `train_aonly`: the A-only PFN (pfn-repro's PFN, trained here on A's data from the pair prior) with context
  lengths up to n_A_max + N_B, so that the effective-data curve (A ∪ m points of B_inA) stays in distribution.
- `eval_aonly`: E1 (NLL vs n_A) and E2 (NLL(m) for m points of B_inA added, several random subsets).
- `train_back`: the back half (`RegBack`) with seed = true B_inA (E3) or true + smooth perturbation (E4), loss =
  mixture NLL + LUPI field loss Σ_t γ^(T-t) |(x̂_t, ŷ_t) - B_inA|² over inside-window B points.
- `eval_back`: NLL per cell × n_A (× perturbation), field error before / after refinement, null-slot weight.
The evaluation pairs are fixed per (d, cell) (`eval_set`) and nested in n_A, so every run scores the same pairs.
"""

import math
import time

import numpy as np
import pandas as pd
import torch
from pfn_repro.surrogates.pfn import PFN

from .model import RegBack
from .prior import CELLS, Batch, PairPrior, PriorCfg

N_A_EVAL = (0, 2, 5, 10, 20)
M_LADDER = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def n_a_curriculum(g: np.random.Generator, n_max: int = 20) -> int:
    """n_A ∈ [0, n_max], weighted toward small values (p ∝ 1/sqrt(n + 1))."""
    p = 1 / np.sqrt(np.arange(n_max + 1) + 1)
    return int(g.choice(n_max + 1, p=p / p.sum()))


def _pad_ctx(x: torch.Tensor, y: torch.Tensor):
    """Context with at least one token; mask False on the padding (n = 0 -> one masked dummy)."""
    if x.shape[1] == 0:
        return (
            x.new_zeros(x.shape[0], 1, x.shape[2]),
            y.new_zeros(y.shape[0], 1),
            torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device),
        )
    return x, y, torch.ones(y.shape, dtype=torch.bool, device=y.device)


def pfn_nll(pfn: PFN, xc, yc, mask, xq, yq) -> torch.Tensor:
    """-> [B, Q] NLL of the plain PFN."""
    lg = pfn(xc, yc, xq, train_key_padding_mask=mask)
    return pfn.bar_dist(lg, yq)


def eval_set(d: int, cell: str, n_pairs: int, n_test: int, n_b: int, seed: int, device: str) -> Batch:
    pr = PairPrior(PriorCfg(d=d), n_pairs, seed + 1009 * list(CELLS).index(cell) + 7 * d, device)
    return pr.sample(cell, max(N_A_EVAL), n_test, n_b)


def train_aonly(d: int, minutes: float, device: str, seed: int = 0, batch: int = 64, n_b: int = 256, log=print):
    pfn = PFN(max_x_dim=d, d_model=256, n_heads=4, n_layers=6, d_ff=512, n_bins=64).to(device)
    opt = torch.optim.AdamW(pfn.parameters(), lr=3e-4, weight_decay=0.0)
    pr = PairPrior(PriorCfg(d=d), batch, seed, device)
    g = np.random.default_rng(seed)
    t0, step, hist = time.time(), 0, []
    total = minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for pg in opt.param_groups:
            pg["lr"] = 3e-4 * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * frac))
        n = n_a_curriculum(g) if g.random() < 0.5 else int(g.integers(0, 20 + n_b + 1))
        b = pr.sample("C00", n, 128, 1)
        xc, yc, m = _pad_ctx(b.xa[:, :n], b.ya[:, :n])
        loss = pfn_nll(pfn, xc, yc, m, b.xa[:, n:], b.ya[:, n:]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pfn.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            hist.append((step, time.time() - t0, float(loss)))
            log(f"aonly step {step} loss {float(loss):.3f}")
    return pfn, pd.DataFrame(hist, columns=["step", "seconds", "loss"])


@torch.no_grad()
def eval_aonly(pfn: PFN, d: int, device: str, n_pairs=128, n_perm=5, seed=12345) -> pd.DataFrame:
    """E1 + E2: rows (cell, n_a, m, perm, nll) with nll the mean over pairs and queries."""
    pfn.eval()
    rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for cell in CELLS:
        b = eval_set(d, cell, n_pairs, 128, 256, seed, device)
        n0 = max(N_A_EVAL)
        xq, yq = b.xa[:, n0:], b.ya[:, n0:]
        for n in N_A_EVAL:
            for m in M_LADDER:
                for p in range(n_perm if m > 0 else 1):
                    # m random inside-window B_inA points per pair (fewer if fewer are inside)
                    key = torch.rand(b.inside.shape, generator=g).to(device) + (~b.inside).float() * 10
                    idx = key.argsort(1)[:, :m]
                    ok = torch.gather(b.inside, 1, idx)
                    xb = torch.gather(b.xb_inA, 1, idx[..., None].expand(-1, -1, d))
                    yb = torch.gather(b.yb_inA, 1, idx)
                    xc = torch.cat([b.xa[:, :n], xb], 1)
                    yc = torch.cat([b.ya[:, :n], yb], 1).clamp(0, 1)
                    mask = torch.cat([torch.ones(n_pairs, n, dtype=torch.bool, device=device), ok], 1)
                    if xc.shape[1] == 0:
                        xc, yc, mask = _pad_ctx(xc, yc)
                    nll = pfn_nll(pfn, xc, yc, mask, xq, yq).mean(1)
                    rows.append((cell, n, m, p, float(nll.mean()), float(ok.float().sum(1).mean())))
    return pd.DataFrame(rows, columns=["cell", "n_a", "m", "perm", "nll", "m_inside"])


def perturb(b: Batch, sig_x: torch.Tensor, sig_y: torch.Tensor, g: torch.Generator, shift_only: torch.Tensor):
    """Seed = true field + smooth perturbation: x̂ = T(u) + δ(u) (δ a random sum of sines of u, or a global shift
    where `shift_only`), ŷ = y_inA + σ_y s(y_inA) (s a random sum of sines of y). RMS-scaled per pair."""
    P, N, d = b.ub.shape
    dev = b.ub.device
    k = torch.arange(1, 4, device=dev).float()
    ph = torch.rand(P, 1, d, 3, generator=g, device=dev) * 2 * math.pi
    amp = torch.randn(P, 1, d, 3, generator=g, device=dev) / k
    dx = (amp * torch.sin(2 * math.pi * k * b.ub[..., None] + ph)).sum(-1)  # [P, N, d]
    sh = torch.randn(P, 1, d, generator=g, device=dev).expand(-1, N, -1)
    dx = torch.where(shift_only[:, None, None], sh, dx)
    dx = dx / dx.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8) * sig_x[:, None, None]
    phy = torch.rand(P, 1, 3, generator=g, device=dev) * 2 * math.pi
    ampy = torch.randn(P, 1, 3, generator=g, device=dev) / k
    dy = (ampy * torch.sin(2 * math.pi * k * b.yb_inA[..., None] + phy)).sum(-1)
    dy = dy / dy.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8) * sig_y[:, None]
    return b.xb_inA + dx, b.yb_inA + dy


def field_mse(x_hat, y_hat, b: Batch) -> torch.Tensor:
    """-> [P] mean over inside-window B points of |(x̂, ŷ) - B_inA|² (the LUPI loss; no sqrt: finite gradient at 0)."""
    e = ((x_hat - b.xb_inA) ** 2).sum(-1) + (y_hat - b.yb_inA) ** 2
    return (e * b.inside).sum(1) / b.inside.sum(1).clamp_min(1)


def field_err(x_hat, y_hat, b: Batch) -> torch.Tensor:
    """-> [P] RMS field error (reporting)."""
    return field_mse(x_hat, y_hat, b).sqrt()


def _back_step(model: RegBack, b: Batch, n: int, x_seed, y_seed, n_iter, gamma=0.8):
    xc, yc, am = _pad_ctx(b.xa[:, :n], b.ya[:, :n])
    bm = torch.ones(b.yb.shape, dtype=torch.bool, device=b.yb.device)
    ls, ln, lw, hist = model(xc, yc, am, b.ub, b.yb, x_seed, y_seed, bm, b.xa[:, n:], n_iter)
    nll = model.mixture_nll(ls, ln, lw, b.ya[:, n:]).mean(1)
    T = len(hist) - 1
    lupi = sum(gamma ** (T - t) * field_mse(x, y, b) for t, (x, y) in enumerate(hist) if t > 0) if T else 0.0
    return nll, lupi, lw, hist


def train_back(
    d: int,
    minutes: float,
    device: str,
    seed_mode: str,
    n_iter: int,
    sig_max: float = 0.25,
    lupi: float = 1.0,
    seed: int = 0,
    batch: int = 32,
    n_b: int = 256,
    log=print,
    resid: bool = False,
    max_step: float | None = None,
):
    model = RegBack(d, n_iter=n_iter, resid=resid, max_step=max_step).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    priors = {c: PairPrior(PriorCfg(d=d), batch, seed + 31 * i, device) for i, c in enumerate(CELLS)}
    g = np.random.default_rng(seed)
    tg = torch.Generator(device=device).manual_seed(seed)
    t0, step, hist = time.time(), 0, []
    total = minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for pg in opt.param_groups:
            pg["lr"] = 3e-4 * min(1.0, step / 300) * 0.5 * (1 + math.cos(math.pi * frac))
        cell = list(CELLS)[step % 4]
        n = n_a_curriculum(g)
        b = priors[cell].sample(cell, n, 64, n_b)
        if seed_mode == "true":
            xs, ys = b.xb_inA, b.yb_inA
        else:
            on = torch.rand(batch, 2, generator=tg, device=device) < 0.5
            sx = torch.rand(batch, generator=tg, device=device) * sig_max * on[:, 0]
            sy = torch.rand(batch, generator=tg, device=device) * sig_max * on[:, 1]
            sh = torch.rand(batch, generator=tg, device=device) < 0.3
            xs, ys = perturb(b, sx, sy, tg, sh)
        nll, lp, _, _ = _back_step(model, b, n, xs, ys, n_iter)
        loss = nll.mean() + lupi * (lp.mean() if torch.is_tensor(lp) else 0.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            hist.append((step, time.time() - t0, float(nll.mean()), float(lp.mean()) if torch.is_tensor(lp) else 0.0))
            log(f"back step {step} nll {hist[-1][2]:.3f} lupi {hist[-1][3]:.4f}")
    return model, pd.DataFrame(hist, columns=["step", "seconds", "nll", "lupi"])


SIG_LADDER = (0.0, 0.02, 0.05, 0.1, 0.2)


@torch.no_grad()
def eval_back(model: RegBack, d: int, device: str, seed_mode: str, n_pairs=128, seed=12345) -> pd.DataFrame:
    """Rows (cell, n_a, kind, sigma, n_iter, nll, w_null, err_seed, err_final, err_change)."""
    model.eval()
    rows = []
    conds = (
        [("none", 0.0)]
        if seed_mode == "true"
        else [("none", 0.0)] + [(k, s) for k in ("x", "shift", "y") for s in SIG_LADDER[1:]]
    )
    tg = torch.Generator(device=device).manual_seed(seed + 1)
    for cell in CELLS:
        b = eval_set(d, cell, n_pairs, 128, 256, seed, device)
        n0 = max(N_A_EVAL)
        for kind, s in conds:
            z = torch.zeros(n_pairs, device=device)
            full = torch.full((n_pairs,), s, device=device)
            xs, ys = perturb(
                b,
                full if kind in ("x", "shift") else z,
                full if kind == "y" else z,
                tg,
                torch.full((n_pairs,), kind == "shift", device=device),
            )
            for n in N_A_EVAL:
                bb = Batch(
                    torch.cat([b.xa[:, :n], b.xa[:, n0:]], 1),
                    torch.cat([b.ya[:, :n], b.ya[:, n0:]], 1),
                    b.ub,
                    b.yb,
                    b.xb_inA,
                    b.yb_inA,
                    b.inside,
                    b.fa_test,
                    b.noise,
                    n,
                )
                for it in sorted({0, model.n_iter}):
                    nll, _, lw, hist = _back_step(model, bb, n, xs, ys, it)
                    e0, e1 = field_err(*hist[0], bb), field_err(*hist[-1], bb)
                    chg = ((hist[-1][0] - hist[0][0]) ** 2).sum(-1).add((hist[-1][1] - hist[0][1]) ** 2).mean(1).sqrt()
                    rows.append(
                        (
                            cell,
                            n,
                            kind,
                            s,
                            it,
                            float(nll.mean()),
                            float(lw[:, 1].exp().mean()),
                            float(e0.mean()),
                            float(e1.mean()),
                            float(chg.mean()),
                        )
                    )
    cols = ["cell", "n_a", "kind", "sigma", "n_iter", "nll", "w_null", "err_seed", "err_final", "field_change"]
    return pd.DataFrame(rows, columns=cols)
