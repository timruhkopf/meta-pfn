"""Field-only (issue #14, notebook 07): can the second stage alone register B into A's frame from a random start?

For one pair (A's first n points, B's N_B points; phase-1 prior, d = 1, h = id), a *field* places every B point in A's
frame: (x̂_i, ŷ_i) for B point i at u_i; the truth is B_inA = (T(u_i), y_i on A's scale). No GW, no landmarks: the
field starts from a random affine guess x̂⁰(u) = 0.5 + a₀ (u + s₀ − 0.5) (s₀ ~ U[−0.3, 0.3], a₀ ~ logU[0.7, 1.4], the
prior's own shift/scale ranges, drawn independently of the truth), ŷ⁰ = y_B; or from the identity x̂⁰ = u (reference).

Data fit: B rendered at x is a soft nearest neighbour r_ℓ(x) = Σ_i softmax_i(−|x − x̂_i|² / 2ℓ²) ŷ_i, and
E_ℓ(x̂, ŷ) = (1/n) Σ_a (y_a − r_ℓ(x_a))², i.e. how well the placed B explains A's few points.

Methods (all move the same field; weight-tied where learned):
- `gd` (no learning): gradient descent (Adam) on E_ℓ with ℓ annealed 0.15 → 0.02 (coarse to fine), over a monotone
  piecewise-linear map of u (8 segments, softplus slopes) and an affine ŷ = α y_B + β, from the given start; plus
  `gd_best8` (8 starts: the given one + 7 random, lowest energy) and `gd_oracle8` (the start closest to the truth).
- `operator`: the phase-1 refinement layer (`regpfn.model.RegBack.refine`, explicit residuals, steps ≤ 0.1): each
  B point attends to A's points near x̂ (distance-biased), a GRU proposes a step, gated by the local evidence mass and
  smoothed over B neighbours in u.
- `optimizer`: a learned optimizer on the same field: the GRU sees the per-point gradients ∂E_ℓ/∂(x̂_i, ŷ_i) at three
  widths ℓ ∈ {0.02, 0.05, 0.15} (each normalized by its RMS over B), their B-means, the lookup of A near x̂, and
  proposes a local step (smoothed over B in u) plus a global step from the pooled state; steps ≤ 0.1, no gate.
The learned methods are trained with a LUPI loss on the field, Σ_t γ^(T−t) |(x̂_t, ŷ_t) − B_inA|² over inside-window
B points (isolated component test; no read-out, no NLL).
Metrics per pair: err_x = RMS_i |x̂_i − T(u_i)| over inside-window B points (A's frame), err_y likewise; the energy
at the fit and at the truth (ℓ = 0.02); pred_rmse = RMS over A's queries inside B's image of |r(x*) − f_A(x*)|.
"""

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .regpfn.model import CrossBlock, Fourier, RegBack, sqdist
from .regpfn.prior import Batch, PairPrior, PriorCfg
from .regpfn.train import eval_set, field_mse

N_A_EVAL = (2, 5, 10, 20)
CELLS = ("C00", "C10")  # h = id (decision 0003: isolation)
INITS = ("random", "identity")
LENS = (0.02, 0.05, 0.15)
ELL_FINE = 0.02


def init_field(b: Batch, g: torch.Generator, identity: bool = False):
    """-> (x̂⁰ [P, N, 1], ŷ⁰ [P, N]): a random affine guess from the prior's shift/scale ranges, or the identity."""
    if identity:
        return b.ub.clone(), b.yb.clone()
    P, dev = b.ub.shape[0], b.ub.device
    s = (2 * torch.rand(P, 1, 1, generator=g, device=dev) - 1) * 0.3
    a = torch.exp(torch.empty(P, 1, 1, device=dev).uniform_(math.log(0.7), math.log(1.4), generator=g))
    return 0.5 + a * (b.ub + s - 0.5), b.yb.clone()


def render(xq: torch.Tensor, x_hat: torch.Tensor, y_hat: torch.Tensor, ell: float) -> torch.Tensor:
    """[P, Q, 1], [P, N, 1], [P, N] -> [P, Q] soft nearest-neighbour value of the placed B at xq."""
    return (torch.softmax(-sqdist(xq, x_hat) / (2 * ell**2), -1) * y_hat[:, None, :]).sum(-1)


def energy(xa, ya, x_hat, y_hat, ell: float) -> torch.Tensor:
    """-> [P] mean squared misfit of A's points to the rendered B."""
    return ((ya - render(xa, x_hat, y_hat, ell)) ** 2).mean(1)


def grad_feats(xa, ya, x_hat, y_hat, lens=LENS) -> torch.Tensor:
    """-> [P, N, 3 · len(lens)]: per width, ∂E/∂x̂_i and ∂E/∂ŷ_i divided by their RMS over B, and log RMS (constants:
    computed on detached copies, so the learned optimizer sees them as inputs)."""
    out = []
    with torch.enable_grad():
        x = x_hat.detach().requires_grad_()
        y = y_hat.detach().requires_grad_()
        for ell in lens:
            gx, gy = torch.autograd.grad(energy(xa, ya, x, y, ell).sum(), (x, y))
            g = torch.cat([gx, gy[..., None]], -1)  # [P, N, 2]
            rms = g.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
            out += [g / rms, torch.log(rms[..., :1]).expand(-1, g.shape[1], -1)]
    return torch.cat(out, -1)


class FieldOptimizer(nn.Module):
    def __init__(self, d: int = 1, dm: int = 64, heads: int = 4, n_iter: int = 12, max_step: float = 0.1):
        super().__init__()
        self.dm, self.n_iter, self.max_step = dm, n_iter, max_step
        self.phi = Fourier(d)
        ng = 3 * len(LENS)
        self.a_in = nn.Linear(self.phi.out + 1, dm)
        self.b_in = nn.Linear(2 * self.phi.out + 2 + 2 * ng, dm)
        self.lookup = CrossBlock(dm, heads, init_len=0.05)
        self.gru = nn.GRUCell(dm, dm)
        self.local = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, d + 1))
        self.glob = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, d + 1))
        for head in (self.local, self.glob):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.log_len_h = nn.Parameter(torch.log(torch.tensor(0.05)))

    def forward(self, xa, ya, ub, yb, x_hat, y_hat, n_iter=None):
        """-> list of (x̂, ŷ) after every iteration (index 0 = start)."""
        P, N, _ = ub.shape
        a_tok = self.a_in(torch.cat([self.phi(xa), ya[..., None]], -1))
        a_mask = torch.ones(ya.shape, dtype=torch.bool, device=ya.device)
        smooth = torch.softmax(-sqdist(ub, ub) / (2 * self.log_len_h.exp() ** 2), -1)
        h = torch.zeros(P * N, self.dm, device=ub.device)
        hist = [(x_hat, y_hat)]
        for _ in range(self.n_iter if n_iter is None else n_iter):
            g = grad_feats(xa, ya, x_hat, y_hat)
            gm = g.mean(1, keepdim=True).expand_as(g)
            b_tok = self.b_in(torch.cat([self.phi(ub), yb[..., None], self.phi(x_hat), y_hat[..., None], g, gm], -1))
            look = self.lookup(b_tok, a_tok, x_hat, xa, a_mask)
            h = self.gru((b_tok + look).reshape(P * N, -1), h)
            hb = h.view(P, N, -1)
            step = smooth @ self.local(hb) + self.glob(hb.mean(1, keepdim=True))
            step = self.max_step * torch.tanh(step)
            x_hat = x_hat + step[..., :1]
            y_hat = y_hat + step[..., 1]
            hist.append((x_hat, y_hat))
        return hist


def make_model(method: str, n_iter: int, dm: int) -> nn.Module:
    if method == "operator":
        return RegBack(1, dm=dm, n_iter=n_iter, resid=True, max_step=0.1)
    if method == "optimizer":
        return FieldOptimizer(dm=dm, n_iter=n_iter)
    raise ValueError(f"learned method must be operator or optimizer, got {method}")


def run_learned(model: nn.Module, b: Batch, n: int, x0, y0):
    """-> field history [(x̂, ŷ)] for A's first n points as evidence."""
    xa, ya = b.xa[:, :n], b.ya[:, :n]
    if isinstance(model, RegBack):
        am = torch.ones(ya.shape, dtype=torch.bool, device=ya.device)
        bm = torch.ones(b.yb.shape, dtype=torch.bool, device=ya.device)
        return model.refine(xa, ya, am, b.ub, b.yb, x0, y0, bm)[0]
    return model(xa, ya, b.ub, b.yb, x0, y0)


def field_rms(x_hat, y_hat, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """-> ([P] RMS x error, [P] RMS y error) over inside-window B points."""
    w = b.inside.float()
    den = w.sum(1).clamp_min(1)
    ex = ((((x_hat - b.xb_inA) ** 2).sum(-1) * w).sum(1) / den).sqrt()
    ey = ((((y_hat - b.yb_inA) ** 2) * w).sum(1) / den).sqrt()
    return ex, ey


@torch.no_grad()
def validate(model, method: str, val: dict, device: str) -> dict[str, float]:
    """Field metrics on fixed validation pairs (C10, random start, n_A = 5 and 10), for the training curves."""
    model.eval()
    out = {}
    for n, (b, x0, y0) in val.items():
        ex, ey = field_rms(*run_learned(model, b, n, x0, y0)[-1], b)
        out[f"val/med_err_x_n{n}"] = float(ex.median())
        out[f"val/frac_005_n{n}"] = float((ex < 0.05).float().mean())
        out[f"val/med_err_y_n{n}"] = float(ey.median())
    model.train()
    return out


def train_learned(
    model, method: str, steps: int, device: str, seed=0, batch=32, n_b=64, lr=1e-3, log=print, metrics=None, every=250,
    gamma=0.8, p_identity=0.1,
):  # fmt: skip
    """LUPI field loss from random (and, with p_identity, identity) starts; metrics(values, step) every `every`."""
    gv = torch.Generator(device=device).manual_seed(seed + 4242)
    val_pr = PairPrior(PriorCfg(d=1), 128, seed + 999, device)
    val = {}
    for n in (5, 10):
        b = val_pr.sample("C10", n, 1, n_b)
        val[n] = (b, *init_field(b, gv))
    pr = {c: PairPrior(PriorCfg(d=1), batch, seed + 17 * i, device) for i, c in enumerate(CELLS)}
    tg = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = np.random.default_rng(seed)
    hist, t0 = [], time.time()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        cell = CELLS[int(g.random() < 0.8)]
        n = int(g.integers(2, 21))
        b = pr[cell].sample(cell, n, 1, n_b)
        x0, y0 = init_field(b, tg, identity=bool(g.random() < p_identity))
        its = run_learned(model, b, n, x0, y0)
        T = len(its) - 1
        per_it = torch.stack([field_mse(x, y, b).mean() for x, y in its[1:]])
        loss = (gamma ** torch.arange(T - 1, -1, -1, device=per_it.device) * per_it).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % every == 0:
            vals = {"train/field_mse_final": float(per_it[-1].detach()), "train/loss": float(loss.detach())}
            vals |= validate(model, method, val, device)
            hist.append({"step": step + 1, "seconds": time.time() - t0, **vals})
            if metrics is not None:
                metrics(vals, step + 1)
            log(f"{method} step {step + 1} mse {vals['train/field_mse_final']:.4f} val frac(n=10) "
                f"{vals['val/frac_005_n10']:.2f} med {vals['val/med_err_x_n10']:.3f}")  # fmt: skip
    return pd.DataFrame(hist)


# ---- gradient-descent reference -------------------------------------------------------------------------------


def _softplus_inv(v: torch.Tensor) -> torch.Tensor:
    return v + torch.log(-torch.expm1(-v))


def monotone_map(u: torch.Tensor, c: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """x(u) = c + Σ_k softplus(θ_k) · |segment k ∩ [0, u]| (K equal segments of [0, 1], linear beyond). u [P, N, 1]."""
    K = theta.shape[-1]
    lo = torch.arange(K, device=u.device) / K
    cover = (u - lo).clamp(0, 1 / K)  # [P, N, K]
    cover = torch.cat([cover[..., :-1], (u - lo[-1]).clamp_min(0)], -1)  # last segment extends right
    below = u.clamp_max(0)  # first segment extends left
    sl = F.softplus(theta)[:, None, :]
    return c[:, None, None] + (cover * sl).sum(-1, keepdim=True) + below * sl[..., :1]


def gd_fit(xa, ya, b: Batch, x0, steps=300, lr=0.05, K=8, lam=1e-3, every=0):
    """Fit (monotone map, affine values) from the affine start x0 -> (x̂, ŷ, final energy at ℓ_fine, trajectory)."""
    u = b.ub
    # recover the start's affine (c, a) from two points of x0 (exact for init_field's starts)
    a0 = ((x0[:, -1, 0] - x0[:, 0, 0]) / (u[:, -1, 0] - u[:, 0, 0]).clamp_min(1e-6)).clamp(0.05, 20)
    c = (x0[:, 0, 0] - a0 * u[:, 0, 0]).clone().requires_grad_()
    theta = _softplus_inv(a0)[:, None].repeat(1, K).clone().requires_grad_()
    al = torch.ones(u.shape[0], device=u.device, requires_grad=True)
    be = torch.zeros(u.shape[0], device=u.device, requires_grad=True)
    opt = torch.optim.Adam([c, theta, al, be], lr=lr)
    traj = []
    with torch.enable_grad():
        for t in range(steps):
            ell = 0.15 * (ELL_FINE / 0.15) ** (t / max(steps - 1, 1))
            x_hat, y_hat = monotone_map(u, c, theta), al[:, None] * b.yb + be[:, None]
            if every and t % every == 0:
                traj.append((t, x_hat.detach(), y_hat.detach()))
            e = energy(xa, ya, x_hat, y_hat, ell) + lam * torch.log(F.softplus(theta)).pow(2).mean(-1)
            opt.zero_grad()
            e.sum().backward()
            opt.step()
    with torch.no_grad():
        x_hat, y_hat = monotone_map(u, c, theta), al[:, None] * b.yb + be[:, None]
        if every:
            traj.append((steps, x_hat, y_hat))
        return x_hat, y_hat, energy(xa, ya, x_hat, y_hat, ELL_FINE), traj


# ---- evaluation -----------------------------------------------------------------------------------------------


def _pair_rows(cell, n, init, method, b: Batch, x_hat, y_hat, x0):
    """Per-pair metrics as dict rows."""
    xa, ya = b.xa[:, :n], b.ya[:, :n]
    ex, ey = field_rms(x_hat, y_hat, b)
    e0, _ = field_rms(x0, b.yb, b)
    e_fit = energy(xa, ya, x_hat, y_hat, ELL_FINE)
    e_true = energy(xa, ya, b.xb_inA, b.yb_inA, ELL_FINE)
    xq = b.xa[:, 20:]  # queries (eval_set: 20 context points, then the tests)
    lo = torch.where(b.inside, b.xb_inA[..., 0], torch.inf).amin(1, keepdim=True)
    hi = torch.where(b.inside, b.xb_inA[..., 0], -torch.inf).amax(1, keepdim=True)
    cov = ((xq[..., 0] >= lo) & (xq[..., 0] <= hi)).float()
    sq = (render(xq, x_hat, y_hat, ELL_FINE) - b.fa_test[:, 20:]) ** 2
    pred = ((sq * cov).sum(1) / cov.sum(1).clamp_min(1)).sqrt()
    cols = dict(err_x=ex, err_y=ey, err_x0=e0, e_fit=e_fit, e_true=e_true, pred_rmse=pred)
    cols = {k: v.cpu().numpy() for k, v in cols.items()}
    return [
        dict(cell=cell, n_a=n, init=init, method=method, pair=p, **{k: float(v[p]) for k, v in cols.items()})
        for p in range(len(ex))
    ]


def _starts(b: Batch, cell: str, seed: int, identity: bool, k: int = 1):
    """Deterministic starts per (cell, seed): the first is the shared start of every method."""
    g = torch.Generator(device=b.ub.device).manual_seed(seed + 101 * CELLS.index(cell))
    first = init_field(b, g, identity)
    return [first] + [init_field(b, g) for _ in range(k - 1)]


@torch.no_grad()
def evaluate(method: str, model, device: str, n_pairs=256, n_b=64, restarts=8, gd_steps=300, seed=12345):
    """Rows per (cell, n_a, init, method, pair). For `gd` also `gd_best8`, `gd_oracle8` and the `start` / `truth`
    references (the field before moving, and B_inA itself)."""
    rows = []
    if model is not None:
        model.eval()
    for cell in CELLS:
        b = eval_set(1, cell, n_pairs, 128, n_b, seed, device)
        for init in INITS:
            starts = _starts(b, cell, seed, init == "identity", restarts if method == "gd" else 1)
            x0, y0 = starts[0]
            for n in N_A_EVAL:
                xa, ya = b.xa[:, :n], b.ya[:, :n]
                if method == "gd":
                    fits = [gd_fit(xa, ya, b, xs, steps=gd_steps)[:3] for xs, _ in starts]
                    rows += _pair_rows(cell, n, init, "gd", b, fits[0][0], fits[0][1], x0)
                    en = torch.stack([f[2] for f in fits])  # [R, P]
                    errs = torch.stack([field_rms(f[0], f[1], b)[0] for f in fits])
                    for name, pick in (("gd_best8", en.argmin(0)), ("gd_oracle8", errs.argmin(0))):
                        xs = torch.stack([f[0] for f in fits])[pick, torch.arange(n_pairs)]
                        ys = torch.stack([f[1] for f in fits])[pick, torch.arange(n_pairs)]
                        rows += _pair_rows(cell, n, init, name, b, xs, ys, x0)
                    rows += _pair_rows(cell, n, init, "start", b, x0, y0, x0)
                    rows += _pair_rows(cell, n, init, "truth", b, b.xb_inA, b.yb_inA, x0)
                else:
                    with torch.no_grad():
                        xf, yf = run_learned(model, b, n, x0, y0)[-1]
                    rows += _pair_rows(cell, n, init, method, b, xf, yf, x0)
    return pd.DataFrame(rows)


@torch.no_grad()
def examples(method: str, model, device: str, n_pairs=6, n_b=64, gd_steps=300, seed=12345):
    """Long tables for 1d plots (C10 and C00, random start, n_A ∈ {5, 10}, the first `n_pairs` evaluation pairs):
    `points` (role A / A_test / B with pos, y and, for B, the truth x_true, y_true) and `traj` (the field (x̂, ŷ) of
    every B point at every iteration; gd: every 30 steps)."""
    pts, trs = [], []
    for cell in CELLS:
        b = eval_set(1, cell, 256, 128, n_b, seed, device)
        x0, y0 = _starts(b, cell, seed, False)[0]
        b = Batch(*(t[:n_pairs] if torch.is_tensor(t) and t.dim() > 0 else t for t in b.__dict__.values()))
        x0, y0 = x0[:n_pairs], y0[:n_pairs]
        for p in range(n_pairs):
            for i in range(n_b):
                pts.append((cell, p, "B", i, float(b.ub[p, i, 0]), float(b.yb[p, i]), float(b.xb_inA[p, i, 0]),
                            float(b.yb_inA[p, i]), bool(b.inside[p, i])))  # fmt: skip
            for i in list(range(20)) + list(range(20, b.xa.shape[1])):
                role = "A" if i < 20 else "A_test"
                pts.append((cell, p, role, i, float(b.xa[p, i, 0]), float(b.ya[p, i]), float("nan"),
                            float(b.fa_test[p, i]), True))  # fmt: skip
        for n in (5, 10):
            xa, ya = b.xa[:, :n], b.ya[:, :n]
            if method == "gd":
                traj = [(t, x, y) for t, x, y in gd_fit(xa, ya, b, x0, steps=gd_steps, every=30)[3]]
            else:
                model.eval()
                with torch.no_grad():
                    traj = [(t, x, y) for t, (x, y) in enumerate(run_learned(model, b, n, x0, y0))]
            for t, x, y in traj:
                x, y = x.cpu().numpy(), y.cpu().numpy()
                for p in range(n_pairs):
                    for i in range(n_b):
                        trs.append((cell, n, p, t, i, float(x[p, i, 0]), float(y[p, i])))
    return (
        pd.DataFrame(pts, columns=["cell", "pair", "role", "i", "pos", "y", "x_true", "y_true", "flag"]),
        pd.DataFrame(trs, columns=["cell", "n_a", "pair", "t", "i", "x", "y"]),
    )
