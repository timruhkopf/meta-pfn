"""AIR pilot 3 (issue #19, notebook 11; docs/a2b_air_pilot1_review_2026-11-03.md): the Rao-Blackwellised transform
latent θ_c = θ̃ + √Σ ε, plateau training, a clean A-only PFN, and the pilot-3 evaluation.

Augmented model: θ̃ ~ N(0, I), θ_c | θ̃ ~ N(θ̃, Σ I), data from θ̃ only, so p(y* | A, B) = ∫ p(θ_c | A, B) p(y* | A, B,
θ_c) dθ_c exactly, with p(θ_c) = N(0, (1 + Σ) I). Σ = 0 is AIR as in pilots 1-2.
- decoder: B moved by T_{θ_c}⁻¹, Σ given as an input (Decoder(sig_cond=True)); Σ drawn per task from `sig_list`; targets
  from the true θ̃ (soft targets of the noise-free values), so it learns p(· | A, B, θ_c, Σ);
- proposal: NPE on θ_c, q(θ_c | A, B, Σ), Σ per task from `sig_list` (Proposal(sig_cond=True));
- clean A-only PFN: the same Decoder, trained with B masked on every pass (the user's baseline; no shared weights);
- plateau rule: constant learning rate after warm-up until the validation NLL has improved by < `tol` over the last
  `window` steps, then a cosine anneal over `anneal` steps, then stop; `max_steps` caps it.
"""

import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest

from . import reference as R
from .decoder import Decoder, decoder_loss
from .estimator import decode, importance, null_branch
from .prior import PriorCfg, Tasks, flow, log_prior, sample_tasks
from .proposal import Proposal
from .train import eval_tasks

N_A_RB = (1, 2, 3, 5, 10)


class Plateau:
    """Learning rate and stop decision for the plateau rule (module docstring)."""

    def __init__(self, lr: float, max_steps: int, warm: int = 500, window: int = 5000, tol: float = 0.01,
                 anneal: int = 5000, min_steps: int = 10000):  # fmt: skip
        self.lr, self.max_steps, self.warm, self.window, self.tol = lr, max_steps, warm, window, tol
        self.anneal, self.min_steps = anneal, min_steps
        self.hist: list[tuple[int, float]] = []
        self.start: int | None = None  # step at which the anneal began

    def rate(self, step: int) -> float:
        if self.start is None:
            return self.lr * min(1.0, (step + 1) / self.warm)
        return self.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - self.start) / self.anneal)))

    def observe(self, step: int, val: float) -> None:
        self.hist.append((step, val))
        if self.start is None and step >= self.min_steps:
            best_now = min(v for _, v in self.hist)
            old = [v for s, v in self.hist if s <= step - self.window]
            if old and min(old) - best_now < self.tol:
                self.start = step
        if self.start is None and step >= self.max_steps - self.anneal:
            self.start = step

    def done(self, step: int) -> bool:
        return (self.start is not None and step - self.start >= self.anneal) or step >= self.max_steps


def _set_lr(opt, lr):
    for pg in opt.param_groups:
        pg["lr"] = lr


def _sig(rng, sig_list, P, device):
    return torch.as_tensor(rng.choice(sig_list, size=P), dtype=torch.float32, device=device)


def train_decoder_rb(dec: Decoder, max_steps: int, device, *, seed: int, batch: int, lr: float, j_mix, sig_list,
                     g_norm: str, clip: float = 1.0, aonly: bool = False, every: int = 250, metrics=None,
                     log=print) -> pd.DataFrame:  # fmt: skip
    """The Σ-conditioned decoder (with its null head), or with aonly=True the clean A-only PFN."""
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    val = sample_tasks(PriorCfg(J=2, g_norm=g_norm), 256, 5, torch.Generator(device=device).manual_seed(seed + 7),
                       device)  # fmt: skip
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    pl = Plateau(lr, max_steps)
    hist, t0, step = [], time.time(), 0
    while True:
        _set_lr(opt, pl.rate(step))
        J, n = int(rng.choice(j_mix)), int(rng.integers(1, 11))
        t = sample_tasks(PriorCfg(J=J, g_norm=g_norm), batch, n, g, device)
        dec.train()
        if aonly:
            loss = l0 = decoder_loss(dec, t.xb, t.yb, t.xa, t.ya, t.xq, t.fa, t.fq, t.sig_a, use_b=False)
            lt = l0 * 0
        else:
            s = _sig(rng, sig_list, batch, device)
            th_c = t.theta + s.sqrt()[:, None] * torch.randn(t.theta.shape, generator=g, device=device)
            xb_t = flow(t.xb, th_c, t.s_t, inverse=True)
            lt = decoder_loss(dec, xb_t, t.yb, t.xa, t.ya, t.xq, t.fa, t.fq, t.sig_a, sig_c=s)
            l0 = decoder_loss(dec, xb_t, t.yb, t.xa, t.ya, t.xq, t.fa, t.fq, t.sig_a, use_b=False, sig_c=s * 0)
            loss = lt + l0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), clip)
        opt.step()
        if step % every == 0:
            dec.eval()
            with torch.no_grad():
                _, lq0 = null_branch(dec, val, n_perm=1)
                v = {"val_aonly_n5": float(-lq0.mean())}
                if not aonly:
                    _, lq, _ = decode(dec, val, val.theta[:, None], n_perm=1, sig=torch.zeros(()))
                    v["val_teacher_n5"] = float(-lq.mean())
            key = "val_aonly_n5" if aonly else "val_teacher_n5"
            pl.observe(step, v[key])
            row = {"step": step, "loss_teacher": float(lt), "loss_null": float(l0), "grad_norm": float(gn), "J": J,
                   "n_a": n, "lr": pl.rate(step), "s": time.time() - t0, **v}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({f"train/{k}": float(row[k]) for k in ("loss_teacher", "loss_null", "grad_norm", "lr")}
                        | {f"val/{k}": float(x) for k, x in v.items()}, step=step)  # fmt: skip
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
        if pl.done(step):
            break
        step += 1
    return pd.DataFrame(hist)


def train_proposal_rb(q: Proposal, J: int, max_steps: int, device, *, seed: int, batch: int, lr: float, sig_list,
                      g_norm: str, clip: float = 20.0, every: int = 250, metrics=None,
                      log=print) -> pd.DataFrame:  # fmt: skip
    """NPE on θ_c with Σ per task from sig_list; validation NLL at Σ = 0, n_A = 5 drives the plateau rule."""
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = PriorCfg(J=J, g_norm=g_norm)
    val = sample_tasks(cfg, 512, 5, torch.Generator(device=device).manual_seed(seed + 7), device)
    opt = torch.optim.AdamW(q.parameters(), lr=lr, weight_decay=1e-4)
    pl = Plateau(lr, max_steps)
    hist, t0, step = [], time.time(), 0
    while True:
        _set_lr(opt, pl.rate(step))
        n = int(rng.integers(1, 11))
        t = sample_tasks(cfg, batch, n, g, device)
        s = _sig(rng, sig_list, batch, device)
        th_c = t.theta + s.sqrt()[:, None] * torch.randn(t.theta.shape, generator=g, device=device)
        q.train()
        loss = q.nll(t.xb, t.yb, t.xa, t.ya, th_c, s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(q.parameters(), clip)
        opt.step()
        if step % every == 0:
            q.eval()
            with torch.no_grad():
                vnll = float(q.nll(val.xb, val.yb, val.xa, val.ya, val.theta, 0.0))
            pl.observe(step, vnll)
            vp = float(-log_prior(val.theta).mean())
            row = {"step": step, "nll": float(loss.detach()), "grad_norm": float(gn), "n_a": n, "lr": pl.rate(step),
                   "s": time.time() - t0, "val_nll_n5": vnll, "val_prior_nll": vp}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({"train/nll": row["nll"], "train/grad_norm": row["grad_norm"], "train/lr": row["lr"],
                         "val/nll_n5": vnll}, step=step)  # fmt: skip
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
        if pl.done(step):
            break
        step += 1
    return pd.DataFrame(hist)


def _theta_c(t: Tasks, sig: float, seed: int) -> torch.Tensor:
    """The evaluation tasks' true θ_c for Σ (fixed per task and Σ)."""
    g = torch.Generator(device=t.theta.device).manual_seed(seed)
    return t.theta + math.sqrt(sig) * torch.randn(t.theta.shape, generator=g, device=t.theta.device)


@torch.no_grad()
def evaluate_rb(dec: Decoder, props: dict[int, Proposal], aonly: Decoder, device, *, sig_list, n_tasks: int, seed: int,
                g_norm: str, ks0=(8, 32, 128, 512, 2048), ks=(8, 32, 128), qmc_ks=(8, 32), k_ref: int = 4096,
                n_sbc: int = 255, cal_tasks: int = 128, cal_grid: int = 61, chunk: int = 16,
                log=print) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """-> (nll, diag), long format with columns J, sig, n_a, task, method / metric, nll / value.

    nll methods: a_only_clean (the clean A-only PFN), a_only_null (the decoder's null head), teacher (Σ = 0, true θ̃),
    grid (G-ref, J = 2), isref (J = 2: prior IS, K_ref; J = 16: q-IS at Σ = 0, K_ref), qis{K} (i.i.d.; K ∈ ks0 at Σ = 0,
    ks at Σ > 0), qmc{K} (scrambled Sobol). sig = -1 marks Σ-independent rows.
    diag: ess_qis{K}, ess_qmc{K}, ess_isref; sbc_rank_j, hpd (q vs the true θ_c); at Σ = 0, J = 2, for `cal_tasks` tasks
    on a cal_grid² grid: kl_dq = KL(π̂_D ‖ q) and pit_j = P_{π̂_D}(θ̃_j ≤ θ̃_true,j) (marginal PIT of the
    decoder-implied posterior π̂_D ∝ p(θ̃) p̂(A | B, θ̃)).
    """
    dec.eval()
    aonly.eval()
    nll_rows, diag_rows = [], []
    g101 = R.grid(101, device=device)
    gcal = R.grid(cal_grid, device=device)
    cell = (8.0 / (cal_grid - 1)) ** 2

    def put(J, sig, n, method, lp, off=0):
        for i in range(lp.shape[0]):
            nll_rows.append(
                {"J": J, "sig": sig, "n_a": n, "task": off + i, "method": method, "nll": -float(lp[i].mean())}
            )

    def dg(J, sig, n, metric, v, off=0):
        for i in range(v.shape[0]):
            diag_rows.append({"J": J, "sig": sig, "n_a": n, "task": off + i, "metric": metric, "value": float(v[i])})

    for J, q in props.items():
        q.eval()
        base = eval_tasks(J, n_tasks, seed, device, g_norm=g_norm)
        for n in N_A_RB:
            t0, t = time.time(), base.head(n)
            P = t.xa.shape[0]
            put(J, -1, n, "a_only_clean", null_branch(aonly, t)[1])
            put(J, -1, n, "a_only_null", null_branch(dec, t)[1])
            put(J, -1, n, "teacher", decode(dec, t, t.theta[:, None], sig=torch.zeros(()))[1][:, 0])
            if J == 2:
                put(J, -1, n, "grid", torch.stack([R.grid_ref(t, i, g101)["lp"] for i in range(P)]))
            ref_lp, ref_ess = [], []
            for c in range(0, P, 8):  # IS-ref
                sub = t.sub(slice(c, c + 8))
                if J == 2:
                    th = torch.randn(sub.xa.shape[0], k_ref, J, device=device)
                    r = importance(dec, sub, th, log_prior(th), sig=0.0)
                else:
                    th, lqm = q.sample_mix(sub.xb, sub.yb, sub.xa, sub.ya, k_ref, sig=0.0)
                    r = importance(dec, sub, th, lqm, sig=0.0)
                ref_lp.append(r["lp1"])
                ref_ess.append(r["ess"] * k_ref)
            put(J, -1, n, "isref", torch.cat(ref_lp))
            dg(J, -1, n, "ess_isref", torch.cat(ref_ess))
            for sig in sig_list:
                for K in ks0 if sig == 0 else ks:
                    for c in range(0, P, chunk):
                        sub = t.sub(slice(c, c + chunk))
                        th, lqm = q.sample_mix(sub.xb, sub.yb, sub.xa, sub.ya, K, sig=sig)
                        r = importance(dec, sub, th, lqm, sig=sig)
                        put(J, sig, n, f"qis{K}", r["lp1"], c)
                        dg(J, sig, n, f"ess_qis{K}", r["ess"], c)
                for K in qmc_ks:
                    for c in range(0, P, chunk):
                        sub = t.sub(slice(c, c + chunk))
                        th, lqm = q.sample_mix(sub.xb, sub.yb, sub.xa, sub.ya, K, sig=sig, qmc=True, seed=seed + c)
                        r = importance(dec, sub, th, lqm, sig=sig)
                        put(J, sig, n, f"qmc{K}", r["lp1"], c)
                        dg(J, sig, n, f"ess_qmc{K}", r["ess"], c)
                th_c = _theta_c(t, sig, seed + 13 * n + int(1000 * sig))
                d = q.dist(t.xb, t.yb, t.xa, t.ya, sig)
                s = d.sample((n_sbc,)).transpose(0, 1)
                ranks = (s < th_c[:, None]).sum(1)
                for jj in range(J):
                    dg(J, sig, n, f"sbc_rank_{jj}", ranks[:, jj])
                lq_s = d.log_prob(s.transpose(0, 1)).transpose(0, 1)
                dg(J, sig, n, "hpd", (lq_s > d.log_prob(th_c)[:, None]).float().mean(1))
            if J == 2:  # P3.1 / P3.2 at Σ = 0
                lqg = q.dist(t.xb, t.yb, t.xa, t.ya, 0.0).log_prob(gcal[:, None].expand(-1, P, -1))  # [G, P]
                for i in range(min(cal_tasks, P)):
                    ti = t.sub(slice(i, i + 1))
                    ev = decode(dec, ti, gcal[None], sig=torch.zeros(()))[0][0]
                    lp = log_prior(gcal) + ev
                    lp = lp - torch.logsumexp(lp, 0)
                    lq = lqg[:, i] + math.log(cell)
                    p = lp.exp()
                    diag_rows.append({"J": J, "sig": 0.0, "n_a": n, "task": i, "metric": "kl_dq",
                                      "value": float((p * (lp - lq)).sum())})  # fmt: skip
                    for jj in range(2):
                        below = (gcal[:, jj] < t.theta[i, jj]).float() + 0.5 * (gcal[:, jj] == t.theta[i, jj]).float()
                        diag_rows.append({"J": J, "sig": 0.0, "n_a": n, "task": i, "metric": f"pit_{jj}",
                                          "value": float((p * below).sum())})  # fmt: skip
            log(f"J={J} n_a={n} done in {time.time() - t0:.0f}s")
    return pd.DataFrame(nll_rows), pd.DataFrame(diag_rows)


def ks_uniform(x: np.ndarray) -> float:
    return float(kstest(x, "uniform").pvalue)
