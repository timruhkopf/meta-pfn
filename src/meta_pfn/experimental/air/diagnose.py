"""Post hoc diagnostics of AIR (issue #19, notebook 10): why stages of the pilot failed. Not pre-registered.

Table `pfail` (long: J, n_a, task, metric, value):
- A, J = 2, finite K with a perfect proposal: `nll_exact_post{K}` = K exact grid-posterior draws, equal weights,
  exact Gaussian predictives; `nll_dec_post{K}` = the same draws with the decoder's predictives. If these show the q-IS
  dependence on K, the K cost belongs to the mixture of sharp predictives, not to the proposal.
- B, all J, K scan: `nll_qis{K}`, `ess_qis{K}`; on one set of K_ess draws, `ess_exact` (exact dense-B evidence) and
  `ess_perm8` (decoder evidence over 8 orders of A), with `nll_exact_q` (exact weights and predictives).
- C, J = 2, fine local grid (cells far below the posterior width): `kl_exact_q`, `kl_dec_q`, `kl_exact_dec`, and
  the posterior widths `w_exact`, `w_dec`, `w_q` (det(cov)^(1/4)), with `w_grid_cell` for the 101² cell width;
  `kl_exact_dec_clean`, `w_dec_clean`: the decoder given noise-free B values (same 128 x).
"""

import dataclasses
import math
import time

import pandas as pd
import torch

from . import reference as R
from .decoder import Decoder
from .estimator import decode, importance
from .prior import log_prior
from .proposal import Proposal
from .train import N_A_EVAL, eval_tasks


def _width(p: torch.Tensor, pts: torch.Tensor) -> float:
    m = (p[:, None] * pts).sum(0)
    d = pts - m
    return float(torch.linalg.det((p[:, None, None] * d[:, :, None] * d[:, None]).sum(0)).clamp_min(0) ** 0.25)


@torch.no_grad()
def diagnose(dec: Decoder, props: dict[int, Proposal], device, *, n_tasks: int, seed: int, ks=(32, 128, 512, 2048),
             k_ess: int = 128, n_scan=(1, 3, 5, 10), fid_tasks: int = 32, fid_n=(1, 2, 5, 10), fine: int = 81,
             log=print, g_norm: str = "minmax") -> pd.DataFrame:  # fmt: skip
    dec.eval()
    rows = []

    def add(J, n, i, metric, v):
        rows.append({"J": J, "n_a": n, "task": i, "metric": metric, "value": float(v)})

    g_pts = R.grid(101, device=device)
    if 2 in props:  # A
        base = eval_tasks(2, n_tasks, seed, device, g_norm=g_norm)
        for n in N_A_EVAL:
            t0, t = time.time(), base.head(n)
            P = t.xa.shape[0]
            post = []
            for i in range(P):
                gr = R.grid_ref(t, i, g_pts)
                post.append(gr["logpost"].exp())
                add(2, n, i, "nll_grid", -gr["lp"].mean())
                for K in (8, *ks):
                    _, lq = R.exact_terms(t, i, g_pts[torch.multinomial(post[i], K, replacement=True)])
                    add(2, n, i, f"nll_exact_post{K}", -(torch.logsumexp(lq, 0) - math.log(K)).mean())
            post = torch.stack(post)
            for K in (32, 128):
                th = g_pts[torch.multinomial(post, K, replacement=True)]
                _, lq, _ = decode(dec, t, th, n_perm=1)
                lp = (torch.logsumexp(lq, 1) - math.log(K)).mean(1)
                for i in range(P):
                    add(2, n, i, f"nll_dec_post{K}", -lp[i])
            log(f"A n_a={n} {time.time() - t0:.0f}s")
    for J, q in props.items():  # B
        q.eval()
        base = eval_tasks(J, n_tasks, seed, device, g_norm=g_norm)
        for n in n_scan:
            t0, t = time.time(), base.head(n)
            P = t.xa.shape[0]
            for K in ks:
                for c in range(0, P, 16):
                    sub = t.sub(slice(c, c + 16))
                    th, lqm = q.sample_mix(sub.xb, sub.yb, sub.xa, sub.ya, K)
                    r = importance(dec, sub, th, lqm)
                    for i in range(sub.xa.shape[0]):
                        add(J, n, c + i, f"nll_qis{K}", -r["lp1"][i].mean())
                        add(J, n, c + i, f"ess_qis{K}", r["ess"][i])
            th, lqm = q.sample_mix(t.xb, t.yb, t.xa, t.ya, k_ess)
            r8 = importance(dec, t, th, lqm, n_perm=8)
            for i in range(P):
                rx = R.is_exact(t, i, th[i], lqm[i])
                add(J, n, i, "ess_exact", rx["ess"] / k_ess)
                add(J, n, i, "nll_exact_q", -rx["lp"].mean())
                add(J, n, i, "ess_perm8", r8["ess"][i])
            log(f"B J={J} n_a={n} {time.time() - t0:.0f}s")
    if 2 in props:  # C
        q = props[2]
        base = eval_tasks(2, n_tasks, seed, device, g_norm=g_norm)
        for n in fid_n:
            t0, t = time.time(), base.head(n)
            for i in range(min(fid_tasks, t.xa.shape[0])):
                ti = t.sub(slice(i, i + 1))
                gr = R.grid_ref(t, i, g_pts)
                p = gr["logpost"].exp()
                m = (p[:, None] * g_pts).sum(0)
                sd = ((p[:, None] * (g_pts - m) ** 2).sum(0)).sqrt()
                d = q.dist(ti.xb, ti.yb, ti.xa, ti.ya)
                sq = d.sample((1024,))[:, 0]
                half = torch.maximum(6 * sd, 4 * sq.std(0)).clamp_min(0.24)
                c = sq.mean(0)
                lo, hi = torch.minimum(m, c) - half, torch.maximum(m, c) + half
                ax = [torch.linspace(float(lo[k]), float(hi[k]), fine, device=device) for k in range(2)]
                pts = torch.cartesian_prod(*ax)
                cell = float((ax[0][1] - ax[0][0]) * (ax[1][1] - ax[1][0]))
                ll, _ = R.exact_terms(t, i, pts)
                ev, _, _ = decode(dec, ti, pts[None], n_perm=2)
                le = log_prior(pts) + ll
                ld = log_prior(pts) + ev[0]
                le, ld = le - torch.logsumexp(le, 0), ld - torch.logsumexp(ld, 0)
                lq = d.log_prob(pts[:, None])[:, 0] + math.log(cell)
                pe, pd_ = le.exp(), ld.exp()
                add(2, n, i, "kl_exact_q", (pe * (le - lq)).sum())
                add(2, n, i, "kl_dec_q", (pd_ * (ld - lq)).sum())
                add(2, n, i, "kl_exact_dec", (pe * (le - ld)).sum())
                add(2, n, i, "w_exact", _width(pe, pts))
                add(2, n, i, "w_dec", _width(pd_, pts))
                add(2, n, i, "w_q", _width(lq.exp() / lq.exp().sum(), pts))
                add(2, n, i, "w_grid_cell", 0.08)
                clean = dataclasses.replace(ti, yb=ti.g(ti.xb))  # noise-free B: the finite-B share
                ev_c, _, _ = decode(dec, clean, pts[None], n_perm=2)
                lc = log_prior(pts) + ev_c[0]
                lc = lc - torch.logsumexp(lc, 0)
                add(2, n, i, "kl_exact_dec_clean", (pe * (le - lc)).sum())
                add(2, n, i, "w_dec_clean", _width(lc.exp(), pts))
            log(f"C n_a={n} {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)
