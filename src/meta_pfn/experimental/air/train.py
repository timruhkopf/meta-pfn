"""Training and evaluation of AIR (proposal §3.5, §5–§8; issue #19, notebook 10).

Training (independent, fresh prior simulations, z = 1, true θ):
- decoder D_ψ: J drawn per batch from `j_mix`, n_A ~ U{1..10}; loss = soft-target CE (x* and evidence queries) with
  B transported by the true θ, + the same with B dropped (null head);
- proposal q_φ (one per J): NPE, −log q(θ̃_true | A, B).
Evaluation (fixed tasks per (J, n_A), nested in n_A; tables `nll` and `diag`, long format):
- nll: a_only (null head), teacher (D at the true θ), grid (G-ref, J = 2), isref (prior IS, K_ref, J = 2),
  qis{K} (q-IS, K ∈ ks), prior{K} (prior IS at the q-IS K, V5), isexact (dense-B IS with q_mix, J > 2);
- diag: ess_qis{K}, ess_prior{K}, ess_isref, ess_isexact; sbc_rank_j (of 255 q samples), hpd (share of samples
  with a higher q density than the truth), kl_grid_q (J = 2), evid_kl and evid_over95 (decoder evidence vs exact,
  41² grid, J = 2), perm_std (8 orders at the true θ), pred_rel (|g(T_s x*) − g(T x*)| / |T_s x* − T x*| over q
  samples), t_err (mean |T_s x* − T x*| over q samples).
"""

import math
import time

import numpy as np
import pandas as pd
import torch

from . import reference as R
from .decoder import Decoder, decoder_loss
from .estimator import decode, importance, null_branch
from .prior import PriorCfg, Tasks, flow, log_prior, sample_tasks
from .proposal import Proposal

N_A_EVAL = (1, 2, 3, 5, 8, 10)


def _lr(opt, step: int, steps: int, lr: float, warm: int = 500):
    for pg in opt.param_groups:
        pg["lr"] = lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))


def train_decoder(dec: Decoder, steps: int, device, *, seed: int, batch: int, lr: float, j_mix, h_kind: str,
                  every: int, metrics=None, log=print) -> pd.DataFrame:  # fmt: skip
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    val = sample_tasks(
        PriorCfg(J=2, h_kind=h_kind), 256, 5, torch.Generator(device=device).manual_seed(seed + 7), device
    )
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    hist, t0 = [], time.time()
    for step in range(steps):
        _lr(opt, step, steps, lr)
        J, n = int(rng.choice(j_mix)), int(rng.integers(1, 11))
        t = sample_tasks(PriorCfg(J=J, h_kind=h_kind), batch, n, g, device)
        xb_t = t.b_in_a(t.theta)
        dec.train()
        lt = decoder_loss(dec, xb_t, t.yb, t.xa, t.ya, t.xq, t.fa, t.fq, t.sig_a)
        l0 = decoder_loss(dec, xb_t, t.yb, t.xa, t.ya, t.xq, t.fa, t.fq, t.sig_a, use_b=False)
        loss = lt + l0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if step % every == 0 or step == steps - 1:
            v = val_decoder(dec, val)
            row = {"step": step, "loss_teacher": float(lt), "loss_null": float(l0), "grad_norm": float(gn), "J": J,
                   "n_a": n, "s": time.time() - t0, **v}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({f"train/{k}": float(row[k]) for k in ("loss_teacher", "loss_null", "grad_norm")}
                        | {f"val/{k}": float(x) for k, x in v.items()}, step=step)  # fmt: skip
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
    return pd.DataFrame(hist)


@torch.no_grad()
def val_decoder(dec: Decoder, t: Tasks) -> dict:
    dec.eval()
    _, lq, _ = decode(dec, t, t.theta[:, None], n_perm=1)
    _, lq0 = null_branch(dec, t, n_perm=1)
    return {"val_teacher_n5": float(-lq.mean()), "val_aonly_n5": float(-lq0.mean())}


def train_proposal(q: Proposal, J: int, steps: int, device, *, seed: int, batch: int, lr: float, h_kind: str,
                   every: int, metrics=None, log=print) -> pd.DataFrame:  # fmt: skip
    g = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = PriorCfg(J=J, h_kind=h_kind)
    val = sample_tasks(cfg, 512, 5, torch.Generator(device=device).manual_seed(seed + 7), device)
    opt = torch.optim.AdamW(q.parameters(), lr=lr, weight_decay=1e-4)
    hist, t0 = [], time.time()
    for step in range(steps):
        _lr(opt, step, steps, lr)
        n = int(rng.integers(1, 11))
        t = sample_tasks(cfg, batch, n, g, device)
        q.train()
        loss = q.nll(t.xb, t.yb, t.xa, t.ya, t.theta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step()
        if step % every == 0 or step == steps - 1:
            q.eval()
            with torch.no_grad():
                vnll = float(q.nll(val.xb, val.yb, val.xa, val.ya, val.theta))
            row = {"step": step, "nll": float(loss), "grad_norm": float(gn), "n_a": n, "s": time.time() - t0,
                   "val_nll_n5": vnll, "val_prior_nll": float(-log_prior(val.theta).mean())}  # fmt: skip
            hist.append(row)
            if metrics:
                metrics({"train/nll": row["nll"], "val/nll_n5": vnll}, step=step)
            log(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()))
    return pd.DataFrame(hist)


def eval_tasks(J: int, n_tasks: int, seed: int, device, h_kind: str = "id") -> Tasks:
    """Fixed evaluation tasks per J (nested in n_A: the first n of 10 A points)."""
    return sample_tasks(PriorCfg(J=J, h_kind=h_kind), n_tasks, max(N_A_EVAL),
                        torch.Generator(device=device).manual_seed(seed + 101 * J), device)  # fmt: skip


@torch.no_grad()
def evaluate(dec: Decoder, props: dict[int, Proposal], device, *, n_tasks: int, seed: int, ks=(8, 32, 128),
             k_ref: int = 4096, k_exact: int = 10_000, n_sbc: int = 255, fid_tasks: int = 32,
             fid_n=(1, 2, 5, 10), log=print) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    dec.eval()
    nll_rows, diag_rows = [], []
    g_pts = R.grid(101, device=device)
    g_fid = R.grid(41, device=device)
    for J, q in props.items():
        q.eval()
        base = eval_tasks(J, n_tasks, seed, device)
        for n in N_A_EVAL:
            t0 = time.time()
            t = base.head(n)
            P = t.xa.shape[0]

            def put(method, lp, J=J, n=n, P=P, rows=nll_rows):
                for i in range(P):
                    rows.append({"J": J, "n_a": n, "task": i, "method": method, "nll": -float(lp[i].mean())})

            def diag(metric, v, J=J, n=n, P=P, rows=diag_rows):
                for i in range(P):
                    rows.append({"J": J, "n_a": n, "task": i, "metric": metric, "value": float(v[i])})

            _, lp0 = null_branch(dec, t)
            put("a_only", lp0)
            _, lqt, _ = decode(dec, t, t.theta[:, None])
            put("teacher", lqt[:, 0])
            d = q.dist(t.xb, t.yb, t.xa, t.ya)
            for K in ks:
                th, lqm = q.sample_mix(t.xb, t.yb, t.xa, t.ya, K)
                r = importance(dec, t, th, lqm)
                put(f"qis{K}", r["lp1"])
                diag(f"ess_qis{K}", r["ess"])
                thp = torch.randn(P, K, J, device=device)
                rp = importance(dec, t, thp, log_prior(thp))
                put(f"prior{K}", rp["lp1"])
                diag(f"ess_prior{K}", rp["ess"])
            # q calibration: SBC ranks, HPD level, prediction-relevant error
            s = d.sample((n_sbc,)).transpose(0, 1)  # [P, S, J]
            ranks = (s < t.theta[:, None]).sum(1)  # [P, J]
            for jj in range(J):
                diag(f"sbc_rank_{jj}", ranks[:, jj])
            lq_s = d.log_prob(s.transpose(0, 1)).transpose(0, 1)
            lq_t = d.log_prob(t.theta)
            diag("hpd", (lq_s > lq_t[:, None]).float().mean(1))
            xq = t.xq[:, None].expand(-1, n_sbc, -1)
            u_s = flow(xq, s, t.s_t)  # [P, S, Q]
            u_t = flow(t.xq, t.theta, t.s_t)[:, None]
            gs = torch.stack([t.g.at(torch.full((n_sbc,), i, device=device, dtype=torch.long))(u_s[i])
                              for i in range(P)])  # fmt: skip
            gt = t.g(u_t[:, 0])[:, None]
            terr = (u_s - u_t).abs().mean((1, 2))
            diag("t_err", terr)
            diag("pred_rel", (gs - gt).abs().mean((1, 2)) / terr.clamp_min(1e-6))
            if J == 2:
                lp_g, kl, isref, essref = [], [], [], []
                for i in range(P):
                    gr = R.grid_ref(t, i, g_pts)
                    lp_g.append(gr["lp"])
                    lqg = d.log_prob(g_pts[:, None].expand(-1, P, -1))[:, i]  # q on the grid for task i
                    lqg = lqg - torch.logsumexp(lqg, 0)
                    kl.append(float((gr["logpost"].exp() * (gr["logpost"] - lqg)).sum()))
                put("grid", torch.stack(lp_g))
                diag("kl_grid_q", torch.tensor(kl))
                for c in range(0, P, 8):  # prior IS with K_ref (IS-ref)
                    sub = t.sub(slice(c, c + 8))
                    thp = torch.randn(sub.xa.shape[0], k_ref, J, device=device)
                    rr = importance(dec, sub, thp, log_prior(thp))
                    isref.append(rr["lp1"])
                    essref.append(rr["ess"] * k_ref)
                put("isref", torch.cat(isref))
                diag("ess_isref", torch.cat(essref))
                if n in fid_n:  # evidence fidelity on a 41² grid, permutation spread at the truth
                    for i in range(min(fid_tasks, P)):
                        ti = t.sub(slice(i, i + 1))
                        ev, _, _ = decode(dec, ti, g_fid[None], n_perm=1)
                        ll, _ = R.exact_terms(t, i, g_fid)
                        pe = log_prior(g_fid) + ll
                        pd_ = log_prior(g_fid) + ev[0]
                        pe, pd_ = pe - torch.logsumexp(pe, 0), pd_ - torch.logsumexp(pd_, 0)
                        far = pe < pe.max() - 10
                        imax = int(pe.argmax())
                        over = (ev[0] - ev[0, imax]) - (ll - ll[imax])
                        kl_e = float((pe.exp() * (pe - pd_)).sum())
                        diag_rows += [
                            {"J": J, "n_a": n, "task": i, "metric": "evid_kl", "value": kl_e},
                            {"J": J, "n_a": n, "task": i, "metric": "evid_over95",
                             "value": float(over[far].quantile(0.95)) if far.any() else float("nan")},
                        ]  # fmt: skip
                        _, _, sd = decode(dec, ti, ti.theta[:, None], n_perm=8)
                        diag_rows.append({"J": J, "n_a": n, "task": i, "metric": "perm_std", "value": float(sd[0, 0])})
            else:
                lp_x, ess_x = [], []
                for i in range(P):
                    ti = t.sub(slice(i, i + 1))
                    th, lqm = q.sample_mix(ti.xb, ti.yb, ti.xa, ti.ya, k_exact)
                    rx = R.is_exact(t, i, th[0], lqm[0])
                    lp_x.append(rx["lp"])
                    ess_x.append(rx["ess"])
                put("isexact", torch.stack(lp_x))
                diag("ess_isexact", torch.tensor(ess_x))
            log(f"J={J} n_a={n} done in {time.time() - t0:.0f}s")
    return pd.DataFrame(nll_rows), pd.DataFrame(diag_rows)
