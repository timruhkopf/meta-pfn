"""MTPFN on the AIR prior: diagnostics (notebook 13). All on pilot 3's evaluation tasks (eval_tasks(J, n, 777, ecdf)).

Tables:
- `ctrl` (J, n_a, task, input, nll): the same tasks with B's input changed (plus references, see air_diagnose):
  `b` (as trained: B in its own frame), `b_registered` (B pulled into A's frame with the true θ, x_b -> T_θ⁻¹(x_b):
  a pre-registered B; if this is much better, the MTPFN's loss is the warp), `b_other` (B of another task: an
  unrelated source), `b_none` (B with no points: only its [TASK] token; off-distribution, ≈ A-only);
- `task_info` (J, task, shift): mean |T_θ(x) − x| over x ∈ [0, 1] (the size of the warp);
- `pit` (J, n_a, task, q, pit): PIT of y* under the MTPFN PPD (bar CDF);
- `sens` (J, n_a, task, xq, read, true_t, conc): where in B the prediction at x* reads: the sensitivity
  s_k = |∂E[y*|x*]/∂y_b,k| over B's points; read = Σ s_k x_b,k / Σ s_k; true_t = T_θ(x*); conc = the share of
  Σ s_k within ±0.05 of true_t;
- `sensmap` (J, n_a, task, xq, xb, s): full sensitivity maps for a few tasks (plots);
- `ex` (J, n_a, task, x, kind, value): example PPDs on an x grid: MTPFN mean and 5/25/50/75/95 % quantiles, the
  true f_A, and for J = 2 the G-ref predictive mean and 5/95 % quantiles; `ex_pts` (J, n_a, task, set, x, y).
"""

import torch

from ..air import reference as R
from ..air.prior import flow
from ..air.train import eval_tasks
from ..air.train_rb import N_A_RB
from .model import MTPFN
from .train import air_batch, nll_per_task


def _with_b(t, xb, yb):
    from dataclasses import replace

    return replace(t, xb=xb, yb=yb)


def _quantiles(model: MTPFN, out: torch.Tensor, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> torch.Tensor:
    return torch.stack([model.bar_dist.icdf(out, q) for q in qs], -1)


def _cdf(model: MTPFN, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bd = model.bar_dist
    p = out.softmax(-1)
    lo, hi = bd.borders[:-1], bd.borders[1:]
    frac = ((y[..., None] - lo) / (hi - lo)).clamp(0, 1)
    return (p * frac).sum(-1)


def air_diagnose(model: MTPFN, cfg, device, aonly=None, log=print) -> dict:
    """aonly: pilot 3's clean A-only PFN (an AIR Decoder); adds ctrl rows input = `aonly_pfn`; for J = 2 also
    input = `gref` (G-ref, the exact dense-B Bayes predictive)."""
    from ..air.estimator import null_branch

    model.eval()
    g101 = R.grid(101, device=device)
    rows = {k: [] for k in ("ctrl", "task_info", "pit", "sens", "sensmap", "ex", "ex_pts")}
    xg = torch.linspace(0, 1, 101, device=device)
    for J in cfg.eval_j:
        base = eval_tasks(J, cfg.n_eval, cfg.eval_seed, device, g_norm=cfg.g_norm)
        P = base.xa.shape[0]
        sh = (flow(xg.expand(P, -1), base.theta, base.s_t) - xg).abs().mean(1)
        rows["task_info"] += [{"J": J, "task": i, "shift": float(v)} for i, v in enumerate(sh)]
        reg = _with_b(base, flow(base.xb, base.theta, base.s_t, inverse=True), base.yb)
        perm = torch.roll(torch.arange(P, device=device), 1)
        other = _with_b(base, base.xb[perm], base.yb[perm])
        for n in N_A_RB:
            t = base.head(n)
            with torch.no_grad():
                for name, tt in (("b", t), ("b_registered", reg.head(n)), ("b_other", other.head(n))):
                    v = nll_per_task(model, air_batch(tt))
                    rows["ctrl"] += [{"J": J, "n_a": n, "task": i, "input": name, "nll": float(x)}
                                     for i, x in enumerate(v)]  # fmt: skip
                extra = []
                if aonly is not None:
                    aonly.eval()
                    extra.append(("aonly_pfn", -null_branch(aonly, t)[1].mean(1)))
                if J == 2:
                    extra.append(("gref", -torch.stack([R.grid_ref(t, i, g101)["lp"] for i in range(P)]).mean(1)))
                for name, v in extra:
                    rows["ctrl"] += [{"J": J, "n_a": n, "task": i, "input": name, "nll": float(z)}
                                     for i, z in enumerate(v)]  # fmt: skip
                x, y, m, xq, yq, nf = air_batch(t)
                m0 = m.clone()
                m0[:, 1] = False
                v = nll_per_task(model, (x, y, m0, xq, yq, nf))
                rows["ctrl"] += [{"J": J, "n_a": n, "task": i, "input": "b_none", "nll": float(z)}
                                 for i, z in enumerate(v)]  # fmt: skip
                out = torch.cat([model(x[c : c + 64], y[c : c + 64], m[c : c + 64], xq[c : c + 64], nf[c : c + 64])
                                 for c in range(0, P, 64)])  # fmt: skip
                pit = _cdf(model, out, yq)
                rows["pit"] += [{"J": J, "n_a": n, "task": i, "q": k, "pit": float(pit[i, k])}
                                for i in range(min(P, 128)) for k in range(pit.shape[1])]  # fmt: skip
            # sensitivity: where in B the predictive mean at x* reads
            ns = min(cfg.n_sens, P)
            tt = t.sub(slice(0, ns))
            x, y, m, _, _, nf = air_batch(tt)
            xq = xg.expand(ns, -1)[..., None]
            yb = y[:, 1].clone().requires_grad_(True)

            def mean_at(yb_, x=x, y=y, m=m, xq=xq, nf=nf):
                y2 = torch.cat([y[:, :1], yb_[:, None]], 1)
                return model.bar_dist.mean(model(x, y2, m, xq, nf)).sum(0)  # [101]; tasks are independent

            jac = torch.autograd.functional.jacobian(mean_at, yb).abs()  # [101, ns, N]
            s = jac.permute(1, 0, 2)[..., : tt.xb.shape[1]]  # [ns, 101, n_b]
            xb = tt.xb
            read = (s * xb[:, None]).sum(-1) / s.sum(-1).clamp_min(1e-12)
            tt_true = flow(xg.expand(ns, -1), tt.theta, tt.s_t)
            near = ((xb[:, None] - tt_true[..., None]).abs() <= 0.05).float()
            conc = (s * near).sum(-1) / s.sum(-1).clamp_min(1e-12)
            for i in range(ns):
                for k in range(0, 101, 5):
                    rows["sens"].append({"J": J, "n_a": n, "task": i, "xq": float(xg[k]), "read": float(read[i, k]),
                                         "true_t": float(tt_true[i, k]), "conc": float(conc[i, k])})  # fmt: skip
            if n in (1, 3, 10):
                for i in range(min(4, ns)):
                    order = xb[i].argsort()
                    for k in range(0, 101, 2):
                        rows["sensmap"] += [{"J": J, "n_a": n, "task": i, "xq": float(xg[k]), "xb": float(xb[i, o]),
                                             "s": float(s[i, k, o])} for o in order.tolist()]  # fmt: skip
                # example PPDs
                with torch.no_grad():
                    ne = min(6, P)
                    te = t.sub(slice(0, ne))
                    x, y, m, _, _, nf = air_batch(te)
                    out = model(x, y, m, xg.expand(ne, -1)[..., None], nf)
                    mu, qq = model.bar_dist.mean(out), _quantiles(model, out)
                    fa = te.ga(flow(xg.expand(ne, -1), te.theta, te.s_t))
                    for i in range(ne):
                        for k in range(101):
                            base_row = {"J": J, "n_a": n, "task": i, "x": float(xg[k])}
                            rows["ex"] += [base_row | {"kind": kd, "value": float(v)} for kd, v in
                                           [("mean", mu[i, k]), ("f_a", fa[i, k])] +
                                           [(f"q{int(100 * q)}", qq[i, k, j]) for j, q in
                                            enumerate((0.05, 0.25, 0.5, 0.75, 0.95))]]  # fmt: skip
                        pts = [("A", te.xa[i], te.ya[i]), ("B", te.xb[i], te.yb[i])]
                        for nm, xs, ys in pts:
                            rows["ex_pts"] += [{"J": J, "n_a": n, "task": i, "set": nm, "x": float(a), "y": float(b)}
                                               for a, b in zip(xs, ys, strict=True)]  # fmt: skip
                    if J == 2:  # G-ref predictive on the x grid (posterior samples of θ; noise σ_A)
                        for i in range(ne):
                            post = R.grid_ref(te, i, g101)["logpost"].exp()
                            idx = torch.multinomial(post, 2000, replacement=True)
                            th = g101[idx]
                            gi = te.g.at(torch.full((2000,), i, device=device, dtype=torch.long))
                            f = gi(flow(xg.expand(2000, -1), th, te.s_t))
                            f = f + te.sig_a[i] * torch.randn_like(f)
                            qs = torch.quantile(f, torch.tensor([0.05, 0.95], device=device), dim=0)
                            for k in range(101):
                                b_ = {"J": J, "n_a": n, "task": i, "x": float(xg[k])}
                                rows["ex"] += [b_ | {"kind": "gref_mean", "value": float(f[:, k].mean())},
                                               b_ | {"kind": "gref_q5", "value": float(qs[0, k])},
                                               b_ | {"kind": "gref_q95", "value": float(qs[1, k])}]  # fmt: skip
            log(f"diag J={J} n_a={n} done")
    import pandas as pd

    return {k: pd.DataFrame(v) for k, v in rows.items()}
