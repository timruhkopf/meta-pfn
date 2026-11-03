"""AIR proposal v2, variant (c): explicit correspondences, a 1-D chain, and a Laplace mixture (issue #21, notebook 14;
docs/a2b_air_proposal_v2_2026-11-03.md). No learned parts; the pilot-3 decoder is only the likelihood in IS.

Per task (d = 1, h = id, J known):
1. B model: a GP on B's 128 noisy points (RBF, constant mean; lengthscale and noise by marginal likelihood on a
   grid) -> predictive mean m(u) and variance v(u). The oracle variant uses the true g (v = 0).
2. Evidence volumes on a grid of G = 256 locations u in B's domain: C_a(u) = B's predictive density of y_a at u, widened
   by σ_A: mean over s ∈ SIG_A_GRID of N(y_a; m(u), v(u) + s²) (σ_A is latent, so its prior range is marginalised).
3. Chain: A sorted by x; p(u_1) and p(u_{a+1} | u_a) from the transform prior (S prior draws of T_θ at the sorted
   x_a, Gaussian kernel of width h on the grid; the joint of consecutive locations is Φ_aᵀ Φ_{a+1}).
   Forward filtering in log space; M coherent correspondence sets by backward sampling (jittered within a cell).
4. Fit per set: Gauss-Newton (Levenberg-Marquardt, finite-difference Jacobian of the flow) on
   ½ Σ_a ((T_θ(x_a) - u_a)/τ)² + ½ |θ|², then on the evidence objective ½ Σ_a ((y_a - m(T_θ(x_a)))/s_a)² + ½ |θ|²;
   Laplace covariance (I + JᵀJ)⁻¹ at the optimum (the proposal's (Λ + F_m)⁻¹).
5. q(θ) = (1 - ε) (1/M) Σ_m N(θ; θ̂_m, Σ_m) + ε N(0, I). Gauss-Newton builds q (no moves after drawing), so its
   density stays exact.
Variant `flow_gn`: the same step 4-5 with M draws of pilot 3's flow as seeds (the flow proposes, GN + Laplace sharpen).
"""

import math

import torch

from .prior import Tasks, flow, log_prior

# ---- 1. B model -----------------------------------------------------------------------------------------------------

ELL_GRID = (0.015, 0.03, 0.06, 0.12, 0.25)
SN_GRID = (0.005, 0.01, 0.02, 0.04)


class GPB:
    """Per-task GP on B (batched); hyperparameters by marginal likelihood on ELL_GRID × SN_GRID."""

    def __init__(self, xb: torch.Tensor, yb: torch.Tensor):
        P, N = xb.shape
        dev, dt = xb.device, torch.float64
        x, y = xb.to(dt), yb.to(dt)
        self.mu = y.mean(1, keepdim=True)
        self.sf2 = y.var(1, keepdim=True).clamp_min(1e-4)
        r = y - self.mu
        d2 = (x[:, :, None] - x[:, None, :]).square()
        best = torch.full((P,), -float("inf"), device=dev, dtype=dt)
        ell = torch.zeros(P, device=dev, dtype=dt)
        sn = torch.zeros(P, device=dev, dtype=dt)
        eye = torch.eye(N, device=dev, dtype=dt)
        for e in ELL_GRID:
            Kx = self.sf2[..., None] * torch.exp(-0.5 * d2 / e**2)
            for s in SN_GRID:
                L, info = torch.linalg.cholesky_ex(Kx + (s**2 + 1e-8) * eye)
                a = torch.cholesky_solve(r[..., None], L)[..., 0]
                lml = -0.5 * (r * a).sum(1) - L.diagonal(dim1=-2, dim2=-1).log().sum(1)
                lml = torch.where(info == 0, lml, torch.full_like(lml, -float("inf")))
                better = lml > best
                best, ell, sn = torch.where(better, lml, best), torch.where(better, e, ell), torch.where(better, s, sn)
        self.ell, self.sn, self.x = ell, sn, x
        Kx = self.sf2[..., None] * torch.exp(-0.5 * d2 / ell[:, None, None] ** 2)
        self.L = torch.linalg.cholesky(Kx + (sn[:, None, None] ** 2 + 1e-8) * eye)
        self.alpha = torch.cholesky_solve(r[..., None], self.L)[..., 0]

    def __call__(self, u: torch.Tensor, idx=None):
        """u [P', M] (tasks idx, default all) -> (mean, variance) [P', M], float64, differentiable in u."""
        sl = slice(None) if idx is None else idx
        x, ell, sf2 = self.x[sl], self.ell[sl], self.sf2[sl]
        k = sf2[..., None] * torch.exp(
            -0.5 * (u.to(x.dtype)[..., None] - x[:, None, :]).square() / ell[:, None, None] ** 2
        )
        m = self.mu[sl] + (k * self.alpha[sl][:, None, :]).sum(-1)
        v = torch.linalg.solve_triangular(self.L[sl], k.transpose(1, 2), upper=False)
        var = (sf2 - v.square().sum(1)).clamp_min(1e-10)
        return m, var


class OracleB:
    """The true g (dense-B limit): variance 0."""

    def __init__(self, t: Tasks):
        self.g = t.g

    def __call__(self, u, idx=None):
        g = self.g if idx is None else self.g.at(idx)
        m = g(u.float()).to(torch.float64)
        return m, torch.zeros_like(m)


# ---- 2.-3. evidence volumes and the chain ---------------------------------------------------------------------------

# σ_A is latent (U[0.005, 0.03] in the AIR prior; neither the flow nor the decoder is given it): the emission is
# marginalised over the midpoints of 4 equal bins of that range. The fits use one value, SIG_FIT ≈ its RMS.
SIG_A_GRID = (0.008125, 0.014375, 0.020625, 0.026875)
SIG_FIT = 0.02


def _log_emission(ys: torch.Tensor, m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """log C_a(u) = log mean_s N(y_a; m(u), v(u) + s²) over s in SIG_A_GRID; ys [P, n], m, v [P, G] -> [P, n, G]."""
    d2 = (ys[:, :, None] - m[:, None, :]).square()
    lc = [-0.5 * (d2 / (s**2 + v[:, None, :]) + torch.log(2 * math.pi * (s**2 + v[:, None, :]))) for s in SIG_A_GRID]
    return torch.logsumexp(torch.stack(lc), 0) - math.log(len(SIG_A_GRID))


def chain_sample(model, t: Tasks, idx: torch.Tensor, M: int, g: torch.Generator, *, G: int = 256, S: int = 4096,
                 h: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Tasks idx [P'] -> (M correspondence sets u [P', M, n] in the order of A's points, order [P', n])."""
    dev = t.xa.device
    xa, ya = t.xa[idx].double(), t.ya[idx].double()
    P, n = xa.shape
    order = xa.argsort(1)
    xs, ys = xa.gather(1, order), ya.gather(1, order)
    ug = (torch.arange(G, device=dev, dtype=torch.float64) + 0.5) / G
    m, v = model(ug.expand(P, -1), idx)
    logc = _log_emission(ys, m, v)  # [P, n, G]
    th = torch.randn(S, t.theta.shape[1], generator=g, device=dev)
    T = flow(xs.float()[:, None, :].expand(P, S, n), th[None].expand(P, S, -1), t.s_t).double()  # [P, S, n]
    phi = torch.exp(-0.5 * (ug - T[..., None]).square() / h**2)  # [P, S, n, G]
    phi = phi / phi.sum(-1, keepdim=True).clamp_min(1e-300)
    logp1 = torch.log(phi[:, :, 0].mean(1).clamp_min(1e-300))  # [P, G]
    logtr = []
    for a in range(n - 1):
        H = torch.einsum("psi,psj->pij", phi[:, :, a], phi[:, :, a + 1]) / S
        logtr.append(torch.log((H / H.sum(-1, keepdim=True).clamp_min(1e-300)).clamp_min(1e-300)))
    alpha = [logp1 + logc[:, 0]]
    for a in range(n - 1):
        alpha.append(torch.logsumexp(alpha[-1][:, :, None] + logtr[a], 1) + logc[:, a + 1])
    u = torch.zeros(P, M, n, device=dev, dtype=torch.long)
    gen = lambda lg: torch.multinomial(lg.softmax(-1).float(), M, replacement=True, generator=g)  # noqa: E731
    u[:, :, n - 1] = gen(alpha[-1])
    for a in range(n - 2, -1, -1):
        nxt = u[:, :, a + 1]  # [P, M]
        lg = alpha[a][:, None, :] + logtr[a].transpose(1, 2).gather(1, nxt[..., None].expand(-1, -1, G))
        u[:, :, a] = torch.multinomial(lg.reshape(P * M, G).softmax(-1).float(), 1, generator=g).view(P, M)
    jit = torch.rand(P, M, n, generator=g, device=dev, dtype=torch.float64)
    us = (u.double() + jit) / G  # sorted order
    out = torch.empty_like(us)
    out.scatter_(2, order[:, None, :].expand(-1, M, -1), us)  # back to A's order
    return out, order


# ---- 4. fits and Laplace --------------------------------------------------------------------------------------------


def _jac(fun, th: torch.Tensor, delta: float = 1e-3):
    """fun: θ [B, J] -> r [B, n]; forward differences -> (r [B, n], Jacobian [B, n, J])."""
    B, J = th.shape
    th_all = torch.cat([th[:, None], th[:, None] + delta * torch.eye(J, device=th.device, dtype=th.dtype)], 1)
    r_all = fun(th_all.reshape(B * (J + 1), J)).reshape(B, J + 1, -1)
    return r_all[:, 0], ((r_all[:, 1:] - r_all[:, :1]) / delta).transpose(1, 2)


def _lm(fun, th: torch.Tensor, iters: int, lam0: float = 1e-2):
    """Levenberg-Marquardt on ½|r(θ)|² + ½|θ|² (batched) -> (θ, J at θ)."""
    lam = torch.full(th.shape[:1], lam0, device=th.device, dtype=th.dtype)
    eye = torch.eye(th.shape[1], device=th.device, dtype=th.dtype)
    r, Jm = _jac(fun, th)
    cost = 0.5 * (r.square().sum(1) + th.square().sum(1))
    for _ in range(iters):
        A = Jm.transpose(1, 2) @ Jm + eye
        grad = (Jm.transpose(1, 2) @ r[..., None])[..., 0] + th
        step = torch.linalg.solve(A + lam[:, None, None] * eye, -grad[..., None])[..., 0]
        cand = th + step
        r_c = fun(cand)
        c_c = 0.5 * (r_c.square().sum(1) + cand.square().sum(1))
        ok = c_c < cost
        th = torch.where(ok[:, None], cand, th)
        cost = torch.where(ok, c_c, cost)
        lam = torch.where(ok, lam / 3, lam * 4).clamp(1e-6, 1e6)
        r, Jm = _jac(fun, th)
    return th, Jm


def fit_evidence(model, t: Tasks, idx: torch.Tensor, th0: torch.Tensor, *, sig_e: float = SIG_FIT,
                 iters: int = 12) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Gauss-Newton (LM) from seeds θ0 [P', M, J] on ½ Σ_a ((y_a - m(T_θ(x_a)))/s_a)² + ½ |θ|²,
    s_a² = σ_e² + v(T_θ(x_a)) -> modes [P', M, J] and Laplace covariances (I + JᵀJ)⁻¹ [P', M, J, J]."""
    P, M, J = th0.shape
    n = t.xa.shape[1]
    xa = t.xa[idx].double()[:, None].expand(P, M, n).reshape(P * M, n)
    ya = t.ya[idx].double()[:, None].expand(P, M, n).reshape(P * M, n)

    def r_ev(th):
        B = th.shape[0]
        x_t = flow(_tile(xa, B, P * M), th, t.s_t)
        mm, vv = model(x_t.reshape(P, -1), idx)
        mm, vv = mm.reshape(B, n), vv.reshape(B, n)
        return (_tile(ya, B, P * M) - mm) / torch.sqrt(sig_e**2 + vv)

    th, Jm = _lm(r_ev, th0.reshape(P * M, J).double(), iters)
    prec = Jm.transpose(1, 2) @ Jm + torch.eye(J, device=th.device, dtype=torch.float64)
    cov = torch.linalg.inv(prec)
    cov = 0.5 * (cov + cov.transpose(1, 2))
    return th.reshape(P, M, J), cov.reshape(P, M, J, J)


def _tile(z, B, rows):
    """Rows are task-major; _jac evaluates J + 1 consecutive variants per row."""
    return z.repeat_interleave(B // rows, 0)


def laplace_mixture(model, t: Tasks, idx: torch.Tensor, u: torch.Tensor, *, tau: float = 0.03, sig_e: float = SIG_FIT,
                    iters: int = 12) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Correspondence sets u [P', M, n] -> fit to the correspondences (seeds) -> fit_evidence."""
    P, M, n = u.shape
    J = t.theta.shape[1]
    xa = t.xa[idx].double()[:, None].expand(P, M, n).reshape(P * M, n)
    uu = u.reshape(P * M, n)

    def r_corr(th):
        B = th.shape[0]
        return (flow(_tile(xa, B, P * M), th, t.s_t) - _tile(uu, B, P * M)) / tau

    th0 = torch.zeros(P * M, J, device=u.device, dtype=torch.float64)
    th, _ = _lm(r_corr, th0, iters)
    return fit_evidence(model, t, idx, th.reshape(P, M, J), sig_e=sig_e, iters=iters)


# ---- 5. the mixture -------------------------------------------------------------------------------------------------


def mix_logq(th: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor, eps: float) -> torch.Tensor:
    """th [P, K, J]; mu [P, M, J]; cov [P, M, J, J] -> log q(θ) [P, K] (float32 in, float64 inside)."""
    L = torch.linalg.cholesky(cov)
    d = th.double()[:, None] - mu[:, :, None]  # [P, M, K, J]
    z = torch.linalg.solve_triangular(L[:, :, None].expand(-1, -1, th.shape[1], -1, -1), d[..., None],
                                      upper=False)[..., 0]  # fmt: skip
    J = th.shape[-1]
    lg = (
        -0.5 * z.square().sum(-1)
        - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)[..., None]
        - 0.5 * J * math.log(2 * math.pi)
    )
    lq = torch.logsumexp(lg, 1) - math.log(mu.shape[1])
    return torch.logaddexp(lq + math.log1p(-eps), log_prior(th.double()) + math.log(eps)).float()


def mix_sample(mu, cov, K: int, eps: float, g: torch.Generator) -> torch.Tensor:
    """K draws [P, K, J] from (1 - ε) mixture + ε N(0, I); the prior share is round(εK) (deterministic split)."""
    P, M, J = mu.shape
    n0 = round(eps * K)
    comp = torch.randint(0, M, (P, K - n0), generator=g, device=mu.device)
    L = torch.linalg.cholesky(cov)
    Lc = L.gather(1, comp[..., None, None].expand(-1, -1, J, J))
    mc = mu.gather(1, comp[..., None].expand(-1, -1, J))
    z = torch.randn(P, K - n0, J, 1, generator=g, device=mu.device, dtype=torch.float64)
    th = mc + (Lc @ z)[..., 0]
    th0 = torch.randn(P, n0, J, generator=g, device=mu.device, dtype=torch.float64)
    return torch.cat([th, th0], 1).float()


def build(model, t: Tasks, M: int, g: torch.Generator, chunk: int = 8, **kw):
    """All tasks -> (mu [P, M, J], cov [P, M, J, J]) for B model `model` (GPB or OracleB)."""
    mus, covs = [], []
    ck = {k: v for k, v in kw.items() if k in ("G", "S", "h")}
    lk = {k: v for k, v in kw.items() if k in ("tau", "sig_e", "iters")}
    for c in range(0, t.xa.shape[0], chunk):
        idx = torch.arange(c, min(c + chunk, t.xa.shape[0]), device=t.xa.device)
        u, _ = chain_sample(model, t, idx, M, g, **ck)
        mu, cov = laplace_mixture(model, t, idx, u, **lk)
        mus.append(mu)
        covs.append(cov)
    return torch.cat(mus), torch.cat(covs)


def build_from_seeds(model, t: Tasks, seeds: torch.Tensor, chunk: int = 8, **kw):
    """Seeds θ0 [P, M, J] (e.g. M flow draws) -> Gauss-Newton + Laplace mixture (mu, cov), as in build."""
    mus, covs = [], []
    for c in range(0, t.xa.shape[0], chunk):
        idx = torch.arange(c, min(c + chunk, t.xa.shape[0]), device=t.xa.device)
        mu, cov = fit_evidence(model, t, idx, seeds[idx], **kw)
        mus.append(mu)
        covs.append(cov)
    return torch.cat(mus), torch.cat(covs)


def n_branches(t: Tasks, n_grid: int = 512) -> torch.Tensor:
    """Monotone pieces of g on [0, 1] per task (1 + sign changes of its slope)."""
    u = torch.linspace(0, 1, n_grid, device=t.xa.device).expand(t.xa.shape[0], -1)
    d = t.g(u).diff(dim=1)
    s = torch.sign(d)
    s = torch.where(d.abs() < 1e-6, torch.zeros_like(s), s)
    ch = ((s[:, 1:] * s[:, :-1]) < 0).sum(1)
    return 1 + ch


# ---- evaluation (notebook 14) ---------------------------------------------------------------------------------------


PROPOSALS = ("flow", "flow_gn", "chain_gp", "chain_oracle")


@torch.no_grad()
def evaluate_chain(dec, props: dict, device, *, n_tasks: int, seed: int, g_norm: str, ks=(8, 32, 128), M: int = 128,
                   eps: float = 0.1, n_rank: int = 255, kl_tasks: int = 64, chunk: int = 16, log=print):  # fmt: skip
    """Proposals (a) `flow` (pilot 3's q at Σ = 0), `flow_gn` (flow seeds + GN + Laplace), (c) `chain_gp` and the
    oracle-emission `chain_oracle`, at equal K,
    with the pilot-3 decoder as the likelihood (Σ = 0). -> (nll, diag, task_info) long tables.
    nll: J, n_a, task, method (`{proposal}_K{K}`, `grid` at J = 2), nll. diag: J, n_a, task, proposal, metric, value
    with metric ∈ {ess_K{K}, logq_true, rank_true (share of q draws with higher q density than θ_true; uniform if q is
    calibrated), kl_post_q (J = 2, grid posterior given g; first kl_tasks tasks), build_s}."""
    import time

    import pandas as pd

    from . import reference as R
    from .estimator import importance
    from .train import eval_tasks
    from .train_rb import N_A_RB

    dec.eval()
    nll, diag, info = [], [], []
    gr = R.grid(101, device=device)
    cell = (8.0 / 100) ** 2
    g = torch.Generator(device=device).manual_seed(seed + 4242)
    for J, q in props.items():
        q.eval()
        base = eval_tasks(J, n_tasks, seed, device, g_norm=g_norm)
        P = base.xa.shape[0]
        info += [{"J": J, "task": i, "branches": int(b)} for i, b in enumerate(n_branches(base))]
        gp = GPB(base.xb, base.yb)
        info += [{"J": J, "task": i, "gp_ell": float(gp.ell[i]), "gp_sn": float(gp.sn[i])} for i in range(P)]
        for n in N_A_RB:
            t = base.head(n)
            if J == 2:
                lp = torch.stack([R.grid_ref(t, i, gr)["lp"] for i in range(P)])
                nll += [{"J": J, "n_a": n, "task": i, "method": "grid", "nll": -float(lp[i].mean())} for i in range(P)]
            comps = {}
            for name, mdl in (("chain_gp", gp), ("chain_oracle", OracleB(t))):
                t0 = time.time()
                comps[name] = build(mdl, t, M, g)
                diag += [{"J": J, "n_a": n, "task": i, "proposal": name, "metric": "build_s",
                          "value": (time.time() - t0) / P} for i in range(P)]  # fmt: skip
            # flow seeds -> Gauss-Newton -> Laplace (the flow proposes, GN sharpens; no ε in the seeds)
            t0 = time.time()
            seeds = torch.cat([q.dist(*(getattr(t.sub(slice(c, c + chunk)), k) for k in ("xb", "yb", "xa", "ya")), 0.0)
                               .sample((M,)).transpose(0, 1) for c in range(0, P, chunk)])  # fmt: skip
            comps["flow_gn"] = build_from_seeds(gp, t, seeds)
            diag += [{"J": J, "n_a": n, "task": i, "proposal": "flow_gn", "metric": "build_s",
                      "value": (time.time() - t0) / P} for i in range(P)]  # fmt: skip
            for c in range(0, P, chunk):
                sl = slice(c, min(c + chunk, P))
                sub = t.sub(sl)
                d = q.dist(sub.xb, sub.yb, sub.xa, sub.ya, 0.0)

                def draw(name, K, d=d, sl=sl, sub=sub, q=q, comps=comps):
                    if name == "flow":
                        return q.sample_mix(sub.xb, sub.yb, sub.xa, sub.ya, K, eps=eps, sig=0.0)
                    mu, cov = comps[name][0][sl], comps[name][1][sl]
                    th = mix_sample(mu, cov, K, eps, g)
                    return th, mix_logq(th, mu, cov, eps)

                def logq(name, th, d=d, sl=sl, q=q, comps=comps):
                    if name == "flow":
                        return q.log_q_mix(d, th, eps, 0.0)
                    return mix_logq(th, comps[name][0][sl], comps[name][1][sl], eps)

                for name in PROPOSALS:
                    for K in ks:
                        th, lq = draw(name, K)
                        r = importance(dec, sub, th, lq, sig=0.0)
                        for k in range(sub.xa.shape[0]):
                            nll.append({"J": J, "n_a": n, "task": c + k, "method": f"{name}_K{K}",
                                        "nll": -float(r["lp1"][k].mean())})  # fmt: skip
                            diag.append({"J": J, "n_a": n, "task": c + k, "proposal": name, "metric": f"ess_K{K}",
                                         "value": float(r["ess"][k])})  # fmt: skip
                    tht = sub.theta[:, None]
                    lqt = logq(name, tht)[:, 0]
                    th, _ = draw(name, n_rank)
                    rank = (logq(name, th) > lqt[:, None]).float().mean(1)
                    for k in range(sub.xa.shape[0]):
                        diag += [{"J": J, "n_a": n, "task": c + k, "proposal": name, "metric": "logq_true",
                                  "value": float(lqt[k])},
                                 {"J": J, "n_a": n, "task": c + k, "proposal": name, "metric": "rank_true",
                                  "value": float(rank[k])}]  # fmt: skip
            if J == 2:
                for i in range(min(kl_tasks, P)):
                    post = R.grid_ref(t, i, gr)["logpost"]
                    ti = t.sub(slice(i, i + 1))
                    for name in PROPOSALS:
                        if name == "flow":
                            dd = q.dist(ti.xb, ti.yb, ti.xa, ti.ya, 0.0)
                            lq = q.log_q_mix(dd, gr[None], eps, 0.0)[0]
                        else:
                            lq = mix_logq(gr[None], comps[name][0][i : i + 1], comps[name][1][i : i + 1], eps)[0]
                        kl = float((post.exp() * (post - (lq + math.log(cell)))).sum())
                        diag.append({"J": J, "n_a": n, "task": i, "proposal": name, "metric": "kl_post_q",
                                     "value": kl})  # fmt: skip
            log(f"chain eval J={J} n_a={n} done")
    return pd.DataFrame(nll), pd.DataFrame(diag), pd.DataFrame(info)
