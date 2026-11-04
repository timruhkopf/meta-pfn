"""CIS: claims (§3.3), FFBS (§3.4), fits (§3.5), sampling and weights (§3.6), rounds with the U update (§3.7).

`run(tasks, surrogate, target, spec, R, K)` returns, per task, the weighted predictive at the queries and the
diagnostics. The target supplies log p̂(A | B, θ) and the predictive given θ: `DecoderTarget` (the transported
decoder D) or `ExactTarget` (the true g and σ_A, h integrated; dense-B toys, test 5, and the oracle's likelihood).

Validity (spec §3.6): the proposal q(c, J, θ) = q(c) q(J | c) q_mix(θ | c, J) is an exact density, positive wherever
the target is (soft monotone FFBS, q(J | c) ≥ 0.02, an ε = 0.1 prior component); the auxiliary target
p(J, θ | A, B) r(c | θ) keeps the (J, θ) marginal exact, as r(c | θ) = Π_a softmax_{u ∈ C_a}(−(u − T_θ(x_a))²/(2τ_a²))
is a normalised distribution over the candidate product set for every θ (τ_a per point, not per assignment:
deviation). Earlier rounds only shape the proposal of the last; their weights never enter the estimate."""

import math
import time
from dataclasses import dataclass

import torch

from ...mtpfn import hd
from ..decoder import N_BINS, Y_HI, Y_LO, bar_logpdf
from ..reference import bucket_lp
from . import ffbs as F
from . import fit as FT
from . import prior as C

M_TOP, M_PRIOR, L_LOCAL = 16, 4, 8
EPS_MIX, J_FLOOR = 0.1, 0.02
H_GRID = 16
DEBUG: list = []  # set to [None] to record the weight terms of every round


# ---- bucket helpers --------------------------------------------------------------------------------------------------


def bucket_cdf(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Predictive CDF at y under bucket probabilities probs [..., G] (uniform within a bucket)."""
    w = (Y_HI - Y_LO) / N_BINS
    z = ((y - Y_LO) / w).clamp(0, N_BINS - 1e-6)
    idx = z.floor().long()
    cum = torch.cumsum(probs, -1)
    below = torch.where(idx > 0, cum.gather(-1, (idx - 1).clamp_min(0)[..., None])[..., 0], torch.zeros_like(z))
    return below + probs.gather(-1, idx[..., None])[..., 0] * (z - idx)


def gauss_bucket_probs(mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """Bucket probabilities of N(mu, sig²): mu [...], sig broadcastable -> [..., G]."""
    b = torch.linspace(Y_LO, Y_HI, N_BINS + 1, device=mu.device, dtype=mu.dtype)
    c = torch.special.ndtr((b - mu[..., None]) / sig[..., None])
    p = (c[..., 1:] - c[..., :-1]).clamp_min(0)
    return p / p.sum(-1, keepdim=True).clamp_min(1e-30)


# ---- targets ---------------------------------------------------------------------------------------------------------


class ExactTarget:
    """The true g and σ_A: log p(A | B, θ) with h integrated analytically (id/affine mixture, π_id); the predictive
    given θ draws one h per row from its posterior (the mixture over rows integrates h)."""

    def __init__(self, t: C.CisTasks, pi_id: float, nb: int = 512):
        self.t, self.pi_id, self.bg = t, pi_id, hd.b_grid(nb, t.xa.device)

    def __call__(self, tidx: torch.Tensor, lat: torch.Tensor, J: torch.Tensor, g: torch.Generator, full: bool = False):
        t = self.t
        gr = t.g.at(tidx)
        x = torch.cat([t.xa[tidx], t.xq[tidx]], 1)
        m = gr(C.transform(x, lat, J))
        n = t.xa.shape[1]
        ll = torch.empty(len(tidx), device=lat.device)
        a = torch.zeros(len(tidx), device=lat.device)
        b = torch.ones(len(tidx), device=lat.device)
        for j in tidx.unique():
            sel = (tidx == j).nonzero()[:, 0]
            s = t.sig_a[j]
            if n == 0:
                ll[sel] = 0
                continue
            L_id, lb, rbar, sd = hd.h_terms(t.ya[j], m[sel, :n], s, self.bg)
            ll[sel] = hd.log_marg(L_id, lb, self.pi_id)
            if self.pi_id < 1:
                L_aff = torch.logsumexp(lb, -1) - math.log(lb.shape[-1])
                p_id = torch.sigmoid(math.log(self.pi_id) + L_id - math.log(1 - self.pi_id) - L_aff)
                is_id = torch.rand(len(sel), generator=g, device=lat.device) < p_id
                k = torch.multinomial((lb - torch.logsumexp(lb, -1, keepdim=True)).exp(), 1, generator=g)[:, 0]
                bb = self.bg[k]
                rb = rbar.gather(1, k[:, None])[:, 0]
                lo, hi = torch.special.ndtr(-rb / sd), torch.special.ndtr((1 - bb - rb) / sd)
                u = lo + (hi - lo) * torch.rand(len(sel), generator=g, device=lat.device)
                aa = (rb + sd * torch.special.ndtri(u.clamp(1e-12, 1 - 1e-12))).clamp(0).minimum(1 - bb)
                a[sel] = torch.where(is_id, torch.zeros_like(aa), aa)
                b[sel] = torch.where(is_id, torch.ones_like(bb), bb)
        mu_q = a[:, None] + b[:, None] * m[:, n:]
        sig = t.sig_a[tidx][:, None]
        out = {"loglik": ll, "lp": bucket_lp(mu_q, sig, t.yq[tidx])}
        probs = gauss_bucket_probs(mu_q, sig.expand_as(mu_q))
        out["cdf"] = bucket_cdf(probs, t.yq[tidx])
        if full:
            out["probs"] = probs
        return out


class DecoderTarget:
    """The transported decoder D (cis/decoder.py): evidence Σ_a log D(y_a | x_a, A_<a, B̃_θ) averaged over n_perm
    random orders; the predictive at x* from the first pass (it sees all of A, order-free)."""

    def __init__(self, dec, t: C.CisTasks, n_perm: int = 2, chunk: int = 2048):
        self.dec, self.t, self.n_perm, self.chunk = dec, t, n_perm, chunk

    @torch.no_grad()
    def __call__(self, tidx, lat, J, g, full: bool = False):
        t, out = self.t, {"loglik": [], "lp": [], "cdf": [], "probs": []}
        for s in range(0, len(tidx), self.chunk):
            ti, la, Jc = tidx[s : s + self.chunk], lat[s : s + self.chunk], J[s : s + self.chunk]
            xb_t, ld = C.inverse(t.xb[ti], la, Jc)
            xa, ya, xq, yq = t.xa[ti], t.ya[ti], t.xq[ti], t.yq[ti]
            evs, lq = [], None
            for _ in range(max(self.n_perm, 1)):
                rank = torch.rand(len(ti), xa.shape[1], generator=g, device=xa.device).argsort(-1).argsort(-1)
                lg_q, lg_e = self.dec(xb_t, t.yb[ti], xa, ya, xq, rank=rank, ld=ld)
                evs.append(bar_logpdf(lg_e, ya).sum(-1) if xa.shape[1] else lg_e.new_zeros(len(ti)))
                lq = lg_q if lq is None else lq
            out["loglik"].append(torch.stack(evs, -1).mean(-1))
            out["lp"].append(bar_logpdf(lq, yq))
            pr = lq.float().softmax(-1)
            out["cdf"].append(bucket_cdf(pr, yq))
            if full:
                out["probs"].append(pr)
        return {k: torch.cat(v) for k, v in out.items() if v}


# ---- claims ----------------------------------------------------------------------------------------------------------


@dataclass
class State:
    """After a round: per (task, sorted A point) the weighted transform locations, for the next round's proposal."""

    centers: torch.Tensor  # [P, K, n] T_θk(x_a)
    w: torch.Tensor  # [P, K] normalised weights
    bw: torch.Tensor  # [P, n] KDE bandwidth
    pi_id: torch.Tensor  # [P] weighted P(h = id)
    h_mean: torch.Tensor  # [P, 2] weighted (α, β) of the affine fits
    h_var: torch.Tensor  # [P, 2]


@dataclass
class Claims:
    u: torch.Tensor  # [P, n, C] candidate locations
    mu: torch.Tensor  # [P, n, C] μ_B(u)
    sd2: torch.Tensor  # [P, n, C] σ_B²(u)
    logC: torch.Tensor  # [P, n, C] log claim score (−inf on invalid slots)
    valid: torch.Tensor  # [P, n, C]
    tau: torch.Tensor  # [P, n]


def _h_grid(device):
    b = C.B_LO + (1 - C.B_LO) * (torch.arange(H_GRID, device=device) + 0.5) / H_GRID
    v = (torch.arange(H_GRID, device=device) + 0.5) / H_GRID
    bb, vv = torch.meshgrid(b, v, indexing="ij")
    return (vv * (1 - bb)).flatten(), bb.flatten()  # α, β on the prior grid (uniform weights)


def log_emission(y, mu, sd2, pi_id: float, state: State | None, sig_bar: float = FT.SIG_BAR):
    """log E_a(u): y [P, n, 1], mu, sd2 [P, n, C] -> [P, n, C]; the h mixture (deviation D2)."""

    def ln(v, m, s2):
        return -0.5 * (v - m) ** 2 / s2 - 0.5 * torch.log(2 * math.pi * s2)

    l_id = ln(y, mu, sd2 + sig_bar**2)
    if pi_id >= 1:
        return l_id
    if state is None:  # round 1: the h-marginal on the 16 × 16 prior grid
        a, b = _h_grid(mu.device)
        la = ln(y[..., None], a + b * mu[..., None], b**2 * sd2[..., None] + sig_bar**2)
        l_aff = torch.logsumexp(la, -1) - math.log(a.numel())
        p = torch.tensor(pi_id, device=mu.device)
    else:  # rounds ≥ 2: plug-in weighted h, its spread inflating the variance
        a, b = state.h_mean[:, 0, None, None], state.h_mean[:, 1, None, None]
        va, vb = state.h_var[:, 0, None, None], state.h_var[:, 1, None, None]
        l_aff = ln(y, a + b * mu, b**2 * sd2 + sig_bar**2 + va + mu**2 * vb)
        p = state.pi_id.clamp(0.02, 0.98)[:, None, None]
    return torch.logaddexp(torch.log(p) + l_id, torch.log1p(-p) + l_aff)


def log_loc_prior(xs, u, tab, state: State | None):
    """log π^{(r)}(u | x_a): xs [P, n], u [P, n, C]."""
    lp = C.log_pi(xs[..., None].expand_as(u), u, tab)
    if state is None:
        return lp
    d = (u[:, :, None, :] - state.centers.transpose(1, 2)[..., None]) / state.bw[:, :, None, None]  # [P, n, K, C]
    lw = torch.log(state.w.clamp_min(1e-30))[:, None, :, None]  # [P, 1, K, 1]
    lk = torch.logsumexp(lw - 0.5 * d**2 - torch.log(state.bw)[:, :, None, None] - 0.5 * math.log(2 * math.pi), 2)
    return torch.logaddexp(math.log(0.8) + lk, math.log(0.2) + lp)


def claims(t: C.CisTasks, xs, ys, sur, tab, spec: FT.Spec, state: State | None, g: torch.Generator) -> Claims:
    P, n = xs.shape
    pi_id = spec.ph[0] if "id" in spec.h_types else 0.0
    if spec.h_types == ("id",):
        pi_id = 1.0
    ub = t.xb[:, None].expand(P, n, -1)  # every B point is a candidate location
    mub, sdb = sur.mu(t.xb)[:, None].expand(P, n, -1), sur.var(t.xb)[:, None].expand(P, n, -1)
    lpi = log_loc_prior(xs, ub, tab, state)
    score = log_emission(ys[..., None], mub, sdb, pi_id, state) + lpi
    top = score.topk(min(M_TOP, ub.shape[-1]), -1).indices
    near = lpi.topk(M_PRIOR, -1).indices
    idx = torch.cat([top, near], -1)
    dup = torch.zeros_like(idx, dtype=torch.bool)
    for k in range(M_TOP, idx.shape[-1]):
        dup[..., k] = (idx[..., k, None] == idx[..., :k]).any(-1)
    u, mu, sd2 = ub.gather(-1, idx), mub.gather(-1, idx), sdb.gather(-1, idx)
    logC, valid = score.gather(-1, idx), ~dup
    if state is not None:  # L local synthetic locations from the KDE
        k = torch.multinomial(state.w, L_LOCAL * n, replacement=True, generator=g).reshape(P, L_LOCAL, n)
        c = state.centers.gather(1, k)  # [P, L, n]
        ul = (c + state.bw[:, None] * torch.randn(c.shape, generator=g, device=c.device)).clamp(0, 1).transpose(1, 2)
        mul, sdl = sur.mu(ul), sur.var(ul)
        sl = log_emission(ys[..., None], mul, sdl, pi_id, state) + log_loc_prior(xs, ul, tab, state)
        u, mu, sd2 = torch.cat([u, ul], -1), torch.cat([mu, mul], -1), torch.cat([sd2, sdl], -1)
        logC, valid = torch.cat([logC, sl], -1), torch.cat([valid, torch.ones_like(sl, dtype=torch.bool)], -1)
    logC = logC.masked_fill(~valid, -math.inf)
    dd = (u[..., :, None] - u[..., None, :]).abs()
    big = torch.full_like(dd, 10.0)
    dd = torch.where(valid[..., None, :] & valid[..., :, None], dd, big)
    dd = dd + torch.eye(u.shape[-1], device=u.device) * 10
    nn2 = dd.topk(2, -1, largest=False).values.mean(-1)  # [P, n, C] mean distance to the 2 nearest candidates
    nn2 = torch.where(valid, nn2, torch.nan)
    tau = nn2.nanmedian(-1).values.clamp(0.003, 0.1)
    return Claims(u, mu, sd2, logC, valid, tau)


# ---- one CIS run -----------------------------------------------------------------------------------------------------


def _surrogate_ll(t, rows_t, m, sig_c2, spec: FT.Spec, sig_bar: float = FT.SIG_BAR):
    """Rounds < R: the GP likelihood with h integrated analytically (scalar σ per task)."""
    pi_id = 1.0 if spec.h_types == ("id",) else spec.ph[0]
    bg = hd.b_grid(256, m.device)
    out = torch.empty(m.shape[0], device=m.device)
    for j in rows_t.unique():
        sel = (rows_t == j).nonzero()[:, 0]
        s = torch.sqrt(sig_bar**2 + sig_c2[sel].mean())
        L_id, lb, _, _ = hd.h_terms(t.ys[j], m[sel], s, bg)
        out[sel] = hd.log_marg(L_id, lb, pi_id)
    return out


def dm_log_qbar(P: int, Kc: int, spec: FT.Spec, fits: dict, qJ: torch.Tensor, ji: torch.Tensor,
                lat: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """Deterministic-mixture proposal density of each draw (J_k, θ_k) of a task under all of that task's sampled
    assignments: log (1/K) Σ_k′ q(J_k | c_k′) q_mix(θ_k | c_k′, J_k) -> [B]. Given the drawn assignments, the
    estimator with these weights is unbiased (deviation: replaces the auxiliary r(c | θ), whose variance collapsed
    the ESS: sd of log r 300-1700 nats on the J = 2 toy)."""
    dev = lat.device
    out = torch.empty(P * Kc, device=dev)
    log_ph = torch.log(torch.tensor(spec.ph, device=dev))
    for p in range(P):
        rp = slice(p * Kc, (p + 1) * Kc)
        jp = ji[rp]
        for i, J in enumerate(spec.js):
            sel = (jp == i).nonzero()[:, 0]
            if not len(sel):
                continue
            idx = spec.idx(J)
            D = len(idx)
            X = lat[rp][sel][:, idx] if D else lat.new_zeros(len(sel), 0)
            comps = [fits[J, h] for h in spec.h_types]
            lz = torch.stack([log_ph[k] + c.log_z[rp] for k, c in enumerate(comps)], -1)  # [K', H]
            lwh = lz - torch.logsumexp(lz, -1, keepdim=True)
            terms = []
            for k, c in enumerate(comps):
                if D:
                    diff = (X[None] - c.mean_t[rp][:, None]).transpose(1, 2)  # [K', D, m]
                    z = torch.linalg.solve_triangular(c.chol_t[rp], diff, upper=False)
                    ldet = torch.log(torch.diagonal(c.chol_t[rp], dim1=1, dim2=2)).sum(1)[:, None]
                    lpdf = -0.5 * (z**2).sum(1) - ldet - 0.5 * D * math.log(2 * math.pi)
                else:
                    lpdf = X.new_zeros(Kc, len(sel))
                terms.append(lwh[:, k, None] + lpdf)
            lmix = torch.logsumexp(torch.stack(terms), 0)  # [K', m]
            lpri = -0.5 * (X**2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
            lq = torch.logaddexp(math.log(1 - EPS_MIX) + lmix, math.log(EPS_MIX) + lpri[None])
            lq = lq + torch.log(qJ[rp][:, i])[:, None]
            out[p * Kc + sel] = torch.logsumexp(lq, 0) - math.log(Kc)
    return out


@dataclass
class _Sorted:
    xa: torch.Tensor
    ys: torch.Tensor


def run(t: C.CisTasks, sur, target, spec: FT.Spec, R: int, K: int, g: torch.Generator, Kr: int = 32,
        tab: dict | None = None, full: bool = False, mu_rows=None) -> dict:  # fmt: skip
    """t: P tasks (n_A ≥ 1) on one device; sur: surrogate with mu/var on [P, ...]; mu_rows(tidx) -> callable on
    [B, n] for the fits (default: sur.at(tidx).mu). -> per-task results and diagnostics."""
    tab = tab or C.prior_tables()
    P, n = t.xa.shape
    dev = t.xa.device
    order = t.xa.argsort(1)
    xs, ys = t.xa.gather(1, order), t.ya.gather(1, order)
    dx = (xs[:, 1:] - xs[:, :-1]).clamp_min(1e-4)
    m_k, s_k = C.stretch_stats(dx, tab)
    jv = torch.tensor(spec.js, device=dev)
    log_pj = torch.log(torch.tensor(spec.pj, device=dev))
    state, timing, ess_r = None, {}, []
    st = _Sorted(xs, ys)
    for r in range(1, R + 1):
        Kc = K if r == R else Kr
        t0 = time.time()
        cl = claims(t, xs, ys, sur, tab, spec, state, g)
        logK = F.log_trans(cl.u, cl.valid, dx, m_k, s_k)
        la, _ = F.forward(cl.logC, logK)
        path, lq_c = F.sample(la, logK, Kc, g)  # [P, Kc, n]
        timing[f"r{r}_claims_ffbs"] = time.time() - t0
        t0 = time.time()
        B = P * Kc
        tidx = torch.arange(P, device=dev).repeat_interleave(Kc)
        pth = path.reshape(B, n)
        gat = lambda a: a[tidx].gather(-1, pth[..., None])[..., 0]  # noqa: E731, B023  [B, n] chosen candidate's value
        uc, sc2 = gat(cl.u), gat(cl.sd2)
        tau, xr, yr = cl.tau[tidx], xs[tidx], ys[tidx]
        mu_fn = (
            mu_rows(tidx) if mu_rows is not None else (sur.rows(tidx).mu if hasattr(sur, "rows") else sur.at(tidx).mu)
        )
        fits = {}
        for J in spec.js:
            for h in spec.h_types:
                fits[J, h] = FT.fit(xr, yr, uc, tau, sc2, mu_fn, J, spec, aff=h == "aff", g=g)
        timing[f"r{r}_fit"] = time.time() - t0
        t0 = time.time()
        log_ph = torch.log(torch.tensor(spec.ph, device=dev))
        hs = list(enumerate(spec.h_types))
        lzh = [torch.stack([log_ph[i] + fits[J, h].log_z for i, h in hs]) for J in spec.js]
        lzJ = torch.stack([torch.logsumexp(v, 0) for v in lzh], -1)  # [B, nJ]
        qJ = (log_pj + lzJ).softmax(-1)
        qJ = (1 - J_FLOOR * len(spec.js)) * qJ + J_FLOOR
        ji = torch.multinomial(qJ, 1, generator=g)[:, 0]
        Js = jv[ji]
        log_qJ = torch.log(qJ.gather(1, ji[:, None])[:, 0])
        lat = torch.zeros(B, C.N_LAT, device=dev)
        log_qth = torch.zeros(B, device=dev)
        hsel = torch.zeros(B, dtype=torch.long, device=dev)
        hval = torch.tensor([0.0, 1.0], device=dev).expand(B, 2).clone()
        for i, J in enumerate(spec.js):
            rows = (ji == i).nonzero()[:, 0]
            if not len(rows):
                continue
            idx = spec.idx(J)
            D = len(idx)
            comps = [fits[J, h] for h in spec.h_types]
            lwh = torch.stack([log_ph[k] + comps[k].log_z[rows] for k in range(len(comps))], -1)
            wh = lwh.softmax(-1)  # [b, H]
            kh = torch.multinomial(wh, 1, generator=g)[:, 0]
            from_prior = torch.rand(len(rows), generator=g, device=dev) < EPS_MIX
            z = torch.randn(len(rows), D, generator=g, device=dev)
            mean = torch.stack([c.mean_t[rows] for c in comps], 1).gather(1, kh[:, None, None].expand(-1, 1, D))[:, 0]
            chol = torch.stack([c.chol_t[rows] for c in comps], 1)[torch.arange(len(rows), device=dev), kh]
            th = torch.where(from_prior[:, None], z, mean + torch.einsum("bij,bj->bi", chol, z))
            lcomp = torch.stack([FT.mvn_logpdf(th, c.mean_t[rows], c.chol_t[rows]) for c in comps], -1)
            lmix = torch.logsumexp(torch.log(wh.clamp_min(1e-30)) + lcomp, -1)
            lpri = -0.5 * (th**2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
            log_qth[rows] = torch.logaddexp(math.log(1 - EPS_MIX) + lmix, math.log(EPS_MIX) + lpri)
            if D:
                lat[rows[:, None], torch.tensor(idx, device=dev)[None]] = th
            hsel[rows] = kh
            hval[rows] = torch.stack([c.h[rows] for c in comps], 1)[torch.arange(len(rows), device=dev), kh]
        Tx = C.transform(xr, lat, Js)  # [B, n] on sorted A
        lr = -((cl.u[tidx] - Tx[..., None]) ** 2) / (2 * tau[..., None] ** 2)
        lr = lr.masked_fill(~cl.valid[tidx], -math.inf).log_softmax(-1).gather(-1, pth[..., None])[..., 0].sum(-1)
        lprior = log_pj[ji] + FT.log_prior(lat, Js, spec)
        lq = lq_c.reshape(B) + log_qJ + log_qth
        timing[f"r{r}_sample"] = time.time() - t0
        t0 = time.time()
        if r < R:
            ll = _surrogate_ll(st, tidx, mu_fn(Tx), sc2, spec)
            res = None
        else:
            res = target(tidx, lat, Js, g, full=full)
            ll = res["loglik"]
        timing[f"r{r}_target"] = time.time() - t0
        if DEBUG:
            DEBUG.append({"lprior": lprior, "ll": ll, "lr": lr, "lq_c": lq_c.reshape(B), "log_qJ": log_qJ,
                          "log_qth": log_qth, "tau": cl.tau, "P": P, "K": Kc, "lq": lq})  # fmt: skip
        lq_dm = dm_log_qbar(P, Kc, spec, fits, qJ, ji, lat)
        lw = (lprior + ll - lq_dm).reshape(P, Kc)
        lw = torch.nan_to_num(lw, nan=-math.inf)
        W = lw.softmax(-1)
        ess_r.append(1 / (W**2).sum(-1))
        if r < R:
            cen = Tx.reshape(P, Kc, n)
            mean_c = (W[..., None] * cen).sum(1)
            sd_c = ((W[..., None] * (cen - mean_c[:, None]) ** 2).sum(1)).sqrt()
            is_id = torch.tensor([h == "id" for h in spec.h_types], device=dev)[hsel].reshape(P, Kc)
            pid = (W * is_id).sum(1)
            wa = W * (~is_id)
            wa = wa / wa.sum(1, keepdim=True).clamp_min(1e-12)
            hv = hval.reshape(P, Kc, 2)
            hm = (wa[..., None] * hv).sum(1)
            hvv = (wa[..., None] * (hv - hm[:, None]) ** 2).sum(1)
            hm = torch.where((pid < 0.999)[:, None], hm, FT.H_MEAN.to(dev).expand(P, 2))
            state = State(cen, W, sd_c.clamp_min(0.01), pid, hm, hvv)
    lpq = res["lp"].reshape(P, K, -1)
    out = {
        "lp": torch.logsumexp(torch.log(W.clamp_min(1e-30))[..., None] + lpq, 1),
        "cdf": (W[..., None] * res["cdf"].reshape(P, K, -1)).sum(1),
        "ess": ess_r[-1], "ess_rounds": torch.stack(ess_r, -1), "log_z1": torch.logsumexp(lw, -1) - math.log(K),
        "W": W, "J": Js.reshape(P, K), "lat": lat.reshape(P, K, C.N_LAT), "timing": timing,
    }  # fmt: skip
    if full:
        out["probs"] = (W[..., None, None] * res["probs"].reshape(P, K, *res["probs"].shape[1:])).sum(1)
    return out
