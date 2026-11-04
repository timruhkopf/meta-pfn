"""The exact oracle on the AIR-CIS prior (deviation D4): the posterior predictive given the true g and σ_A.

p(y* | A, B) = Σ_J p(J | A) ∫ p(lat | A, J) p(y* | lat, A) dlat, with h integrated analytically (id/affine mixture,
as notebook 16's oracle), lat the free transform latents of J (N(0, I) prior), sampled by likelihood-tempered SMC
(adaptive temperatures keeping ESS = 0.5 K per reweight, multinomial resampling, random-walk MH moves with the
particles' covariance), vectorised over tasks; p(J | A) ∝ p(J) Ẑ_J from the SMC evidences. Two independent halves
give the MC error. It knows g exactly (the dense-B limit): a bound for learners that see 128 noisy B points."""

import math

import torch

from ...mtpfn import hd
from ..reference import bucket_lp
from . import core as CO
from . import fit as FT
from . import prior as C


def h_terms_b(ya: torch.Tensor, m: torch.Tensor, sig: torch.Tensor, bg: torch.Tensor):
    """hd.h_terms batched over tasks: ya [P, n], m [P, K, n], sig [P], bg [nb] -> (L_id [P, K], lb [P, K, nb],
    rbar [P, K, nb], s [P])."""
    n = ya.shape[-1]
    sg = sig[:, None]
    L_id = (-0.5 * ((ya[:, None] - m) / sg[..., None]) ** 2).sum(-1) - n * (torch.log(sg) + 0.5 * hd.LOG2PI)
    yb, mb = ya.mean(-1), m.mean(-1, keepdim=True)
    yc, mc = ya - yb[:, None], m - mb
    syy = (yc**2).sum(-1)[:, None, None]
    sym = (yc[:, None] * mc).sum(-1, keepdim=True)
    smm = (mc**2).sum(-1, keepdim=True)
    ss = (syy - 2 * bg * sym + bg**2 * smm).clamp_min(0)
    rbar = yb[:, None, None] - bg * mb
    s = sig / math.sqrt(n)
    s3 = s[:, None, None]
    la = 0.5 * hd.LOG2PI + torch.log(s3) + hd.log_phi_diff((1 - bg - rbar) / s3, -rbar / s3) - torch.log(1 - bg)
    lb = -ss / (2 * sig[:, None, None] ** 2) - n * (torch.log(sig) + 0.5 * hd.LOG2PI)[:, None, None] + la
    return L_id, lb, rbar, s


def log_marg_b(L_id, lb, pi_id: float):
    if pi_id >= 1:
        return L_id
    L_aff = torch.logsumexp(lb, -1) - math.log(lb.shape[-1])
    return torch.logaddexp(math.log(pi_id) + L_id, math.log(1 - pi_id) + L_aff)


class Model:
    """Task chunk with the true g: m(lat) = g(T(x)) at A's points (and queries), the h-marginal likelihood."""

    def __init__(self, t: C.CisTasks, J: int, spec: FT.Spec, pi_id: float, nb: int = 512):
        self.t, self.J, self.spec, self.pi_id = t, J, spec, pi_id
        self.idx = spec.idx(J)
        self.bg = hd.b_grid(nb, t.xa.device)

    def lat(self, th: torch.Tensor) -> torch.Tensor:
        P, K, _ = th.shape
        lat = th.new_zeros(P, K, C.N_LAT)
        if self.idx:
            lat[..., self.idx] = th
        return lat

    def m(self, th: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """th [P, K, D], x [P, M] -> g(T(x)) [P, K, M]."""
        P, K, _ = th.shape
        Jt = torch.full((P, K), self.J, device=th.device, dtype=torch.long)
        u = C.transform(x[:, None].expand(P, K, -1), self.lat(th), Jt)
        return self.t.g(u)

    def loglik(self, th: torch.Tensor) -> torch.Tensor:
        t = self.t
        if t.xa.shape[1] == 0:
            return th.new_zeros(th.shape[:2])
        L_id, lb, _, _ = h_terms_b(t.ya, self.m(th, t.xa), t.sig_a, self.bg)
        return torch.nan_to_num(log_marg_b(L_id, lb, self.pi_id), nan=-math.inf)


def _log_ess(lw: torch.Tensor) -> torch.Tensor:
    return 2 * torch.logsumexp(lw, -1) - torch.logsumexp(2 * lw, -1)


@torch.no_grad()
def smc(model: Model, K: int, g: torch.Generator, n_mh: int = 5, target: float = 0.5, max_steps: int = 300):
    """-> (particles [P, K, D] at β = 1, log Ẑ [P], diagnostics)."""
    t = model.t
    P, D, dev = t.xa.shape[0], len(model.idx), t.xa.device
    th = torch.randn(P, K, D, generator=g, device=dev)
    L = model.loglik(th)
    if D == 0:
        return th, L[:, 0], {"steps": 0, "acc": 1.0}
    beta = torch.zeros(P, device=dev)
    logz = torch.zeros(P, device=dev)
    sc = torch.ones(P, device=dev)
    lt = math.log(target * K)
    steps, accs = 0, []

    def lpri(x):
        return -0.5 * (x**2).sum(-1)

    def moves(nm):
        nonlocal th, L, sc
        X = th - th.mean(1, keepdim=True)
        cov = X.transpose(1, 2) @ X / (K - 1) + 1e-6 * torch.eye(D, device=dev)
        ch = torch.linalg.cholesky(cov)
        for _ in range(nm):
            z = torch.randn(P, K, D, generator=g, device=dev)
            prop = th + (2.38 / math.sqrt(D)) * sc[:, None, None] * torch.einsum("pij,pkj->pki", ch, z)
            Lp = model.loglik(prop)
            la = lpri(prop) + beta[:, None] * Lp - lpri(th) - beta[:, None] * L
            acc = torch.log(torch.rand(P, K, generator=g, device=dev)) < la
            th = torch.where(acc[..., None], prop, th)
            L = torch.where(acc, Lp, L)
            a = acc.float().mean(1)
            accs.append(float(a.mean()))
            sc = sc * torch.where(a < 0.15, 0.7, torch.where(a > 0.4, 1.3, 1.0))

    while bool((beta < 1).any()) and steps < max_steps:
        rem = 1 - beta
        Lf = torch.nan_to_num(L, neginf=-1e30)
        ok_full = _log_ess(rem[:, None] * Lf) >= lt
        lo, hi = torch.zeros(P, device=dev), rem.clone()
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            good = _log_ess(mid[:, None] * Lf) >= lt
            lo, hi = torch.where(good, mid, lo), torch.where(good, hi, mid)
        db = torch.where(ok_full, rem, lo.clamp_min(1e-7)).minimum(rem)
        db = torch.where(beta >= 1, torch.zeros_like(db), db)
        lw = db[:, None] * Lf
        logz = logz + torch.logsumexp(lw, -1) - math.log(K)
        beta = torch.where(beta + db >= 1 - 1e-6, torch.ones_like(beta), beta + db)
        idx = torch.multinomial(lw.softmax(-1), K, replacement=True, generator=g)
        th, L = th.gather(1, idx[..., None].expand(-1, -1, D)), L.gather(1, idx)
        moves(n_mh)
        steps += 1
    moves(2 * n_mh)
    return th, logz, {"steps": steps, "acc": sum(accs) / max(len(accs), 1)}


@torch.no_grad()
def predictive(model: Model, th: torch.Tensor, g: torch.Generator):
    """Equal-weight particles -> per particle (log bucket density at y* [P, K, Q], CDF at y* [P, K, Q]) with one
    posterior h draw each (exact mixture over particles)."""
    t = model.t
    m_q = model.m(th, t.xq)
    P, K = th.shape[:2]
    a = torch.zeros(P, K, device=th.device)
    b = torch.ones(P, K, device=th.device)
    if model.pi_id < 1 and t.xa.shape[1]:
        L_id, lb, rbar, s = h_terms_b(t.ya, model.m(th, t.xa), t.sig_a, model.bg)
        L_aff = torch.logsumexp(lb, -1) - math.log(lb.shape[-1])
        p_id = torch.sigmoid(math.log(model.pi_id) + L_id - math.log(1 - model.pi_id) - L_aff)
        is_id = torch.rand(P, K, generator=g, device=th.device) < p_id
        k = torch.multinomial(lb.reshape(P * K, -1).softmax(-1), 1, generator=g).reshape(P, K)
        bb = model.bg[k]
        rb = rbar.gather(-1, k[..., None])[..., 0]
        s2 = s[:, None]
        lo, hi = torch.special.ndtr(-rb / s2), torch.special.ndtr((1 - bb - rb) / s2)
        u = lo + (hi - lo) * torch.rand(P, K, generator=g, device=th.device)
        aa = (rb + s2 * torch.special.ndtri(u.clamp(1e-12, 1 - 1e-12))).clamp(0).minimum(1 - bb)
        a, b = torch.where(is_id, torch.zeros_like(aa), aa), torch.where(is_id, torch.ones_like(bb), bb)
    elif model.pi_id < 1:  # n_A = 0: h from its prior
        is_id = torch.rand(P, K, generator=g, device=th.device) < model.pi_id
        bb = C.B_LO + (1 - C.B_LO) * torch.rand(P, K, generator=g, device=th.device)
        aa = (1 - bb) * torch.rand(P, K, generator=g, device=th.device)
        a, b = torch.where(is_id, torch.zeros_like(aa), aa), torch.where(is_id, torch.ones_like(bb), bb)
    mu = a[..., None] + b[..., None] * m_q
    sig = t.sig_a[:, None, None]
    lp = bucket_lp(mu, sig, t.yq[:, None])
    cdf = 0.5 * (1 + torch.erf((t.yq[:, None] - mu) / (sig * math.sqrt(2))))
    return lp, cdf


@torch.no_grad()
def oracle(t: C.CisTasks, spec: FT.Spec, K: int = 2048, seed: int = 0, pi_id: float = C.PI_ID, chunk: int = 32,
           full: bool = False) -> dict:  # fmt: skip
    """-> {"lp": log predictive density at y* [P, Q], "cdf": [P, Q], "pJ": p(J | A) [P, nJ], "logz": [P, nJ],
    "mc": MC error per task (|half 1 − half 2| mean over queries) [P]} (+ "probs" [P, Q, G] if full)."""
    P = t.xa.shape[0]
    halves = []
    for h in range(2):
        g = torch.Generator(device=t.xa.device).manual_seed(seed * 1000 + 17 * h + 1)
        lps, cdfs, lzs, prs = [], [], [], []
        for J in spec.js:
            lp_J, cdf_J, lz_J, pr_J = [], [], [], []
            for s in range(0, P, chunk):
                sub = t.sub(torch.arange(s, min(s + chunk, P), device=t.xa.device))
                mdl = Model(sub, J, spec, pi_id)
                th, lz, _ = smc(mdl, K // 2, g)
                lp, cdf = predictive(mdl, th, g)
                lp_J.append(torch.logsumexp(lp, 1) - math.log(K // 2))
                cdf_J.append(cdf.mean(1))
                lz_J.append(lz)
                if full:
                    a_b = _probs(mdl, th, g)
                    pr_J.append(a_b)
            lps.append(torch.cat(lp_J))
            cdfs.append(torch.cat(cdf_J))
            lzs.append(torch.cat(lz_J))
            if full:
                prs.append(torch.cat(pr_J))
        lz = torch.stack(lzs, -1)  # [P, nJ]
        lpJ = torch.log(torch.tensor(spec.pj, device=t.xa.device)) + lz
        wJ = lpJ.softmax(-1)
        lp = torch.logsumexp(torch.log(wJ.clamp_min(1e-30))[..., None] + torch.stack(lps, 1), 1)
        cdf = (wJ[..., None] * torch.stack(cdfs, 1)).sum(1)
        half = {"lp": lp, "cdf": cdf, "pJ": wJ, "logz": lz}
        if full:
            half["probs"] = (wJ[..., None, None] * torch.stack(prs, 1)).sum(1)
        halves.append(half)
    h1, h2 = halves
    out = {
        "lp": torch.logaddexp(h1["lp"], h2["lp"]) - math.log(2),
        "cdf": 0.5 * (h1["cdf"] + h2["cdf"]),
        "pJ": 0.5 * (h1["pJ"] + h2["pJ"]),
        "logz": torch.logaddexp(h1["logz"], h2["logz"]) - math.log(2),
        "mc": (h1["lp"] - h2["lp"]).abs().mean(-1),
    }
    if full:
        out["probs"] = 0.5 * (h1["probs"] + h2["probs"])
    return out


def _probs(mdl: Model, th: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Bucket probabilities of the particle mixture at the queries [P, Q, G] (h = id toys only)."""
    m_q = mdl.m(th, mdl.t.xq)
    return CO.gauss_bucket_probs(m_q, mdl.t.sig_a[:, None, None].expand_as(m_q)).mean(1)
