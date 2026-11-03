"""The MTPFN paper's prior (Algorithm A.1, robust isotropic full-rank ICM; A.2 with ard=True) and exact GP references.

Prior, per dataset: x ~ U[0, 1]^d; task proportions π ~ Dir(α); task ids ~ Cat(π); K_T ~ LKJ(η = 1) (a correlation
matrix); ℓ ~ Gamma(3, 6) (rate 6, mean 0.5; BoTorch's old default); y ~ N(0, K_T ⊗ K_X); then every source task j
is, with probability p, resampled from an independent RBF GP with its own ℓ_j ~ Gamma(3, 6).
Not specified in the paper (deviations, here): observation noise σ ~ U[σ_lo, σ_hi] per dataset (added to every y);
α = 1; the target (task 0) gets ≥ 1 point; T and n_tot per batch from config ranges. Resampling an unrelated source
is done in one joint draw: its block of the covariance is replaced by its own RBF kernel and its cross-covariances
are zero, which is the same distribution as Alg. A.1 lines 11-15.

References for one evaluation dataset (exact Gaussian algebra, float64):
- bayes: the Bayes predictive under the prior (p), by adaptive importance sampling (population Monte Carlo) over
  (K_T, ℓ, σ, ℓ_j) in unconstrained coordinates (K draws from the prior, then 2K and 4K from a Gaussian fitted to the
  weighted draws with inflated covariance, mixed 10 % with the prior), the 2^(T-1) unrelated-masks enumerated exactly;
  weights ∝ p(z) p(mask) N(y; 0, C) / q(z); the last stage is the estimate (plain prior IS gave ESS 4-16 of 8192);
- bayes_p0: the same restricted to mask = 0 (the Bayes predictive under p = 0);
- ml_icm: the single draw with the highest marginal likelihood (a best-of-K stand-in for the paper's MAP ICM);
- oracle_icm: true (K_T, ℓ, σ, mask, ℓ_j);
- gp_target_bayes / gp_target_oracle: target data only, IS over (ℓ, σ) / true (ℓ, σ).
Predictives are per query point (marginal Gaussians, mixed over draws), matching a PFN's pointwise PPD.
"""

import itertools
import math
from dataclasses import dataclass

import torch
from torch.distributions import Gamma, LKJCholesky
from torch.distributions.transforms import CorrCholeskyTransform


@dataclass
class ICMBatch:
    x: torch.Tensor  # [P, T, N, d]
    y: torch.Tensor  # [P, T, N]
    mask: torch.Tensor  # [P, T, N] bool
    xq: torch.Tensor  # [P, Q, d]  target queries
    yq: torch.Tensor  # [P, Q]
    kt: torch.Tensor  # [P, T, T]
    ell: torch.Tensor  # [P, d]  (isotropic: all equal)
    ell_src: torch.Tensor  # [P, T, d]  unrelated sources' own lengthscales
    unrel: torch.Tensor  # [P, T] bool (task 0 always False)
    sig: torch.Tensor  # [P]


def _gamma(shape, g, device):
    # Gamma(3, rate 6) via the sum of 3 exponentials (seeded by g; torch's Gamma sampler takes no generator)
    u = torch.rand(*shape, 3, generator=g, device=device).clamp_min(1e-12)
    return -u.log().sum(-1) / 6.0


def _lkj(P, T, g, device):
    # LKJ(η = 1) Cholesky via the onion/C-vine construction of torch's LKJCholesky, made seedable through a forked RNG
    with torch.random.fork_rng(devices=[device] if str(device).startswith("cuda") else []):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g, device=device)))
        L = LKJCholesky(T, concentration=1.0).sample((P,)).to(device)
    return L @ L.transpose(-1, -2)


def rbf(a: torch.Tensor, b: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    """a [..., M, d], b [..., N, d], ell [..., d] -> exp(-½ Σ_k (a_k - b_k)² / ℓ_k²)."""
    a, b = a / ell[..., None, :], b / ell[..., None, :]
    d2 = (a[..., :, None, :] - b[..., None, :, :]).square().sum(-1)
    return torch.exp(-0.5 * d2)


def joint_cov(x, tid, kt, ell, ell_src, unrel):
    """x [..., M, d], tid [..., M] (task id per point), kt [..., T, T], ell [..., d], ell_src [..., T, d],
    unrel [..., T] -> noise-free covariance [..., M, M] of the robust ICM."""
    base = torch.gather(torch.gather(kt, -2, tid[..., :, None].expand(*tid.shape, kt.shape[-1])), -1,
                        tid[..., None, :].expand(*tid.shape[:-1], tid.shape[-1], tid.shape[-1]))  # fmt: skip
    base = base * rbf(x, x, ell)
    u = torch.gather(unrel, -1, tid)  # [..., M]
    same = tid[..., :, None] == tid[..., None, :]
    own_ell = torch.gather(ell_src, -2, tid[..., None].expand(*tid.shape, ell_src.shape[-1]))  # [..., M, d]
    a, b = x / own_ell, x / own_ell
    own = torch.exp(-0.5 * (a[..., :, None, :] - b[..., None, :, :]).square().sum(-1))  # same-task pairs only
    either = u[..., :, None] | u[..., None, :]
    return torch.where(either, torch.where(same, own, torch.zeros_like(own)), base)


def _pack(xf, yf, tid, T):
    """Flat points [P, M, d] with task ids [P, M] -> padded [P, T, N, d], y [P, T, N], mask."""
    P, M, d = xf.shape
    oh = torch.nn.functional.one_hot(tid, T)  # [P, M, T]
    rank = (oh.cumsum(1) - 1).gather(-1, tid[..., None])[..., 0]  # position within its task
    N = int(oh.sum(1).max())
    x = xf.new_zeros(P, T, N, d)
    y = yf.new_zeros(P, T, N)
    mask = torch.zeros(P, T, N, dtype=torch.bool, device=xf.device)
    pi = torch.arange(P, device=xf.device)[:, None].expand(P, M)
    x[pi, tid, rank] = xf
    y[pi, tid, rank] = yf
    mask[pi, tid, rank] = True
    return x, y, mask


def sample_icm(P: int, T: int, d: int, Q: int, g: torch.Generator, device, *, p: float = 0.2, n_tot: int = 60,
               counts: list[int] | None = None, alpha: float = 1.0, ard: bool = False, sig_lo: float = 0.01,
               sig_hi: float = 0.1, force_unrel: int | None = None) -> ICMBatch:  # fmt: skip
    """counts (per task, target first) fixes the design (evaluation); else n_tot points, Dirichlet(α) proportions."""
    r = lambda *s: torch.rand(*s, generator=g, device=device)  # noqa: E731
    if counts is not None:
        tid = torch.cat([torch.full((c,), t, device=device) for t, c in enumerate(counts)]).expand(P, -1).clone()
    else:
        pi = torch.distributions.Dirichlet(torch.full((T,), alpha, device=device))
        with torch.random.fork_rng(devices=[device] if str(device).startswith("cuda") else []):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g, device=device)))
            probs = pi.sample((P,))
        tid = torch.multinomial(probs, n_tot, replacement=True, generator=g)
        tid[:, 0] = 0  # the target has ≥ 1 point
    M = tid.shape[1]
    xf, xq = r(P, M, d), r(P, Q, d)
    kt = _lkj(P, T, g, device)
    ell = _gamma((P, 1 if not ard else d), g, device).clamp_min(0.02).expand(P, d).contiguous()
    ell_src = _gamma((P, T, 1), g, device).clamp_min(0.02).expand(P, T, d).contiguous()
    if force_unrel is None:
        unrel = r(P, T) < p
    else:
        order = r(P, T - 1).argsort(-1)
        unrel = torch.cat([torch.zeros(P, 1, device=device), (order < force_unrel).float()], 1) > 0
    unrel[:, 0] = False
    sig = sig_lo + (sig_hi - sig_lo) * r(P)
    xa = torch.cat([xf, xq], 1).double()
    ta = torch.cat([tid, tid.new_zeros(P, Q)], 1)
    C = joint_cov(xa, ta, kt.double(), ell.double(), ell_src.double(), unrel)
    C = C + (sig.double().square()[:, None, None] + 1e-8) * torch.eye(M + Q, device=device, dtype=torch.float64)
    L = torch.linalg.cholesky(C)
    ya = (L @ torch.randn(P, M + Q, 1, generator=g, device=device, dtype=torch.float64))[..., 0].float()
    x, y, mask = _pack(xf, ya[:, :M], tid, T)
    return ICMBatch(x, y, mask, xq, ya[:, M:], kt, ell, ell_src, unrel, sig)


# ---- exact references ------------------------------------------------------------------------------------------------


def _flat(b: ICMBatch, i: int):
    m = b.mask[i]
    T, N = m.shape
    tid = torch.arange(T, device=m.device)[:, None].expand(T, N)[m]
    return b.x[i][m].double(), b.y[i][m].double(), tid


def _gp_predict(C, ks, kss_diag, y, sig2):
    """Batched exact GP: C [K, M, M] (incl. noise), ks [K, M, Q], kss_diag [K, Q], y [M], sig2 [K]
    -> (log marginal likelihood [K], predictive mean [K, Q], predictive variance incl. noise [K, Q])."""
    L, info = torch.linalg.cholesky_ex(C)
    a = torch.cholesky_solve(y.expand(C.shape[0], -1)[..., None], L)  # [K, M, 1]
    M = C.shape[-1]
    lml = -0.5 * (y[None, :, None] * a).sum((-1, -2)) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    lml = lml - 0.5 * M * math.log(2 * math.pi)
    lml = torch.where(info == 0, lml, torch.full_like(lml, -float("inf")))
    mu = (ks * a).sum(-2)
    v = torch.linalg.solve_triangular(L, ks, upper=False)
    var = (kss_diag - v.square().sum(-2)).clamp_min(1e-10) + sig2[:, None]
    return lml, mu, var


def _lognorm(y, mu, var):
    return -0.5 * (math.log(2 * math.pi) + var.log() + (y - mu).square() / var)


def _mix(logw, lp):
    """logw [K], lp [K, Q] -> log Σ_k w_k p_k(y_q) [Q] (self-normalised)."""
    w = logw - torch.logsumexp(logw, 0)
    return torch.logsumexp(w[:, None] + lp, 0)


def _ess(logw):
    w = (logw - torch.logsumexp(logw, 0)).exp()
    return float(1.0 / w.square().sum())


@torch.no_grad()
def references(b: ICMBatch, i: int, p: float, K: int, g: torch.Generator, chunk: int = 2048) -> tuple[dict, dict]:
    """Exact GP references for dataset i of an evaluation batch. -> {name: log density per query [Q]} plus ESS."""
    dev = b.x.device
    T, d = b.kt.shape[-1], b.x.shape[-1]
    x, y, tid = _flat(b, i)
    xq, yq = b.xq[i].double(), b.yq[i].double()
    Q = xq.shape[0]
    xa = torch.cat([x, xq])
    ta = torch.cat([tid, tid.new_zeros(Q)])
    M = x.shape[0]
    masks = torch.tensor(list(itertools.product([False, True], repeat=T - 1)), device=dev)
    masks = torch.cat([torch.zeros(len(masks), 1, dtype=torch.bool, device=dev), masks], 1)  # [2^(T-1), T]
    log_pm = (masks[:, 1:].float() * math.log(max(p, 1e-300)) + (~masks[:, 1:]).float() * math.log1p(-p)).sum(-1)

    def run(kt, ell, ell_src, sig, unrel):
        n = kt.shape[0]
        C = joint_cov(xa.expand(n, -1, -1), ta.expand(n, -1), kt, ell, ell_src, unrel)
        Cc = C[:, :M, :M] + (sig.square()[:, None, None] + 1e-8) * torch.eye(M, device=dev, dtype=torch.float64)
        return _gp_predict(Cc, C[:, :M, M:], C[:, M:, M:].diagonal(dim1=-2, dim2=-1), y, sig.square())

    out = {}
    # oracle
    lml, mu, var = run(b.kt[i : i + 1].double(), b.ell[i : i + 1].double(), b.ell_src[i : i + 1].double(),
                       b.sig[i : i + 1].double(), b.unrel[i : i + 1])  # fmt: skip
    out["oracle_icm"] = _lognorm(yq, mu[0], var[0])
    # adaptive IS (population Monte Carlo) over hyperparameters in an unconstrained space; masks enumerated
    nc = T * (T - 1) // 2
    zdim = nc + 2 + (T - 1)

    def lik_all(z):
        """z [k, zdim] -> (log p(z) [k], per-mask log lik [k, 2^(T-1)], log predictive [k, 2^(T-1), Q])."""
        kt, ell, ell_src, sig, lpz = _from_z(z, T, d)
        lls, lps = [], []
        for mi in range(len(masks)):
            lml, mu, var = run(kt, ell, ell_src, sig, masks[mi].expand(z.shape[0], -1))
            lls.append(lml)
            lps.append(_lognorm(yq, mu, var))
        return lpz, torch.stack(lls, 1), torch.stack(lps, 1)

    def stage(z, logq):
        lpz, ll, lp = [], [], []
        for c in range(0, z.shape[0], chunk):
            a, b_, e = lik_all(z[c : c + chunk])
            lpz.append(a)
            ll.append(b_)
            lp.append(e)
        lpz, ll, lp = torch.cat(lpz), torch.cat(ll), torch.cat(lp)
        lw = lpz[:, None] + ll + log_pm[None] - logq[:, None]  # [k, masks]
        return lw, ll, lp

    def fit(z, lw, scale):
        w = (lw - torch.logsumexp(lw.flatten(), 0)).exp().sum(1)  # mask-marginal weights [k]
        m = (w[:, None] * z).sum(0)
        dz = z - m
        S = (w[:, None, None] * dz[:, :, None] * dz[:, None, :]).sum(0)
        return m, scale * S + 0.01 * torch.eye(zdim, device=dev, dtype=torch.float64)

    def draw_mix(k, m, S, eps=0.1):
        n0 = max(1, round(eps * k))
        z0 = _prior_z(n0, T, g, dev)
        L = torch.linalg.cholesky(S)
        z1 = m + (L @ torch.randn(k - n0, zdim, 1, generator=g, device=dev, dtype=torch.float64))[..., 0]
        z = torch.cat([z0, z1])
        lg = torch.distributions.MultivariateNormal(m, scale_tril=L).log_prob(z)
        lpz = _from_z(z, T, d)[4]
        return z, torch.logsumexp(torch.stack([math.log(eps) + lpz, math.log(1 - eps) + lg]), 0)

    z = _prior_z(K, T, g, dev)
    lw, ll, lp = stage(z, _from_z(z, T, d)[4])  # prior draws: logq = log p(z)
    best = [ll.max()]
    best_lp = [lp.flatten(0, 1)[int(ll.flatten().argmax())]]
    ess_stage = [_ess(lw.flatten())]
    for scale, k in ((4.0, 2 * K), (2.0, 4 * K)):
        m, S = fit(z, lw, scale)
        z, logq = draw_mix(k, m, S)
        lw, ll, lp = stage(z, logq)
        best.append(ll.max())
        best_lp.append(lp.flatten(0, 1)[int(ll.flatten().argmax())])
        ess_stage.append(_ess(lw.flatten()))
    out["bayes"] = _mix(lw.flatten(), lp.flatten(0, 1))
    lw0 = lw[:, 0] - log_pm[0]
    out["bayes_p0"] = _mix(lw0, lp[:, 0])
    out["ml_icm"] = best_lp[int(torch.stack(best).argmax())]
    ess = {"ess_bayes": ess_stage[-1], "ess_bayes_p0": _ess(lw0), "ess_bayes_stage0": ess_stage[0],
           "n_bayes": float(lw.numel())}  # fmt: skip
    # target only
    xt, yt = x[tid == 0], y[tid == 0]
    xat = torch.cat([xt, xq])
    Mt = xt.shape[0]

    def run_t(ell, sig):
        n = ell.shape[0]
        Kx = rbf(xat.expand(n, -1, -1), xat.expand(n, -1, -1), ell)
        Cc = Kx[:, :Mt, :Mt] + (sig.square()[:, None, None] + 1e-8) * torch.eye(Mt, device=dev, dtype=torch.float64)
        return _gp_predict(Cc, Kx[:, :Mt, Mt:], Kx[:, Mt:, Mt:].diagonal(dim1=-2, dim2=-1), yt, sig.square())

    lml, mu, var = run_t(b.ell[i : i + 1].double(), b.sig[i : i + 1].double())
    out["gp_target_oracle"] = _lognorm(yq, mu[0], var[0])
    ell = _gamma((K, 1), g, dev).clamp_min(0.02).expand(K, d).double()
    sig = (SIG_LO + (SIG_HI - SIG_LO) * torch.rand(K, generator=g, device=dev)).double()
    lml, mu, var = run_t(ell, sig)
    out["gp_target_bayes"] = _mix(lml, _lognorm(yq, mu, var))
    ess["ess_gp_target"] = _ess(lml)
    return {k: v.float() for k, v in out.items()}, ess


SIG_LO, SIG_HI = 0.01, 0.1  # the prior's noise range; references() assumes the defaults of sample_icm
_CC = CorrCholeskyTransform()
_GAM = Gamma(torch.tensor(3.0, dtype=torch.float64), torch.tensor(6.0, dtype=torch.float64))


def _from_z(z, T, d):
    """Unconstrained z [k, T(T-1)/2 + 2 + (T-1)] -> (K_T, ℓ [k, d], ℓ_src [k, T, d], σ, log prior density of z).
    z = (CorrCholesky coordinates of K_T, log ℓ, logit((σ - lo)/(hi - lo)), log ℓ_1..ℓ_{T-1})."""
    nc = T * (T - 1) // 2
    zc, zl, zs, zj = z[:, :nc], z[:, nc], z[:, nc + 1], z[:, nc + 2 :]
    L = _CC(zc)
    lp = LKJCholesky(T, concentration=1.0).log_prob(L) + _CC.log_abs_det_jacobian(zc, L)
    ell, ellj = zl.exp(), zj.exp()
    lp = lp + _GAM.log_prob(ell) + zl + (_GAM.log_prob(ellj) + zj).sum(-1)
    u = torch.sigmoid(zs)
    lp = lp + torch.nn.functional.logsigmoid(zs) + torch.nn.functional.logsigmoid(-zs)
    sig = SIG_LO + (SIG_HI - SIG_LO) * u
    kt = L @ L.transpose(-1, -2)
    ell_src = torch.cat([torch.ones_like(ellj[:, :1]), ellj], 1)[..., None].expand(-1, T, d)
    return kt, ell[:, None].expand(-1, d), ell_src, sig, lp


def _prior_z(k, T, g, dev):
    """Prior draws in z coordinates."""
    L = torch.linalg.cholesky(_lkj(k, T, g, dev).double())
    zc = _CC.inv(L)
    zl = _gamma((k,), g, dev).double().clamp_min(0.02).log()
    u = torch.rand(k, generator=g, device=dev, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    zj = _gamma((k, T - 1), g, dev).double().clamp_min(0.02).log()
    return torch.cat([zc, zl[:, None], torch.logit(u)[:, None], zj], 1)
