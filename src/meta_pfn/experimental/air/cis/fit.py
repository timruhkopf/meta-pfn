"""Field fit per assignment (spec §3.5): ridge initialisation on the linearised map, damped Gauss–Newton (LM, 5
iterations, autograd through RK4), Laplace Gaussian of the transform block, Laplace evidence.

Latent space: the standardized transform latents of the CIS prior (prior N(0, I)), restricted to the entries a
`Spec` makes free for a given J; for h = affine also (α, β) with the moment-matched Gaussian of their uniform
prior. The residual model is e_a = y_a − h(μ_B(T(x_a))), σ_a² = σ̄_A² + β² σ_c,a², where σ_c,a² is the surrogate's
variance at the assigned candidate (held fixed during the fit; proposal side only)."""

import math
from dataclasses import dataclass, field

import torch

from . import prior as C

SIG_BAR = 0.02
H_MEAN = torch.tensor([0.175, 0.65])  # (α, β) under β ~ U[0.3, 1], α ~ U[0, 1 − β]
H_COV = torch.tensor([[0.023819, -0.020417], [-0.020417, 0.040833]])


@dataclass(frozen=True)
class Spec:
    """Which latents the proposal (and the target prior) treats as free."""

    js: tuple = C.J_VALUES
    pj: tuple = C.P_J
    extras: tuple = (C.ZS, C.ZW, C.ZC)
    h_types: tuple = ("id", "aff")
    ph: tuple = (C.PI_ID, 1 - C.PI_ID)
    fixed: dict = field(default_factory=dict, hash=False, compare=False)  # unused hook

    def idx(self, J: int) -> list[int]:
        out = list(range(J))
        if J > 0 and C.ZS in self.extras:
            out.append(C.ZS)
        out += [k for k in (C.ZW, C.ZC) if k in self.extras]
        return out


def log_prior(lat: torch.Tensor, J: torch.Tensor, spec: Spec) -> torch.Tensor:
    """log N(0, I) of the free entries of lat [..., N_LAT] for J [...] under `spec`."""
    mask = torch.zeros(*J.shape, C.N_LAT, device=lat.device, dtype=lat.dtype)
    for Jv in spec.js:
        m = torch.zeros(C.N_LAT, device=lat.device, dtype=lat.dtype)
        m[spec.idx(Jv)] = 1
        mask = torch.where((Jv == J)[..., None], m, mask)
    return -0.5 * ((lat**2) * mask).sum(-1) - 0.5 * mask.sum(-1) * math.log(2 * math.pi)


def _features(x: torch.Tensor, J: int) -> torch.Tensor:
    """v_θ(x) = Σ_j θ̃_j s₀/(j+1) x(1 − x) cos(jπx) is linear in θ̃: -> F [..., n, J] (at z_s = 0)."""
    j = torch.arange(J, device=x.device, dtype=x.dtype)
    return x[..., None] * (1 - x[..., None]) * torch.cos(j * math.pi * x[..., None]) * (C.s0_for(J) / (j + 1))


def init(xa, uc, tau, J: int, spec: Spec, mu_c=None, ya=None, aff: bool = False) -> torch.Tensor:
    """Ridge initialisation so that T(x_a) ≈ uc: -> ψ [B, D] (transform latents in spec.idx(J) order, then (α, β))."""
    B, n = xa.shape
    w2 = 1 / tau**2
    if C.ZW in spec.extras:  # weighted ridge fit of u ≈ c + w x towards the prior means (w 0.75, c 0.125)
        lam = torch.tensor([1 / 0.15**2, 1 / 0.15**2], device=xa.device, dtype=xa.dtype)
        X = torch.stack([torch.ones_like(xa), xa], -1)  # columns (c, w)
        A = torch.einsum("bni,bn,bnj->bij", X, w2, X) + torch.diag(lam)
        rhs = torch.einsum("bni,bn,bn->bi", X, w2, uc) + lam * torch.tensor([0.125, 0.75], device=xa.device)
        cw = torch.linalg.solve(A, rhs)
        w = cw[:, 1].clamp(C.W_LO + 1e-3, 1 - 1e-3)
        c = cw[:, 0].clamp(1e-3).minimum(1 - w - 1e-3).clamp_min(1e-4)
        zw = torch.special.ndtri(((w - C.W_LO) / (1 - C.W_LO)).clamp(1e-4, 1 - 1e-4))
        zc = torch.special.ndtri((c / (1 - w)).clamp(1e-4, 1 - 1e-4))
    else:
        zw = zc = torch.zeros(B, device=xa.device, dtype=xa.dtype)
        w = C.W_LO + (1 - C.W_LO) * 0.5 * torch.ones_like(zw)
        c = (1 - w) * 0.5
    parts = []
    if J > 0:
        Fm = _features(xa, J)
        r = (uc - c[:, None]) / w[:, None] - xa
        Wt = (w[:, None] ** 2) * w2
        A = torch.einsum("bnj,bn,bnk->bjk", Fm, Wt, Fm) + torch.eye(J, device=xa.device, dtype=xa.dtype)
        parts.append(torch.linalg.solve(A, torch.einsum("bnj,bn,bn->bj", Fm, Wt, r)))
        if C.ZS in spec.extras:
            parts.append(torch.zeros(B, 1, device=xa.device, dtype=xa.dtype))
    if C.ZW in spec.extras:
        parts.append(zw[:, None])
    if C.ZC in spec.extras:
        parts.append(zc[:, None])
    if aff:
        if n >= 2:
            mc, yc = mu_c - mu_c.mean(1, keepdim=True), ya - ya.mean(1, keepdim=True)
            b = ((mc * yc).sum(1) / (mc**2).sum(1).clamp_min(1e-6)).clamp(C.B_LO, 1.0)
        else:
            b = torch.full((B,), 0.65, device=xa.device, dtype=xa.dtype)
        a = (ya.mean(1) - b * mu_c.mean(1)).clamp(0).minimum(1 - b)
        parts.append(torch.stack([a, b], 1))
    return torch.cat(parts, 1) if parts else xa.new_zeros(B, 0)


def to_lat(psi: torch.Tensor, J: int, spec: Spec) -> torch.Tensor:
    """ψ [B, D] -> full transform latent [B, N_LAT] (entries not free under spec are 0)."""
    idx = spec.idx(J)
    lat = psi.new_zeros(psi.shape[0], C.N_LAT)
    if idx:
        lat[:, idx] = psi[:, : len(idx)]
    return lat


def safe_chol(A: torch.Tensor) -> torch.Tensor:
    """Lower Cholesky factor of SPD matrices [B, D, D] with jitter (relative to the mean diagonal) added only on rows
    that fail (ill-conditioned Laplace covariances in float32; log, Deviations R10). The proposal samples with and
    evaluates its density under the returned factor, so the weights stay exact."""
    A = 0.5 * (A + A.transpose(1, 2))
    D = A.shape[-1]
    if D == 0:
        return A
    eye = torch.eye(D, device=A.device, dtype=A.dtype)
    scale = torch.diagonal(A, dim1=1, dim2=2).abs().mean(-1).clamp_min(1e-12)[:, None, None]
    L, info = torch.linalg.cholesky_ex(A)
    for eps in (1e-7, 1e-5, 1e-3, 1e-1):
        bad = info != 0
        if not bool(bad.any()):
            break
        L2, info2 = torch.linalg.cholesky_ex(A + eps * scale * eye)
        L = torch.where(bad[:, None, None], L2, L)
        info = torch.where(bad, info2, info)
    bad = info != 0
    if bool(bad.any()):  # last resort: the diagonal
        L = torch.where(
            bad[:, None, None], torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2).clamp_min(1e-12).sqrt()), L
        )
    return torch.nan_to_num(L)


GUARD: list = []  # (rows with the H fallback, rows) per fit call; diagnostics only


def _prior_terms(D_t: int, aff: bool, device, dtype):
    lam = torch.eye(D_t + 2 * aff, device=device, dtype=dtype)
    m0 = torch.zeros(D_t + 2 * aff, device=device, dtype=dtype)
    if aff:
        lam[D_t:, D_t:] = torch.linalg.inv(H_COV.to(device, dtype))
        m0[D_t:] = H_MEAN.to(device, dtype)
    return lam, m0


@dataclass
class Fit:
    psi: torch.Tensor  # [B, D] mode
    mean_t: torch.Tensor  # [B, D_t] transform-block mean
    chol_t: torch.Tensor  # [B, D_t, D_t] Cholesky of the transform-block marginal covariance [H⁻¹]_TT
    log_z: torch.Tensor  # [B] Laplace evidence
    h: torch.Tensor  # [B, 2] (α, β) at the mode ((0, 1) for h = id)


def fit(xa, ya, uc, tau, sig_c2, mu_fn, J: int, spec: Spec, aff: bool, iters: int = 5, sig_bar: float = SIG_BAR,
        loc_iters: int = 5, n_lis: int = 128, g: torch.Generator | None = None) -> Fit:  # fmt: skip
    """xa, ya, uc, tau, sig_c2 [B, n]; mu_fn(u [B, n]) -> μ_B(u) [B, n] (differentiable)."""
    B, n = xa.shape
    dev, dt = xa.device, xa.dtype
    D_t = len(spec.idx(J))
    Jt = torch.full((B,), J, device=dev, dtype=torch.long)
    mu_c = mu_fn(uc) if aff else None
    psi = init(xa, uc, tau, J, spec, mu_c, ya, aff).detach()
    lam, m0 = _prior_terms(D_t, aff, dev, dt)
    D = psi.shape[1]
    if D_t and loc_iters:  # deviation: LM on the location residuals through the actual flow before the data fit
        psi = torch.cat([_loc_fit(xa, uc, tau, psi[:, :D_t], J, spec, loc_iters), psi[:, D_t:]], 1)

    def resid(p):
        """p [B, D] or [B, S, D] (S draws per row) -> (e, σ²) [B, n] or [B, S, n]."""
        rep = p.dim() == 3
        xr, yr, sr = (v[:, None].expand(-1, p.shape[1], -1) if rep else v for v in (xa, ya, sig_c2))
        Jr = Jt[:, None].expand(-1, p.shape[1]) if rep else Jt
        lat = to_lat(p[..., :D_t].reshape(-1, D_t), J, spec).reshape(*p.shape[:-1], C.N_LAT)
        m = mu_fn(C.transform(xr, lat, Jr))
        if aff:
            a, b = p[..., D_t, None], p[..., D_t + 1, None]
            return yr - (a + b * m), sig_bar**2 + b**2 * sr
        return yr - m, sig_bar**2 + sr

    def objective(p):
        e, s2 = resid(p)
        d = p - m0
        return (e**2 / (2 * s2)).sum(1) + 0.5 * torch.einsum("bi,ij,bj->b", d, lam, d)

    def logjoint(p):
        e, s2 = resid(p)
        d = p - m0
        return ((-0.5 * e**2 / s2 - 0.5 * torch.log(2 * math.pi * s2)).sum(-1)
                - 0.5 * torch.einsum("...i,ij,...j->...", d, lam, d)
                + 0.5 * torch.logdet(lam) - 0.5 * D * math.log(2 * math.pi))  # fmt: skip

    def jac(p):
        p = p.detach().requires_grad_(True)
        e, s2 = resid(p)
        rows = [torch.autograd.grad(e[:, a].sum(), p, retain_graph=a < n - 1)[0] for a in range(n)]
        return e.detach(), s2.detach(), -torch.stack(rows, 1)  # r = −∂e/∂ψ: [B, n, D]

    if D == 0:
        e, s2 = resid(psi)
        lz = (-0.5 * e**2 / s2 - 0.5 * torch.log(2 * math.pi * s2)).sum(1)
        return Fit(psi, psi, psi.new_zeros(B, 0, 0), lz, _h(psi, D_t, aff))
    lm = torch.full((B,), 1e-2, device=dev, dtype=dt)
    obj = objective(psi).detach()
    for _ in range(iters):
        e, s2, r = jac(psi)
        Fm = torch.einsum("bni,bn,bnj->bij", r, 1 / s2, r)
        grad = torch.einsum("bni,bn,bn->bi", r, 1 / s2, e) - (psi - m0) @ lam
        H = lam + Fm
        step = torch.linalg.solve(H + lm[:, None, None] * torch.eye(D, device=dev, dtype=dt), grad)
        cand = psi + step
        with torch.no_grad():
            oc = objective(cand)
        ok = torch.isfinite(oc) & (oc < obj)
        psi = torch.where(ok[:, None], cand, psi)
        obj = torch.where(ok, oc, obj)
        lm = torch.where(ok, lm / 3, lm * 5)
    psi = torch.where(torch.isfinite(psi).all(1, keepdim=True), psi, m0.expand_as(psi))
    e, s2, r = jac(psi)
    Fm = torch.einsum("bni,bn,bnj->bij", r, 1 / s2, r)
    H = lam + Fm
    # guard (log, Deviations R10): a non-finite Jacobian (flow clamp / saturated g) leaves H unusable on rare rows;
    # those rows fall back to the prior precision (a wider proposal component: costs ESS, never validity)
    eyeD = torch.eye(D, device=dev, dtype=dt)
    Lh, info = torch.linalg.cholesky_ex(H + 1e-9 * eyeD)
    bad = (info != 0) | ~torch.isfinite(H).flatten(1).all(1)
    GUARD.append((int(bad.sum()), B))
    if bool(bad.any()):
        H = torch.where(bad[:, None, None], lam.expand_as(H), H)
        Lh = torch.linalg.cholesky(H + 1e-9 * eyeD)
    cov = torch.cholesky_inverse(Lh)
    ct = safe_chol(cov[:, :D_t, :D_t])
    d = psi - m0
    ld_h = torch.logdet(H)
    ld_l = torch.logdet(lam)
    lz = ((-0.5 * e**2 / s2 - 0.5 * torch.log(2 * math.pi * s2)).sum(1) - 0.5 * torch.einsum("bi,ij,bj->b", d, lam, d)
          - 0.5 * (ld_h - ld_l))  # fmt: skip
    if n_lis:  # fallback for test 4 (log, Deviations): importance-corrected Laplace evidence
        lz = _lis(psi.detach(), safe_chol(cov), logjoint, n_lis, g)
    lz = torch.nan_to_num(lz, nan=-1e30, posinf=-1e30)  # a component with no usable evidence gets no proposal mass
    return Fit(psi.detach(), psi[:, :D_t].detach(), ct.detach(), lz.detach(), _h(psi.detach(), D_t, aff))


@torch.no_grad()
def _lis(mode, chol, logjoint, S: int, g, infl: float = 1.5) -> torch.Tensor:
    """log Ẑ ≈ log mean_s exp(log p(A, ψ_s) − log q(ψ_s)), ψ_s ~ q = N(mode, infl² H⁻¹), S draws (the first at the
    mode): the Laplace evidence is off by up to ~0.5 nats on the skewed posteriors of this prior (test 4)."""
    B, D = mode.shape
    z = torch.randn(B, S, D, generator=g, device=mode.device, dtype=mode.dtype)
    z[:, 0] = 0
    ps = mode[:, None] + infl * torch.einsum("bij,bsj->bsi", chol, z)
    lj = logjoint(ps)
    lq = (-0.5 * (z**2).sum(-1) - torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(1, keepdim=True)
          - D * math.log(infl) - 0.5 * D * math.log(2 * math.pi))  # fmt: skip
    return torch.logsumexp(lj - lq, 1) - math.log(S)


def _loc_fit(xa, uc, tau, p0, J: int, spec: Spec, iters: int) -> torch.Tensor:
    """LM on Σ_a (u_a − T(x_a))²/(2τ_a²) + |ψ|²/2 (prior N(0, I)) from the ridge start: the linearisation
    T ≈ c + w(x + v) is poor for strong transforms, and a bad start sends the data fit to another basin."""
    B, D = p0.shape
    Jt = torch.full((B,), J, device=xa.device, dtype=torch.long)
    eye = torch.eye(D, device=xa.device, dtype=xa.dtype)

    def resid(p):
        return (uc - C.transform(xa, to_lat(p, J, spec), Jt)) / tau

    def obj(p):
        return 0.5 * (resid(p) ** 2).sum(1) + 0.5 * (p**2).sum(1)

    p, lm = p0, torch.full((B,), 1e-2, device=xa.device, dtype=xa.dtype)
    o = obj(p).detach()
    for _ in range(iters):
        q = p.detach().requires_grad_(True)
        e = resid(q)
        r = -torch.stack([torch.autograd.grad(e[:, a].sum(), q, retain_graph=a < e.shape[1] - 1)[0]
                          for a in range(e.shape[1])], 1)  # fmt: skip
        e = e.detach()
        H = eye + torch.einsum("bni,bnj->bij", r, r)
        gr = torch.einsum("bni,bn->bi", r, e) - p
        cand = p + torch.linalg.solve(H + lm[:, None, None] * eye, gr)
        with torch.no_grad():
            oc = obj(cand)
        ok = torch.isfinite(oc) & (oc < o)
        p, o = torch.where(ok[:, None], cand, p), torch.where(ok, oc, o)
        lm = torch.where(ok, lm / 3, lm * 5)
    return p.detach()


def _h(psi, D_t, aff):
    if aff:
        return psi[:, D_t : D_t + 2]
    return torch.tensor([0.0, 1.0], device=psi.device, dtype=psi.dtype).expand(psi.shape[0], 2)


def mvn_logpdf(x: torch.Tensor, mean: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """x, mean [B, D], chol [B, D, D] -> [B]."""
    D = x.shape[-1]
    if D == 0:
        return x.new_zeros(x.shape[0])
    z = torch.linalg.solve_triangular(chol, (x - mean)[..., None], upper=False)[..., 0]
    return (
        -0.5 * (z**2).sum(-1)
        - torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        - 0.5 * D * math.log(2 * math.pi)
    )
