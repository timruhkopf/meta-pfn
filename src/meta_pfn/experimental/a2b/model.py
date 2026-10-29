"""The A→B architecture (docs/design_2026-10-28_A_to_B_architecture.md §2) for S3/S4 (issue #18).

Frozen side: B is encoded once by the frozen PFN (per-layer K/V of B's tokens and their last-layer states);
`FrozenQB.query`
is the PFN's test-token path against that cache, returning bar logits and the last hidden state, differentiable in x.
Learned side:
- Perceiver over B's tokens (γ(x_b), y_b, frozen PFN state) -> Z_B [M, D];
- A-encoder (2 self-attention layers over (γ(x_a), y_a)) and a sink token;
- K particle tokens, DETR-style init: learned queries read Z_B, then A, then each other;
- decoders: T_k(x) = σ(logit(x) + Δ(γ(x), P_k)) (a residual field, identity at init); with `learn_h`, a monotone knot
  map h_k on y (J pieces, h(0) and h(1) free, identity at init);
- R weight-tied evidence blocks: query B at T_k(x_a), push through h_k, log-likelihood λ_ka and PIT; evidence tokens
  E_ka; self-attention within a particle, lookup into Z_B, particle update; explicit weights
  log w_k = prior_head(P_k) + Σ_a λ_ka (no tempering);
- prediction per particle: h_k pushed through Q_B(T_k(x*)) (`pseudo`), or, at the last step, the PFN re-encoded on
  B ∪ (T_k(x_a), h_k⁻¹(y_a)) and queried at T_k(x*) (`context`: A's evidence enters B's context); Bayes mixture over k.
Densities on A's scale use change of variables, p_A(y) = p_B(h⁻¹(y)) |(h⁻¹)'(y)|, mixed with ε·Unif for robustness.
The bar log-density is interpolated between bin centres so that it is differentiable in y (for learning h).
"""

import math

import torch
import torch.nn as nn
from pfn_repro.surrogates.pfn import _pad_and_rescale_features

EPS_MIX = 1e-3
N_FREQ = 4


def fourier(x: torch.Tensor) -> torch.Tensor:
    """γ(x) = [x, sin(2π 2^j x), cos(2π 2^j x)], j < N_FREQ, per coordinate: [..., d] -> [..., d (1 + 2 N_FREQ)]."""
    a = x[..., None] * (2 * math.pi * 2.0 ** torch.arange(N_FREQ, device=x.device))
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def gdim(d: int) -> int:
    return d * (1 + 2 * N_FREQ)


class FrozenQB:
    """The frozen PFN's context encoder and test path, with hidden states. Not an nn.Module (never trained)."""

    def __init__(self, pfn):
        self.pfn = pfn
        b = pfn.bar_dist.borders.float()
        self.borders, self.width = b, b[1:] - b[:-1]
        self.centers = 0.5 * (b[1:] + b[:-1])

    def encode(self, x, y, mask=None, grad: bool = False):
        """x [B, N, d], y [B, N] -> (kv per layer, n_features, last-layer train states [B, N, D_B])."""
        pfn = self.pfn
        with torch.set_grad_enabled(grad):
            nf = torch.full((x.shape[0],), x.shape[-1], dtype=torch.long, device=x.device)
            xs = _pad_and_rescale_features(x, pfn.max_x_dim, nf)
            tok = pfn.train_embed(torch.cat([xs, y[..., None]], -1))
            kv = []
            for blk in pfn.blocks:
                c = blk.ln1(tok)
                kv.append((blk.attn.k_proj(c), blk.attn.v_proj(c)))
                tok, _ = blk(tok, tok[:, :0], mask, None)
        return kv, nf, tok

    def query(self, kv, nf, x, mask=None):
        """x [B, Q, d] -> (logits [B, Q, G], hidden [B, Q, D_B]); differentiable in x (and in kv)."""
        pfn = self.pfn
        xs = _pad_and_rescale_features(x, pfn.max_x_dim, nf)
        tok = pfn.test_x_embed(xs) + pfn.test_placeholder.view(1, 1, -1)
        B, Q, Dm = tok.shape
        for blk, (K, V) in zip(pfn.blocks, kv, strict=True):
            a = blk.attn
            N = K.shape[1]
            q = a.q_proj(blk.ln1(tok)).view(B, Q, a.h, a.dk).transpose(1, 2)
            k = K.view(B, N, a.h, a.dk).transpose(1, 2)
            v = V.view(B, N, a.h, a.dk).transpose(1, 2)
            s = (q @ k.transpose(-2, -1)) / (a.dk**0.5)
            if mask is not None:
                s = s.masked_fill(~mask.view(B, 1, 1, N), torch.finfo(s.dtype).min)
            tok = tok + a.out_proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, Q, Dm))
            tok = tok + blk.ff(blk.ln2(tok))
        hid = pfn.out_ln(tok)
        return pfn.out_head(hid).float(), hid

    def logpdf(self, logits, u):
        """log density of the bar distribution at u (same shape as logits[..., 0]), linearly interpolated between bin
        centres (differentiable in u)."""
        return self.logpdf_grid(logits, u[..., None])[..., 0]

    def logpdf_grid(self, logits, u):
        """Several points per distribution: logits [..., G], u [..., Y] -> [..., Y]."""
        ld = logits.log_softmax(-1) - self.width.log()
        G = ld.shape[-1]
        pos = ((u - self.centers[0]) / self.width[0]).clamp(0, G - 1)
        i0 = pos.floor().long().clamp(max=G - 2)
        fr = pos - i0
        return (1 - fr) * ld.gather(-1, i0) + fr * ld.gather(-1, i0 + 1)

    def cdf(self, logits, u):
        p = logits.softmax(-1)
        G = p.shape[-1]
        idx = (torch.searchsorted(self.borders.contiguous(), u.contiguous().detach(), right=True) - 1).clamp(0, G - 1)
        c = p.cumsum(-1).gather(-1, idx[..., None])[..., 0]
        pi = p.gather(-1, idx[..., None])[..., 0]
        return c - pi + pi * ((u - self.borders[idx]) / self.width[idx]).clamp(0, 1)


def mix_eps(lp: torch.Tensor) -> torch.Tensor:
    """log((1 - ε) e^lp + ε): a uniform floor so one outlier can't zero a particle."""
    return torch.logaddexp(lp + math.log1p(-EPS_MIX), torch.full_like(lp, math.log(EPS_MIX)))


class Block(nn.Module):
    """Pre-LN (cross-)attention + MLP."""

    def __init__(self, D: int, heads: int = 4):
        super().__init__()
        self.lq, self.lkv, self.l2 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, q, kv=None):
        kv = q if kv is None else kv
        q = q + self.att(self.lq(q), self.lkv(kv), self.lkv(kv), need_weights=False)[0]
        return q + self.ff(self.l2(q))


class MonotoneH(nn.Module):
    """h_k on [0, 1]: J linear pieces with knot values H_j = c + Σ_{i<j} inc_i, (c, inc_1..J, tail) = softmax(W P_k), so
    h(0) = c, h(1) = 1 - tail, strictly increasing. Identity at init."""

    def __init__(self, D: int, J: int = 16):
        super().__init__()
        self.J = J
        self.lin = nn.Linear(D, J + 2)
        nn.init.zeros_(self.lin.weight)
        with torch.no_grad():
            self.lin.bias.zero_()
            self.lin.bias[0] = self.lin.bias[-1] = -8.0

    def knots(self, P):
        s = self.lin(P).softmax(-1)
        inc = s[..., 1:-1]
        H = s[..., :1] + torch.cat([torch.zeros_like(inc[..., :1]), inc.cumsum(-1)], -1)
        return H, inc

    def inv(self, P, y):
        """P [..., D], y [..., M] -> (h⁻¹(y) in [0, 1], log |(h⁻¹)'(y)|, -inf outside h's range)."""
        H, inc = self.knots(P)
        j = (torch.searchsorted(H[..., 1:-1].contiguous(), y.contiguous().detach())).clamp(0, self.J - 1)
        Hj, ij = H.gather(-1, j), inc.gather(-1, j)
        u = (j + (y - Hj) / ij) / self.J
        lj = -(ij * self.J).log()
        inside = (y >= H[..., :1]) & (y <= H[..., -1:])
        return u.clamp(0, 1), torch.where(inside, lj, torch.full_like(lj, -torch.inf))


class A2B(nn.Module):
    def __init__(self, pfn, d: int, D: int = 128, M: int = 16, K: int = 8, R: int = 4, learn_h: bool = False,
                 J: int = 16, predictive: str = "pseudo", null: bool = True):  # fmt: skip
        super().__init__()
        self._qb = [FrozenQB(pfn)]  # a list: keeps the frozen PFN out of state_dict and the optimizer
        self.d, self.D, self.K, self.R, self.learn_h, self.predictive = d, D, K, R, learn_h, predictive
        gx, gy, DB = gdim(d), gdim(1), pfn.out_head.in_features
        self.b_in = nn.Linear(gx + gy + DB, D)
        self.latents = nn.Parameter(torch.randn(M, D) * 0.02)
        self.perc_x, self.perc = Block(D), nn.ModuleList([Block(D), Block(D)])
        self.a_in = nn.Linear(gx + gy, D)
        self.a_enc = nn.ModuleList([Block(D), Block(D)])
        self.sink = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.q0 = nn.Parameter(torch.randn(K, D) * 0.02)
        self.init_b, self.init_a, self.init_p = Block(D), Block(D), Block(D)
        self.t_dec = nn.Sequential(nn.Linear(gx + D, D), nn.GELU(), nn.Linear(D, D), nn.GELU())
        self.t_out = nn.Linear(D, d)
        nn.init.zeros_(self.t_out.weight)
        nn.init.zeros_(self.t_out.bias)
        self.h_dec = MonotoneH(D, J) if learn_h else None
        self.ev_in = nn.Sequential(nn.Linear(2 * gx + gy + DB + 2, D), nn.GELU(), nn.Linear(D, D))
        self.ev_self, self.ev_look, self.p_read, self.p_self = Block(D), Block(D), Block(D), Block(D)
        self.prior_head = nn.Linear(D, 1)
        self.null = null
        self.null_bias = nn.Parameter(torch.zeros(()))  # learned prior log-odds of the null particle
        self.w_in = nn.Linear(2, D)

    @property
    def qb(self) -> FrozenQB:
        return self._qb[0]

    def T(self, P, x):
        """P [B, K, D], x [B, M, d] -> T_k(x) [B, K, M, d]."""
        B, K, _ = P.shape
        xe = x[:, None].expand(B, K, -1, -1)
        g = fourier(xe)
        delta = self.t_out(self.t_dec(torch.cat([g, P[:, :, None].expand(-1, -1, x.shape[1], -1)], -1)))
        xc = xe.clamp(1e-4, 1 - 1e-4)
        return torch.sigmoid(torch.log(xc) - torch.log1p(-xc) + delta)

    def h_inv(self, P, y):
        """y [B, M] -> (u [B, K, M], logjac [B, K, M])."""
        ye = y[:, None].expand(-1, P.shape[1], -1)
        if self.h_dec is None:
            return ye, torch.zeros_like(ye)
        return self.h_dec.inv(P, ye)

    def particle_lp(self, logits, u, lj):
        return mix_eps(self.qb.logpdf(logits, u) + lj)

    @torch.no_grad()
    def null_particle(self, xa, ya, xq, yq) -> dict:
        """The null particle ("B is unrelated"): the frozen PFN on A alone. Evidence = its chain-rule log-likelihood of
        A's labels, Σ_a log PFN(y_a | A_<a, x_a); predictive = PFN(A) at x*. Empty contexts use the repo's convention
        (masked (0, 0) dummy tokens, as `oracle.logp`). No gradient (no parameters)."""
        B, n, d = xa.shape
        if n == 0:
            xc, yc, m = xa.new_zeros(B, 1, d), ya.new_zeros(B, 1), torch.zeros(B, 1, dtype=torch.bool, device=xa.device)
            ev = xa.new_zeros(B)
        else:
            xc, yc, m = xa, ya, torch.ones(B, n, dtype=torch.bool, device=xa.device)
            keep = torch.ones(n, n, device=xa.device).tril(-1).bool()  # row i: tokens j < i
            Xc = (xa[:, None] * keep[None, :, :, None]).reshape(B * n, n, d)
            Yc = (ya[:, None] * keep[None]).reshape(B * n, n)
            Mc = keep[None].expand(B, -1, -1).reshape(B * n, n)
            kvc, nfc, _ = self.qb.encode(Xc, Yc, mask=Mc)
            lgc, _ = self.qb.query(kvc, nfc, xa.reshape(B * n, 1, d), mask=Mc)
            ev = mix_eps(self.qb.logpdf(lgc[:, 0], ya.reshape(-1))).reshape(B, n).sum(1)
        kv, nf, _ = self.qb.encode(xc, yc, mask=m)
        lg, _ = self.qb.query(kv, nf, xq, mask=m)
        return {"ev": ev, "lq": mix_eps(self.qb.logpdf(lg, yq)), "logits": lg}

    def forward(self, xb, yb, xa, ya, xq, yq, R: int | None = None) -> list[dict]:
        """-> one dict per step r = 0..R: logw [B, K], lq [B, K, Q] (log density of yq per particle), ta [B, K, n, d],
        tq [B, K, Q, d], lam [B, K, n]."""
        R = self.R if R is None else R
        B, n, Q, d, K = xb.shape[0], xa.shape[1], xq.shape[1], self.d, self.K
        kv, nf, sb = self.qb.encode(xb, yb)
        tb = self.b_in(torch.cat([fourier(xb), fourier(yb[..., None]), sb], -1))
        Z = self.perc_x(self.latents[None].expand(B, -1, -1), tb)
        for blk in self.perc:
            Z = blk(Z)
        sink = self.sink.expand(B, -1, -1)
        H = self.a_in(torch.cat([fourier(xa), fourier(ya[..., None])], -1))
        for blk in self.a_enc:
            H = blk(H) if n else H
        P = self.q0[None].expand(B, -1, -1)
        P = self.init_b(P, Z)
        P = self.init_a(P, torch.cat([H, sink], 1))
        P = self.init_p(P)
        kvK = [(k.repeat_interleave(K, 0), v.repeat_interleave(K, 0)) for k, v in kv]
        nfK = nf.repeat_interleave(K, 0)

        def q_b(t):  # t [B, K, M, d] -> logits [B, K, M, G], hidden
            lg, hd = self.qb.query(kvK, nfK, t.reshape(B * K, -1, d))
            return lg.reshape(B, K, -1, lg.shape[-1]), hd.reshape(B, K, -1, hd.shape[-1])

        null = self.null_particle(xa, ya, xq, yq) if self.null else None
        out = []
        for r in range(R + 1):
            ta, tq = self.T(P, xa), self.T(P, xq)
            if n:
                lg_a, hd_a = q_b(ta)
                ua, lja = self.h_inv(P, ya)
                lam = self.particle_lp(lg_a, ua, lja)
            else:
                lam = P.new_zeros(B, K, 0)
            logw = self.prior_head(P)[..., 0] + lam.sum(-1)
            last = r == R
            uq, ljq = self.h_inv(P, yq)
            if last and self.predictive == "context" and n:
                xc = torch.cat([xb[:, None].expand(-1, K, -1, -1), ta], 2).reshape(B * K, -1, d)
                yc = torch.cat([yb[:, None].expand(-1, K, -1), ua], 2).reshape(B * K, -1)
                kvc, nfc, _ = self.qb.encode(xc, yc, grad=True)
                lg_q, hq = self.qb.query(kvc, nfc, tq.reshape(B * K, Q, d))
                lg_q, hq = lg_q.reshape(B, K, Q, -1), hq.reshape(B, K, Q, -1)
            else:
                lg_q, hq = q_b(tq)
            lq = self.particle_lp(lg_q, uq, ljq)
            o = {"logw": logw, "lq": lq, "ta": ta, "tq": tq, "lam": lam, "P": P, "hq": hq}
            if null is not None:
                o |= {"null_lw": self.null_bias + null["ev"], "null_lq": null["lq"], "null_logits": null["logits"]}
            out.append(o)
            if last:
                break
            if n:  # evidence block
                pit = self.qb.cdf(lg_a, ua)
                feats = torch.cat([fourier(xa)[:, None].expand(-1, K, -1, -1), fourier(ta),
                                   fourier(ya[..., None])[:, None].expand(-1, K, -1, -1), hd_a,
                                   (lam / 10)[..., None], pit[..., None]], -1)  # fmt: skip
                E = (self.ev_in(feats) + P[:, :, None]).reshape(B * K, n, -1)
                E = self.ev_self(E)
                E = self.ev_look(E, Z.repeat_interleave(K, 0))
                ctx = torch.cat([E, sink.repeat_interleave(K, 0)], 1)
            else:
                ctx = sink.repeat_interleave(K, 0)
            P = self.p_read(P.reshape(B * K, 1, -1), ctx).reshape(B, K, -1)
            lw = self.prior_head(P)[..., 0] + lam.sum(-1)
            lwn = lw - torch.logsumexp(lw, -1, keepdim=True)
            rank = lw.argsort(-1).argsort(-1).float() / max(K - 1, 1)
            P = P + self.w_in(torch.stack([lwn.clamp(min=-20) / 10, rank], -1))
            P = self.p_self(P)
        return out


def mix_weights(o: dict, with_null: bool = True) -> torch.Tensor:
    """Normalized log weights [B, K (+1 for the null particle, last, if present and with_null)]."""
    lw = o["logw"] if not (with_null and "null_lw" in o) else torch.cat([o["logw"], o["null_lw"][:, None]], 1)
    return lw - torch.logsumexp(lw, -1, keepdim=True)


def mixture_lp(o: dict, with_null: bool = True) -> torch.Tensor:
    """Bayes mixture log density of yq: [B, Q]; `with_null=False`: the particles only."""
    lq = o["lq"] if not (with_null and "null_lw" in o) else torch.cat([o["lq"], o["null_lq"][:, None]], 1)
    return torch.logsumexp(mix_weights(o, with_null)[..., None] + lq, 1)
