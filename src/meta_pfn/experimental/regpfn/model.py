"""Back half of the registration model (docs/spec_phase1_experiments.md §6; design doc steps 3–5).

One seeded slot plus the null slot. The slot's state is a field per B token: its position x̂ and value ŷ in A's
frame and a hidden state h. Refinement (weight-tied, `n_iter` times):
  1. lookup:  each B token cross-attends to A's context with bias -|x̂_i - x_a|² / 2ℓ² (ℓ learned per head);
              an evidence mass e_i = Σ_a exp(-|x̂_i - x_a|² / 2ℓ_e²) gates the update, so with no A point
              near (or none at all) the field does not move;
  2. update:  h_i = GRU(h_i, [lookup_i, φ(x̂_i), ŷ_i, φ(u_i), y_i]);
  3. propose: δ_i = tanh(e_i) · MLP(h_i) ∈ R^{d+1};
  4. smooth:  Δ_i = Σ_j softmax_j(-|u_i - u_j|² / 2ℓ_h²) δ_j in B's own frame (coherence), then (x̂, ŷ) += Δ.
Read-out: A's queries attend to the context tokens (A's points, plus B rendered at (x̂, ŷ) for the slot) with a
distance bias in A's frame, a few cross-attention layers, then bar-distribution logits. The null slot reads
out from A's points alone. Mixture: p = w_seed p_seed + w_null p_null, w from a head on pooled slot features.
φ = Fourier features of x.
"""

import math

import torch
import torch.nn.functional as F
from pfn_repro.objectives.bar_distribution import BarDistribution, uniform_bin_borders
from torch import nn


def sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a_i - b_j|² [B, Na, Nb], with finite gradients at zero distance (unlike cdist)."""
    return ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)


class Fourier(nn.Module):
    def __init__(self, d: int, n_freq: int = 8):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(n_freq)) * math.pi)
        self.out = d * (2 * n_freq + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[..., None] * self.f
        return torch.cat([x, z.sin().flatten(-2), z.cos().flatten(-2)], -1)


class BiasedAttention(nn.Module):
    """Multi-head attention with an additive distance bias -|p_q - p_k|² / 2ℓ_h² per head (ℓ learned); one head
    is left unbiased. key_mask: [B, Nk] True = real."""

    def __init__(self, dm: int, heads: int, init_len: float = 0.1):
        super().__init__()
        self.h, self.dh = heads, dm // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dm, dm) for _ in range(4))
        ls = torch.log(torch.full((heads,), init_len)) + torch.linspace(-1, 1, heads)
        self.log_len = nn.Parameter(ls)
        self.register_buffer("biased", (torch.arange(heads) > 0).float())

    def forward(self, xq, xk, pq, pk, key_mask=None):
        B, Nq, _ = xq.shape
        Nk = xk.shape[1]
        q = self.q(xq).view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(xk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(xk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        d2 = sqdist(pq, pk)  # [B, Nq, Nk]
        bias = -d2[:, None] / (2 * self.log_len.exp()[None, :, None, None] ** 2) * self.biased[None, :, None, None]
        if key_mask is not None:
            bias = bias.masked_fill(~key_mask[:, None, None, :], -1e9)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = self.o(out.transpose(1, 2).reshape(B, Nq, -1))
        if key_mask is not None:  # no real key at all (n_A = 0): no message
            out = out * key_mask.any(-1).float()[:, None, None]
        return out


class CrossBlock(nn.Module):
    def __init__(self, dm: int, heads: int, init_len: float = 0.1):
        super().__init__()
        self.att = BiasedAttention(dm, heads, init_len)
        self.n1, self.n2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.ff = nn.Sequential(nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))

    def forward(self, xq, xk, pq, pk, key_mask=None):
        xq = xq + self.att(self.n1(xq), xk, pq, pk, key_mask)
        return xq + self.ff(self.n2(xq))


class Readout(nn.Module):
    """Queries at x* attend to context tokens at positions p (A's frame): self-attention over the context, then
    query cross-attention, then bar logits."""

    def __init__(self, d: int, dm: int, heads: int, layers: int, n_bins: int, extra: int):
        super().__init__()
        self.phi = Fourier(d)
        self.ctx_in = nn.Linear(self.phi.out + 1 + extra, dm)
        self.q_in = nn.Linear(self.phi.out, dm)
        self.ctx = nn.ModuleList([CrossBlock(dm, heads) for _ in range(layers)])
        self.qry = nn.ModuleList([CrossBlock(dm, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, n_bins))

    def forward(self, pos, y, extra, mask, xq):
        c = self.ctx_in(torch.cat([self.phi(pos), y[..., None], extra], -1))
        q = self.q_in(self.phi(xq))
        for bc, bq in zip(self.ctx, self.qry, strict=True):
            c = bc(c, c, pos, pos, mask)
            q = bq(q, c, xq, pos, mask)
        return self.out(q)


class RegBack(nn.Module):
    def __init__(
        self,
        d: int,
        dm: int = 128,
        heads: int = 4,
        layers: int = 3,
        n_bins: int = 64,
        n_iter: int = 8,
        resid: bool = False,
        max_step: float | None = None,
    ):
        """resid: also feed the kernel-weighted local residuals (mean y_a - ŷ_i, mean x_a - x̂_i near x̂_i) to the
        GRU; max_step: bound each correction to ±max_step (tanh)."""
        super().__init__()
        self.d, self.dm, self.n_iter, self.resid, self.max_step = d, dm, n_iter, resid, max_step
        self.phi = Fourier(d)
        self.bar = BarDistribution(uniform_bin_borders(n_bins))
        self.a_in = nn.Linear(self.phi.out + 1, dm)
        self.b_in = nn.Linear(2 * self.phi.out + 2, dm)
        self.lookup = CrossBlock(dm, heads, init_len=0.05)
        self.gru = nn.GRUCell(dm + self.phi.out + 1 + (d + 2 if resid else 0), dm)
        self.delta = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, d + 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.log_len_e = nn.Parameter(torch.log(torch.tensor(0.05)))
        self.log_len_h = nn.Parameter(torch.log(torch.tensor(0.05)))
        self.readout = Readout(d, dm, heads, layers, n_bins, extra=2)  # extra: [is_B, evidence]
        self.w_head = nn.Sequential(nn.Linear(2 * dm + 1, dm), nn.GELU(), nn.Linear(dm, 2))

    def refine(self, xa, ya, a_mask, ub, yb, x_hat, y_hat, b_mask, n_iter=None):
        """-> list of (x̂, ŷ) after every iteration (index 0 = seed), final hidden state, final evidence."""
        B, Nb, _ = ub.shape
        a_tok = self.a_in(torch.cat([self.phi(xa), ya[..., None]], -1))
        h = torch.zeros(B * Nb, self.dm, device=ub.device)
        hist = [(x_hat, y_hat)]
        smooth = torch.softmax(
            (-sqdist(ub, ub) / (2 * self.log_len_h.exp() ** 2)).masked_fill(~b_mask[:, None, :], -1e9),
            -1,
        )
        e = torch.zeros(B, Nb, device=ub.device)
        for _ in range(self.n_iter if n_iter is None else n_iter):
            b_tok = self.b_in(torch.cat([self.phi(ub), yb[..., None], self.phi(x_hat), y_hat[..., None]], -1))
            look = self.lookup(b_tok, a_tok, x_hat, xa, a_mask)
            d2 = sqdist(x_hat, xa)
            k = torch.exp(-d2 / (2 * self.log_len_e.exp() ** 2)) * a_mask[:, None, :]  # [B, Nb, Na]
            e = k.sum(-1)
            feats = [look, self.phi(x_hat), y_hat[..., None]]
            if self.resid:
                kn = k / e[..., None].clamp_min(1e-6)
                ry = (kn * (ya[:, None, :] - y_hat[..., None])).sum(-1, keepdim=True)
                rx = (kn[..., None] * (xa[:, None, :, :] - x_hat[:, :, None, :])).sum(2)
                feats += [ry, rx, torch.tanh(e)[..., None]]
            h = self.gru(torch.cat(feats, -1).reshape(B * Nb, -1), h)
            step = self.delta(h).view(B, Nb, -1)
            if self.max_step is not None:
                step = self.max_step * torch.tanh(step)
            delta = torch.tanh(e)[..., None] * step
            delta = smooth @ delta
            x_hat = x_hat + delta[..., : self.d]
            y_hat = y_hat + delta[..., self.d]
            hist.append((x_hat, y_hat))
        return hist, h.view(B, Nb, -1), e

    def forward(self, xa, ya, a_mask, ub, yb, x_hat, y_hat, b_mask, xq, n_iter=None):
        """-> mixture logits pieces: (logits_seed, logits_null, log_w [B, 2]), field history."""
        hist, h, e = self.refine(xa, ya, a_mask, ub, yb, x_hat, y_hat, b_mask, n_iter)
        xf, yf = hist[-1]
        B, Na = ya.shape
        zeros_a = torch.zeros(B, Na, 2, device=ya.device)
        b_extra = torch.stack([torch.ones_like(yf), torch.tanh(e)], -1)
        pos = torch.cat([xa, xf], 1)
        yy = torch.cat([ya, yf], 1)
        ex = torch.cat([zeros_a, b_extra], 1)
        m = torch.cat([a_mask, b_mask], 1)
        lg_seed = self.readout(pos, yy, ex, m, xq)
        lg_null = self.readout(xa, ya, zeros_a, a_mask, xq)
        hb = (h * b_mask[..., None]).sum(1) / b_mask.sum(1, keepdim=True).clamp_min(1)
        ha = (self.a_in(torch.cat([self.phi(xa), ya[..., None]], -1)) * a_mask[..., None]).sum(1)
        ha = ha / a_mask.sum(1, keepdim=True).clamp_min(1)
        n_feat = torch.log1p(a_mask.sum(1, keepdim=True).float())
        log_w = torch.log_softmax(self.w_head(torch.cat([hb, ha, n_feat], -1)), -1)
        return lg_seed, lg_null, log_w, hist

    def mixture_nll(self, lg_seed, lg_null, log_w, yq):
        """-> [B, Nq] -log(w_s p_s + w_n p_n)."""
        ls = -self.bar(lg_seed, yq)
        ln = -self.bar(lg_null, yq)
        return -torch.logsumexp(torch.stack([log_w[:, :1] + ls, log_w[:, 1:] + ln]), 0)
