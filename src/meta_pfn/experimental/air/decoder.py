"""Transported PFN D_ψ (proposal §3.2): A + B pulled into A's frame by θ -> predictive at x* and chain-rule evidence.

Sequence: [sink, B (N_B), A (n), queries x* (Q), evidence queries x_a (n)]. Attention (row attends to column):
- sink: itself; B tokens: sink + all B (B never sees A, so B's representations carry no A information);
- A token a: sink + B + A tokens with permutation rank ≤ rank(a) (causal within A, in a random order per task);
- query x*: sink + B + all A;
- evidence query a: sink + B + A tokens with rank < rank(a), so it predicts y_a from B and A_<a only;
- queries never attend to queries.
`use_b=False` masks B out: the null head (A only), the same weights.
Output: bar-distribution logits over y on A's scale (uniform borders on [Y_LO, Y_HI]).
Loss: soft-target cross-entropy (noise-free value with the task's σ_A, integrated over buckets) on x* and on the
evidence queries; NLL for evaluation uses the observed y (log bucket mass − log width).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

Y_LO, Y_HI, N_BINS = -0.25, 1.25, 400
N_FREQ = 6


def borders(device) -> torch.Tensor:
    return torch.linspace(Y_LO, Y_HI, N_BINS + 1, device=device)


def fourier(x: torch.Tensor) -> torch.Tensor:
    """x [...] -> [..., 1 + 2 N_FREQ]."""
    a = x[..., None] * (math.pi * 2.0 ** torch.arange(N_FREQ, device=x.device))
    return torch.cat([x[..., None], a.sin(), a.cos()], -1)


def soft_targets(f: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """Bucket probabilities of N(f, σ²): f [P, M], sig [P] -> [P, M, G]."""
    b = borders(f.device)
    cdf = 0.5 * (1 + torch.erf((b - f[..., None]) / (sig[:, None, None] * math.sqrt(2))))
    p = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


def bar_logpdf(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log density of y under the bar distribution: logits [..., G], y [...] -> [...]."""
    w = (Y_HI - Y_LO) / N_BINS
    idx = ((y - Y_LO) / w).floor().long().clamp(0, N_BINS - 1)
    return logits.log_softmax(-1).gather(-1, idx[..., None])[..., 0] - math.log(w)


class Block(nn.Module):
    def __init__(self, D: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.out = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = x + self.out(a.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.ln2(x))


def build_mask(n_b: int, n: int, Q: int, rank: torch.Tensor, use_b: bool) -> torch.Tensor:
    """rank [P, n] permutation ranks of A's points -> bool mask [P, L, L] (True = may attend)."""
    P, dev = rank.shape[0], rank.device
    L = 1 + n_b + n + Q + n
    m = torch.zeros(P, L, L, dtype=torch.bool, device=dev)
    sB, sA, sQ, sE = 1, 1 + n_b, 1 + n_b + n, 1 + n_b + n + Q
    m[:, :, 0] = True  # everyone sees the sink
    if use_b:
        m[:, sB:sA, sB:sA] = True  # B <-> B
        m[:, sA:, sB:sA] = True  # A, queries, evidence -> B
    ra, rb = rank[:, :, None], rank[:, None, :]
    m[:, sA:sQ, sA:sQ] = rb <= ra  # A causal in the permutation order (incl. itself)
    m[:, sQ:sE, sA:sQ] = True  # x* -> all A
    m[:, sE:, sA:sQ] = rb < ra  # evidence a -> A with smaller rank
    return m


class Decoder(nn.Module):
    def __init__(self, D: int = 128, layers: int = 6, heads: int = 4, sig_cond: bool = False):
        super().__init__()
        self.sig_cond = sig_cond
        if sig_cond:  # pilot 3: Σ of the noisy transform view θ_c, added to every token
            self.emb_s = nn.Linear(2, D)
        fx = 1 + 2 * N_FREQ
        self.emb_x, self.emb_y = nn.Linear(fx, D), nn.Linear(fx, D)
        self.src = nn.Parameter(torch.randn(4, D) * 0.02)  # B, A, query, evidence-query
        self.sink = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, N_BINS)

    def forward(self, xb_t, yb, xa, ya, xq, rank: torch.Tensor | None = None, use_b: bool = True,
                sig: torch.Tensor | None = None):  # fmt: skip
        """xb_t: B's inputs already in A's frame [P, N_B]; A [P, n]; queries [P, Q]; rank [P, n] (random if None).
        -> (logits at x* [P, Q, G], evidence logits at x_a [P, n, G])."""
        P, n_b, n, Q = xb_t.shape[0], xb_t.shape[1], xa.shape[1], xq.shape[1]
        if rank is None:
            rank = torch.rand(P, n, device=xa.device).argsort(-1).argsort(-1)
        tok = torch.cat([
            self.sink.expand(P, -1, -1),
            self.emb_x(fourier(xb_t)) + self.emb_y(fourier(yb)) + self.src[0],
            self.emb_x(fourier(xa)) + self.emb_y(fourier(ya)) + self.src[1],
            self.emb_x(fourier(xq)) + self.src[2],
            self.emb_x(fourier(xa)) + self.src[3],
        ], 1)  # fmt: skip
        if self.sig_cond:
            s = torch.zeros(P, device=xa.device) if sig is None else sig.to(xa.dtype).expand(P)
            tok = tok + self.emb_s(sig_feat(s))[:, None]
        mask = build_mask(n_b, n, Q, rank, use_b)
        for blk in self.blocks:
            tok = blk(tok, mask)
        out = self.head(self.ln(tok))
        s = 1 + n_b + n
        return out[:, s : s + Q], out[:, s + Q :]


def sig_feat(s: torch.Tensor) -> torch.Tensor:
    """Σ [P] -> features [P, 2]: (Σ, log(Σ + 0.01))."""
    return torch.stack([s, torch.log(s + 0.01)], -1)


def decoder_loss(dec: Decoder, xb_t, yb, xa, ya, xq, fa, fq, sig, use_b: bool = True,
                 sig_c: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """Soft-target CE on x* and on the evidence queries (mean per token). sig = σ_A (targets); sig_c = Σ (pilot 3)."""
    lq, le = dec(xb_t, yb, xa, ya, xq, use_b=use_b, sig=sig_c)
    tq = soft_targets(fq, sig)
    loss = -(tq * lq.log_softmax(-1)).sum(-1).mean()
    if xa.shape[1]:
        te = soft_targets(fa, sig)
        loss = loss - (te * le.log_softmax(-1)).sum(-1).mean()
    return loss + math.log((Y_HI - Y_LO) / N_BINS) * (1 + (xa.shape[1] > 0))  # report as NLL in density units
