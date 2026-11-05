"""GMM-token PFN (issue #23): context tokens carry a GMM over a d-dim displacement, the output at a query is a GMM.

Token i = MLP([emb(x_i), sum_m MLP(w_im, mu_im / ALPHA, log(s_i / ALPHA))]) (DeepSets over the token's components,
so their order carries nothing). A learned null token is always in the context. Pre-LN PFN layers: every token
attends to the context tokens only (queries never see each other). Head: K diagonal Gaussians in units of ALPHA.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .prior import ALPHA, Batch

LOG2PI = math.log(2 * math.pi)


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Layer(nn.Module):
    def __init__(self, D: int, heads: int, dff: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q, self.kv, self.o = nn.Linear(D, D), nn.Linear(D, 2 * D), nn.Linear(D, D)
        self.ff = mlp(D, dff, D)

    def forward(self, h: torch.Tensor, n_ctx: int) -> torch.Tensor:
        B, N, D = h.shape
        z = self.ln1(h)
        q = self.q(z).view(B, N, self.h, -1).transpose(1, 2)
        k, v = self.kv(z[:, :n_ctx]).view(B, n_ctx, 2, self.h, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        h = h + self.o(a)
        return h + self.ff(self.ln2(h))


class GMMPFN(nn.Module):
    def __init__(self, d: int, D: int = 128, layers: int = 4, heads: int = 4, dff: int = 256, K: int = 10):
        super().__init__()
        self.d, self.K = d, K
        self.xe = nn.Linear(d, D)
        self.ce = mlp(2 + d, D, D)
        self.te = mlp(2 * D, D, D)
        self.null = nn.Parameter(torch.zeros(1, 1, D))
        self.qe = nn.Parameter(torch.zeros(1, 1, D))
        self.layers = nn.ModuleList(Layer(D, heads, dff) for _ in range(layers))
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, K * (1 + 2 * d))

    def forward(self, b: Batch) -> dict:
        """Mixture over u* per query, in the original units: logw [B, Q, K], mean [B, Q, K, d], logstd same."""
        B, n, d = b.x.shape
        logs = torch.log(b.s / ALPHA)[..., None, None].expand(-1, -1, b.w.shape[-1], 1)
        comp = torch.cat([b.w[..., None], b.mu / ALPHA, logs], -1)
        pooled = (self.ce(comp) * b.cmask[..., None]).sum(-2)  # [B, n, D]
        ctx = self.te(torch.cat([self.xe(b.x), pooled], -1))
        h = torch.cat([self.null.expand(B, -1, -1), ctx, self.xe(b.xq) + self.qe], 1)
        for layer in self.layers:
            h = layer(h, n + 1)
        o = self.head(self.ln(h[:, n + 1 :]))  # [B, Q, K(1+2d)]
        Q, K = o.shape[1], self.K
        logw = torch.log_softmax(o[..., :K], -1)
        mean = o[..., K : K + K * d].view(B, Q, K, d) * ALPHA
        logstd = (o[..., K + K * d :].view(B, Q, K, d).clamp(-7.0, 2.0)) + math.log(ALPHA)
        return {"logw": logw, "mean": mean, "logstd": logstd}

    @staticmethod
    def log_density(p: dict, y: torch.Tensor) -> torch.Tensor:
        """log p(y) [B, Q] for y [B, Q, d]."""
        z = (y[:, :, None] - p["mean"]) * torch.exp(-p["logstd"])
        lp = (-0.5 * (LOG2PI + z.square()) - p["logstd"]).sum(-1)
        return torch.logsumexp(p["logw"] + lp, -1)

    @staticmethod
    def log_marginals(p: dict, y: torch.Tensor) -> torch.Tensor:
        """Per-dimension marginal log densities [B, Q, d] (for the factorised comparison)."""
        z = (y[:, :, None] - p["mean"]) * torch.exp(-p["logstd"])
        lp = -0.5 * (LOG2PI + z.square()) - p["logstd"]
        return torch.logsumexp(p["logw"][..., None] + lp, -2)
