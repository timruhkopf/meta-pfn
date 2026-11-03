"""Proposal q_φ(θ̃ | A, B) (proposal §3.3): encoder E0 + conditional neural spline flow, trained by NPE.

E0: A tokens (γ(x_a), y_a) and B tokens (γ(x_b), y_b, B in its own coordinates), source embeddings; 2 self-attention
layers within each task (shared weights); 2 layers where A attends to B (cross-attention); pooling by multi-head
attention with 4 learned seeds over the A tokens -> c ∈ R^{4·64}. Flow: zuko NSF over θ̃ ∈ R^J (standardized, prior
N(0, I)), 5 transforms, hidden 128, conditioned on c. Loss: -log q(θ̃_true | c) (forward KL: mass-covering).
At inference: the defensive mixture q_mix = (1 - ε) q + ε N(0, I), ε = 0.1.
"""

import math

import torch
import zuko
from torch import nn

from .decoder import fourier, sig_feat
from .prior import log_prior_s

EPS_DEF = 0.1


class XBlock(nn.Module):
    def __init__(self, D: int, heads: int = 4):
        super().__init__()
        self.lq, self.lk, self.l2 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, q, kv=None):
        kv = q if kv is None else kv
        q = q + self.att(self.lq(q), self.lk(kv), self.lk(kv), need_weights=False)[0]
        return q + self.ff(self.l2(q))


class Encoder(nn.Module):
    def __init__(self, D: int = 64, seeds: int = 4):
        super().__init__()
        f = 2 * (1 + 2 * 6)
        self.inp = nn.Linear(f, D)
        self.src = nn.Parameter(torch.randn(2, D) * 0.02)
        self.self_blocks = nn.ModuleList([XBlock(D), XBlock(D)])
        self.cross_blocks = nn.ModuleList([XBlock(D), XBlock(D)])
        self.seeds = nn.Parameter(torch.randn(seeds, D) * 0.02)
        self.pool = XBlock(D)
        self.out_dim = seeds * D

    def forward(self, xb, yb, xa, ya) -> torch.Tensor:
        A = self.inp(torch.cat([fourier(xa), fourier(ya)], -1)) + self.src[1]
        B = self.inp(torch.cat([fourier(xb), fourier(yb)], -1)) + self.src[0]
        for blk in self.self_blocks:
            A, B = blk(A), blk(B)
        for blk in self.cross_blocks:
            A = blk(A, B)
        s = self.pool(self.seeds[None].expand(A.shape[0], -1, -1), A)
        return s.flatten(1)


class Proposal(nn.Module):
    def __init__(self, J: int, D: int = 64, hidden: int = 128, transforms: int = 5, sig_cond: bool = False):
        super().__init__()
        self.J, self.sig_cond = J, sig_cond
        self.enc = Encoder(D)
        self.flow = zuko.flows.NSF(features=J, context=self.enc.out_dim + 2 * sig_cond, transforms=transforms,
                                   hidden_features=[hidden, hidden])  # fmt: skip

    def dist(self, xb, yb, xa, ya, sig=None):
        """q(θ | A, B) (sig_cond: q(θ_c | A, B, Σ), sig a float or [P])."""
        c = self.enc(xb, yb, xa, ya)
        if self.sig_cond:
            s = torch.as_tensor(0.0 if sig is None else sig, device=c.device, dtype=c.dtype).expand(c.shape[0])
            c = torch.cat([c, sig_feat(s)], -1)
        return self.flow(c)

    def nll(self, xb, yb, xa, ya, theta, sig=None) -> torch.Tensor:
        return -self.dist(xb, yb, xa, ya, sig).log_prob(theta).mean()

    @torch.no_grad()
    def sample_mix(self, xb, yb, xa, ya, K: int, eps: float = EPS_DEF, sig: float = 0.0, qmc: bool = False,
                   seed: int = 0):  # fmt: skip
        """K draws per task from q_mix = (1 - ε) q + ε N(0, (1 + Σ) I) -> (θ [P, K, J], log q_mix(θ) [P, K]).
        qmc: scrambled Sobol points through the flow's base (and the prior), with the mixture split deterministically
        (round(εK) prior draws): deterministic-mixture IS, weights unchanged, so the estimator stays consistent."""
        d = self.dist(xb, yb, xa, ya, sig)
        P, J, dev = xb.shape[0], self.J, xb.device
        sd = math.sqrt(1.0 + sig)
        if not qmc:
            th_q = d.sample((K,)).transpose(0, 1)  # [P, K, J]
            th_p = torch.randn_like(th_q) * sd
            pick = torch.rand(P, K, 1, device=dev) < eps
            th = torch.where(pick, th_p, th_q)
        else:
            k_p = round(eps * K)
            z = _sobol_normal(P, K, J, seed, dev)  # [P, K, J]
            th_q = d.transform.inv(z[:, k_p:].transpose(0, 1)).transpose(0, 1)
            th = torch.cat([z[:, :k_p] * sd, th_q], 1)
        return th, self.log_q_mix(d, th, eps, sig)

    @staticmethod
    def log_q_mix(d, th: torch.Tensor, eps: float = EPS_DEF, sig: float = 0.0) -> torch.Tensor:
        lq = d.log_prob(th.transpose(0, 1)).transpose(0, 1)  # [P, K]
        return torch.logaddexp(lq + math.log1p(-eps), log_prior_s(th, sig) + math.log(eps))


def _sobol_normal(P: int, K: int, J: int, seed: int, device) -> torch.Tensor:
    """P independent scrambled Sobol sets of K points in R^J, mapped to N(0, I) -> [P, K, J]."""
    out = []
    for p in range(P):
        u = torch.quasirandom.SobolEngine(J, scramble=True, seed=seed * 100003 + p).draw(K).clamp(1e-6, 1 - 1e-6)
        out.append(math.sqrt(2) * torch.erfinv(2 * u - 1))
    return torch.stack(out).to(device)
