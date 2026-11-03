"""MTPFN: the multi-task PFN with hierarchical attention (Li, Daulton, Müller, Wilson, Bakshy, "Robust Transfer for
Bayesian Optimization with Prior-Data Fitted Networks", SPIGM @ NeurIPS 2025; §3.2 and Fig. 2), built from
pfn-repro's own pieces: `PFNBlock` / `MaskedMHA` (pre-LN block, train tokens self-attend, test tokens cross-attend
to train tokens only), `_pad_and_rescale_features` (variable d) and `BarDistribution`.

Layout (Fig. 2): feature encoder -> N x [intra-task encoder, inter-task encoder] -> intra-task encoder -> output layer.
- Every task t gets a learnable [TASK] token prepended to its points. The target is task 0; its [TASK] token
  carries an extra learned "target" embedding (the paper does not say how the target is marked; Fig. 2 draws it
  last; a flag keeps the model independent of T).
- Intra-task block: within a task, [TASK] + point tokens attend bidirectionally to each other (padding masked);
  the target's query tokens x* cross-attend to the target's [TASK] + point tokens (never to each other).
- Inter-task block: the T [TASK] tokens attend to each other. This is the only route between tasks.
- Output layer: on the target's query tokens; bar distribution (pfn-repro standard) or Gaussian (μ, σ²; the
  paper's Fig. 2 labels the output this way).
Paper size: 23 layers (12 intra + 11 inter), 4 heads, width 512. Here N, width and heads are config.
"""

import math

import torch
from pfn_repro.objectives.bar_distribution import BarDistribution
from pfn_repro.surrogates.pfn import PFNBlock, _pad_and_rescale_features
from torch import nn


class MTPFN(nn.Module):
    def __init__(self, x_dim: int, borders: torch.Tensor, d_model: int = 256, heads: int = 4, n_pairs: int = 4,
                 d_ff: int = 512, head: str = "bar"):  # fmt: skip
        super().__init__()
        self.x_dim, self.head_kind = x_dim, head
        self.ctx_embed = nn.Linear(x_dim + 1, d_model)  # (x, y), as pfn-repro's PFN.train_embed
        self.q_embed = nn.Linear(x_dim, d_model)  # x*, as PFN.test_x_embed
        self.q_placeholder = nn.Parameter(torch.zeros(d_model))
        self.task_tok = nn.Parameter(torch.randn(d_model) * 0.02)
        self.target_flag = nn.Parameter(torch.randn(d_model) * 0.02)
        self.intra = nn.ModuleList([PFNBlock(d_model, heads, d_ff) for _ in range(n_pairs + 1)])
        self.inter = nn.ModuleList([PFNBlock(d_model, heads, d_ff) for _ in range(n_pairs)])
        self.out_ln = nn.LayerNorm(d_model)
        self.bar_dist = BarDistribution(borders)
        self.out_head = nn.Linear(d_model, self.bar_dist.num_bars if head == "bar" else 2)

    def forward(self, x, y, mask, xq, n_features: torch.Tensor | None = None) -> torch.Tensor:
        """x [P, T, N, d], y [P, T, N], mask [P, T, N] (True = real point), xq [P, Q, d]; task 0 is the target.
        n_features [P] (real input dims; default d). -> bar logits [P, Q, G] or Gaussian (μ, log σ²) [P, Q, 2]."""
        P, T, N, d = x.shape
        nf = torch.full((P,), d, device=x.device) if n_features is None else n_features
        xs = _pad_and_rescale_features(x.reshape(P, T * N, d), self.x_dim, nf).view(P, T, N, self.x_dim)
        xq = _pad_and_rescale_features(xq, self.x_dim, nf)
        pts = self.ctx_embed(torch.cat([xs, y[..., None]], -1))
        tt = self.task_tok.expand(P, T, -1).clone()
        tt[:, 0] = tt[:, 0] + self.target_flag
        tok = torch.cat([tt[:, :, None], pts], 2)  # [P, T, 1 + N, D]
        kpm = torch.cat([mask.new_ones(P, T, 1), mask], 2)
        q = self.q_embed(xq) + self.q_placeholder
        for i in range(len(self.inter)):
            tok, q = self._intra(self.intra[i], tok, q, kpm)
            tok = self._inter(self.inter[i], tok)
        tok, q = self._intra(self.intra[-1], tok, q, kpm)
        return self.out_head(self.out_ln(q))

    @staticmethod
    def _intra(blk, tok, q, kpm):
        P, T, L, D = tok.shape
        flat = tok.reshape(P * T, L, D)
        new, _ = blk(flat, flat[:, :0], kpm.reshape(P * T, L))
        _, q = blk(tok[:, 0], q, kpm[:, 0])  # the target's queries read the target's (pre-update) tokens
        return new.view(P, T, L, D), q

    @staticmethod
    def _inter(blk, tok):
        tt, _ = blk(tok[:, :, 0], tok[:, :0, 0])
        return torch.cat([tt[:, :, None], tok[:, :, 1:]], 2)

    def log_density(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log p(y) per query: out [P, Q, *], y [P, Q] -> [P, Q]."""
        if self.head_kind == "bar":
            return -self.bar_dist(out, y)
        mu, lv = out[..., 0], out[..., 1].clamp(-14, 6)
        return -0.5 * (math.log(2 * math.pi) + lv + (y - mu) ** 2 / lv.exp())

    def mean_std(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_kind == "bar":
            return self.bar_dist.mean(out), self.bar_dist.variance(out).clamp_min(0).sqrt()
        return out[..., 0], (0.5 * out[..., 1].clamp(-14, 6)).exp()
