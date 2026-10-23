"""Models that read raw (A, B) and predict A's queries (brainstorm §5.1–5.2, §5.7; issue #12).

- `TwoCtxPFN` (M0): a PFN whose context tokens carry a role (0 = A, 1 = B, 2 = privileged B_inA); queries attend to
  the context with pfn-repro's PFNBlock. Batteries included: trained with B dropped in a fraction of batches, so
  "the same model, A only" is its own baseline. Privileged tokens are LUPI contexts (training only).
- `RegE2E` (M3, end to end): the registration back half started from the identity field (x̂, ŷ) = (u, y_B), refined
  by a weight-tied GRU from A's local evidence, and read out by the *frozen* A-only PFN (notebook 04 diagnosis 1).
  Three slots: refined field, raw identity field, null (A only); weights from each slot's leave-one-out evidence
  on A's points (diagnosis 2), w ∝ exp(b_k + τ E_k). The refinement gets a Gauss–Newton direction feature for
  position errors (diagnosis 4): with local field slope s and local residual r of A's points against the field's
  linear interpolation, the shift that explains r is -r s / |s|².
"""

import torch
from pfn_repro.objectives.bar_distribution import BarDistribution, uniform_bin_borders
from pfn_repro.surrogates.pfn import PFN, PFNBlock
from torch import nn

from ..regpfn.model import CrossBlock, Fourier, sqdist


class TwoCtxPFN(nn.Module):
    def __init__(self, d: int, dm: int = 256, heads: int = 4, layers: int = 6, d_ff: int = 512, n_bins: int = 64):
        super().__init__()
        self.embed = nn.Linear(d + 1, dm)
        self.role = nn.Embedding(3, dm)
        self.q_embed = nn.Linear(d, dm)
        self.q_ph = nn.Parameter(torch.zeros(dm))
        self.blocks = nn.ModuleList([PFNBlock(dm, heads, d_ff) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, n_bins))
        self.bar = BarDistribution(uniform_bin_borders(n_bins))

    def forward(self, xc, yc, role, mask, xq) -> torch.Tensor:
        """xc [P, N, d], yc [P, N], role [P, N] long, mask [P, N] True = real, xq [P, Q, d] -> logits [P, Q, bins]."""
        c = self.embed(torch.cat([xc, yc[..., None]], -1)) + self.role(role)
        q = self.q_embed(xq) + self.q_ph
        for blk in self.blocks:
            c, q = blk(c, q, mask)
        return self.out(q)

    def nll(self, logits, yq) -> torch.Tensor:
        return self.bar(logits, yq)


class Refiner(nn.Module):
    """Weight-tied refinement of the field (x̂_i, ŷ_i) of every B token, from the identity seed."""

    def __init__(self, d: int, dm: int = 128, heads: int = 4, max_step: float = 0.1, gn: bool = True):
        super().__init__()
        self.d, self.dm, self.max_step, self.gn = d, dm, max_step, gn
        self.phi = Fourier(d)
        self.a_in = nn.Linear(self.phi.out + 1, dm)
        self.b_in = nn.Linear(2 * self.phi.out + 2, dm)
        self.lookup = CrossBlock(dm, heads, init_len=0.05)
        n_feat = 2 + d + (2 * d if gn else 0)  # ry, tanh(e), rx [, slope, gn step]
        self.gru = nn.GRUCell(dm + self.phi.out + 1 + n_feat, dm)
        self.delta = nn.Sequential(nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, d + 1))
        last = self.delta[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.log_len_e = nn.Parameter(torch.log(torch.tensor(0.05)))
        self.log_len_h = nn.Parameter(torch.log(torch.tensor(0.05)))
        self.log_len_s = nn.Parameter(torch.log(torch.tensor(0.05)))

    def slope(self, ub, x_hat, y_hat, b_mask) -> torch.Tensor:
        """Local linear slope dŷ/dx̂ of the field, from B neighbours in B's own frame -> [P, Nb, d]."""
        w = torch.softmax((-sqdist(ub, ub) / (2 * self.log_len_s.exp() ** 2)).masked_fill(~b_mask[:, None], -1e9), -1)
        mx, my = w @ x_hat, w @ y_hat[..., None]
        dx = x_hat[:, None] - mx[:, :, None]  # [P, Nb, Nb, d]
        dy = y_hat[:, None, :, None] - my[:, :, None]  # [P, Nb, Nb, 1]
        sxx = torch.einsum("pij,pijk,pijl->pikl", w, dx, dx) + 1e-4 * torch.eye(self.d, device=ub.device)
        sxy = torch.einsum("pij,pijk,pijl->pikl", w, dx, dy)
        return torch.linalg.solve(sxx, sxy)[..., 0]

    def forward(self, xa, ya, a_mask, ub, yb, b_mask, n_iter: int):
        """-> list of (x̂, ŷ) per iteration (index 0 = the identity seed), final evidence mass e [P, Nb]."""
        P, Nb, _ = ub.shape
        x_hat, y_hat = ub, yb
        a_tok = self.a_in(torch.cat([self.phi(xa), ya[..., None]], -1))
        h = torch.zeros(P * Nb, self.dm, device=ub.device)
        smooth = torch.softmax(
            (-sqdist(ub, ub) / (2 * self.log_len_h.exp() ** 2)).masked_fill(~b_mask[:, None], -1e9), -1
        )
        hist, e = [(x_hat, y_hat)], torch.zeros(P, Nb, device=ub.device)
        for _ in range(n_iter):
            b_tok = self.b_in(torch.cat([self.phi(ub), yb[..., None], self.phi(x_hat), y_hat[..., None]], -1))
            look = self.lookup(b_tok, a_tok, x_hat, xa, a_mask)
            k = torch.exp(-sqdist(x_hat, xa) / (2 * self.log_len_e.exp() ** 2)) * a_mask[:, None, :]
            e = k.sum(-1)
            kn = k / e[..., None].clamp_min(1e-6)
            off = xa[:, None, :, :] - x_hat[:, :, None, :]  # [P, Nb, Na, d]
            rx = (kn[..., None] * off).sum(2)
            feats = [look, self.phi(x_hat), y_hat[..., None], torch.tanh(e)[..., None], rx]
            if self.gn:
                s = self.slope(ub, x_hat, y_hat, b_mask)
                pred = y_hat[..., None] + (off * s[:, :, None, :]).sum(-1)  # field's linear interpolation at x_a
                ry = (kn * (ya[:, None, :] - pred)).sum(-1, keepdim=True)
                gn = (-ry * s / (s.pow(2).sum(-1, keepdim=True) + 0.01)).clamp(-0.3, 0.3)
                feats += [ry, s.clamp(-10, 10) / 10, gn]
            else:
                feats.insert(3, (kn * (ya[:, None, :] - y_hat[..., None])).sum(-1, keepdim=True))
            h = self.gru(torch.cat(feats, -1).reshape(P * Nb, -1), h)
            step = self.max_step * torch.tanh(self.delta(h).view(P, Nb, -1))
            delta = smooth @ (torch.tanh(e)[..., None] * step)
            x_hat, y_hat = x_hat + delta[..., : self.d], y_hat + delta[..., self.d]
            hist.append((x_hat, y_hat))
        return hist, e


class RegE2E(nn.Module):
    SLOTS = ("refined", "raw", "null")

    def __init__(self, d: int, pfn: PFN, n_iter: int = 8, gn: bool = True, max_step: float = 0.1):
        super().__init__()
        self.d, self.n_iter = d, n_iter
        self.refiner = Refiner(d, gn=gn, max_step=max_step)
        self.pfn = pfn
        for p in self.pfn.parameters():
            p.requires_grad_(False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def train(self, mode: bool = True):
        super().train(mode)
        self.pfn.eval()
        return self

    def _ctx(self, xa, ya, a_mask, xb, yb):
        """A's points plus B rendered at (xb, yb); B points outside A's window are masked."""
        if xb is None:
            return xa, ya, a_mask
        inside = ((xb >= 0) & (xb <= 1)).all(-1)
        return torch.cat([xa, xb], 1), torch.cat([ya, yb.clamp(0, 1)], 1), torch.cat([a_mask, inside], 1)

    def logp(self, xa, ya, a_mask, xb, yb, xq, yq) -> torch.Tensor:
        """-> [P, Q] log predictive density of the frozen PFN with context A ∪ B(rendered)."""
        xc, yc, m = self._ctx(xa, ya, a_mask, xb, yb)
        return -self.pfn.bar_dist(self.pfn(xc, yc, xq, train_key_padding_mask=m), yq)

    @torch.no_grad()
    def evidence(self, xa, ya, n: int, fields, k_max: int | None = None, g: torch.Generator | None = None):
        """Leave-one-out log evidence Σ_j log p(y_j | x_j, A without j, slot field) of A's first n points per slot
        -> [P, n_slots] (0 at n = 0). With k_max, a random subset of k_max points, rescaled by n / k_max."""
        P = xa.shape[0]
        if n == 0:
            return torch.zeros(P, len(fields), device=xa.device)
        idx = torch.arange(n, device=xa.device)
        if k_max is not None and n > k_max:
            idx = torch.randperm(n, generator=g, device="cpu")[:k_max].to(xa.device)
        k = len(idx)
        xr, yr = xa[:, :n].repeat_interleave(k, 0), ya[:, :n].repeat_interleave(k, 0)
        mask = torch.ones(P * k, n, dtype=torch.bool, device=xa.device)
        mask[torch.arange(P * k), idx.repeat(P)] = False
        xq = xa[:, idx].reshape(P * k, 1, -1)
        yq = ya[:, idx].reshape(P * k, 1)
        out = []
        for f in fields:
            xb, yb = (None, None) if f is None else (f[0].repeat_interleave(k, 0), f[1].repeat_interleave(k, 0))
            out.append(self.logp(xr, yr, mask, xb, yb, xq, yq).view(P, k).sum(1) * (n / k))
        return torch.stack(out, -1)

    def forward(self, xa, ya, n: int, ub, yb, xq, yq, n_iter=None, k_max=None, g=None):
        """-> (log p per slot [P, Q, 3], log w [P, 3], field history)."""
        a_mask = torch.ones(xa.shape[0], n, dtype=torch.bool, device=xa.device)
        b_mask = torch.ones(yb.shape, dtype=torch.bool, device=yb.device)
        xc, yc = xa[:, :n], ya[:, :n]
        hist, _ = self.refiner(xc, yc, a_mask, ub, yb, b_mask, self.n_iter if n_iter is None else n_iter)
        fields = [hist[-1], (ub, yb), None]
        lp = torch.stack(
            [self.logp(xc, yc, a_mask, f[0] if f else None, f[1] if f else None, xq, yq) for f in fields], -1
        )
        ev = self.evidence(xc, yc, n, [(f[0].detach(), f[1].detach()) if f else None for f in fields], k_max, g)
        log_w = torch.log_softmax(self.bias + self.log_tau.exp() * ev, -1)
        return lp, log_w, hist

    @staticmethod
    def mixture_nll(lp, log_w) -> torch.Tensor:
        return -torch.logsumexp(lp + log_w[:, None, :], -1)
