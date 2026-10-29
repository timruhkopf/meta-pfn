"""Pre-check for the cache-residual idea: how large, local and low-rank is ΔKV_l = KV_l(B | B ∪ A_inB) − KV_l(B | B)?"""

import sys

import pandas as pd
import torch
from pfn_repro.checkpoints import load

from meta_pfn.experimental.a2b.model import FrozenQB, mix_eps
from meta_pfn.experimental.a2b.prior import PairPrior, PriorCfg, apply_t

torch.manual_seed(0)
d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
n_b, P = 80, 64
pfn = load("pfn_large")[0]
qb = FrozenQB(pfn)
rows = []
for n_a in (2, 5, 10):
    inst = PairPrior(PriorCfg(d=d), P, 7 + n_a, "cpu").sample("warp", n_a, 32, n_b)
    ta = apply_t(inst.xa, inst.t_al, inst.t_be)
    with torch.no_grad():
        kv0, nf0, _ = qb.encode(inst.xb, inst.yb)
        kv1, nf1, _ = qb.encode(torch.cat([inst.xb, ta], 1), torch.cat([inst.yb, inst.ya], 1))
        tq = apply_t(inst.xq, inst.t_al, inst.t_be)
        l0 = mix_eps(qb.logpdf(qb.query(kv0, nf0, tq)[0], inst.yq))
        l1 = mix_eps(qb.logpdf(qb.query(kv1, nf1, tq)[0], inst.yq))
    gain = float((l1 - l0).mean())  # nats/query: A in B's context vs B alone, true T
    dist = torch.cdist(inst.xb, ta).min(-1).values  # [P, N_B] distance of each B point to the nearest transported A
    for li, ((k0, v0), (k1, v1)) in enumerate(zip(kv0, kv1, strict=True)):
        for name, a, b in (("K", k0, k1[:, :n_b]), ("V", v0, v1[:, :n_b])):
            D = b - a  # [P, N_B, Dm]
            e = D.pow(2).sum(-1)  # energy per B row
            tot = e.sum(-1)
            rel = float((tot.sqrt() / a.pow(2).sum((-1, -2)).sqrt()).mean())
            r = {"n_a": n_a, "layer": li, "kv": name, "rel_norm": rel, "gain_nats": gain}
            for k in (4, 8, 16):
                near = dist.argsort(-1)[:, :k]
                r[f"near{k}"] = float((e.gather(1, near).sum(-1) / tot).mean())
                r[f"best{k}"] = float((e.sort(-1, descending=True).values[:, :k].sum(-1) / tot).mean())
            sv = torch.linalg.svdvals(D).pow(2)
            cum = sv.cumsum(-1) / sv.sum(-1, keepdim=True)
            r["rank90"] = float((cum < 0.9).sum(-1).float().add(1).mean())
            rows.append(r)
pd.set_option("display.width", 200)
df = pd.DataFrame(rows)
print(f"d={d}, N_B={n_b}, {P} warp pairs; rel_norm = ||ΔKV|| / ||KV|| over B rows")
print(df.round(3).to_string(index=False))
