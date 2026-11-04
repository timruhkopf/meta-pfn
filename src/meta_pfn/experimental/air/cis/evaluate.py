"""AIR-CIS evaluation (notebook 17; log entry 2026-11-04_air-cis, §4-§5 and the addendum): every method on the same
fixed evaluation tasks (prior.eval_tasks, nested in n_A), per n_A in task.n_as.

Methods (column `method`):
- `aonly`: the clean A-only PFN (MTPFN architecture, trained on A alone; task.aonly_ckpt);
- `null`: the decoder's null head (B masked out);
- `mtpfn`: MTPFN trained on this prior (task.mtpfn_ckpt);
- `cis_r{R}_k{K}`: CIS-U with the GP surrogate on B and the decoder as target;
- `isref`: prior importance sampling with the same decoder, K = task.k_ref;
- `teacher`: the decoder with the true θ (and J);
- `oracle`: the exact posterior predictive (true g and σ_A; cis/oracle.py), with its MC error;
- `cis_exact_r3_k128` (task.exact_cis): CIS with the true g and the exact likelihood (proposal quality alone).

Tables: `nll` (method, n_a, task, nll = −mean log density over the queries), `query` (per query: x, lp, cdf; for
the leak audit and calibration), `diag` (per CIS run and task: ESS, rounds, branch hit, q-mass on the truth, p(J),
time), `tasks` (J, number of extrema, h_id, σ_A, a, b)."""

import time

import numpy as np
import pandas as pd
import torch

from ..decoder import bar_logpdf
from . import bsurrogate as BS
from . import core as CO
from . import decoder as CD
from . import fit as FT
from . import mt
from . import oracle as O
from . import prior as C


def _load(m, path, dev):
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval()


def _cdf_from_logits(lg, y):
    return CO.bucket_cdf(lg.float().softmax(-1), y)


@torch.no_grad()
def _pfn_lp_cdf(model, batch, chunk: int = 64):
    x, y, mask, xq, yq, _ = batch
    lps, cdfs = [], []
    for s in range(0, x.shape[0], chunk):
        sl = slice(s, s + chunk)
        lg = model(x[sl], y[sl], mask[sl], xq[sl]).float()
        lps.append(bar_logpdf(lg, yq[sl]))
        cdfs.append(_cdf_from_logits(lg, yq[sl]))
    return torch.cat(lps), torch.cat(cdfs)


@torch.no_grad()
def _dec_lp_cdf(dec, t: C.CisTasks, use_b: bool, lat=None, J=None, chunk: int = 128):
    lps, cdfs = [], []
    for s in range(0, t.xa.shape[0], chunk):
        idx = torch.arange(s, min(s + chunk, t.xa.shape[0]), device=t.xa.device)
        sub = t.sub(idx)
        xb_t, ld = sub.b_in_a(None if lat is None else lat[idx], None if J is None else J[idx])
        lg, _ = dec(xb_t, sub.yb, sub.xa, sub.ya, sub.xq, ld=ld, use_b=use_b)
        lps.append(bar_logpdf(lg.float(), sub.yq))
        cdfs.append(_cdf_from_logits(lg, sub.yq))
    return torch.cat(lps), torch.cat(cdfs)


@torch.no_grad()
def isref(t: C.CisTasks, target, K: int, g: torch.Generator, spec: FT.Spec, chunk_tasks: int = 4) -> dict:
    """Prior importance sampling: J ~ p(J), free latents ~ N(0, I); weights = the target's log p̂(A | B, θ)."""
    P, dev = t.xa.shape[0], t.xa.device
    jv = torch.tensor(spec.js, device=dev)
    pj = torch.tensor(spec.pj, device=dev)
    lps, cdfs, esss = [], [], []
    for s in range(0, P, chunk_tasks):
        tid = torch.arange(s, min(s + chunk_tasks, P), device=dev)
        p = len(tid)
        tidx = tid.repeat_interleave(K)
        Js = jv[torch.multinomial(pj, p * K, replacement=True, generator=g)]
        lat = torch.randn(p * K, C.N_LAT, generator=g, device=dev)
        mask = torch.zeros_like(lat)
        for Jv in spec.js:
            m = torch.zeros(C.N_LAT, device=dev)
            m[spec.idx(Jv)] = 1
            mask = torch.where((Js == Jv)[:, None], m, mask)
        res = target(tidx, lat * mask, Js, g)
        lw = res["loglik"].reshape(p, K)
        W = lw.softmax(-1)
        lps.append(torch.logsumexp(torch.log(W.clamp_min(1e-30))[..., None] + res["lp"].reshape(p, K, -1), 1))
        cdfs.append((W[..., None] * res["cdf"].reshape(p, K, -1)).sum(1))
        esss.append(1 / (W**2).sum(-1))
    return {"lp": torch.cat(lps), "cdf": torch.cat(cdfs), "ess": torch.cat(esss)}


def _branch_stats(t: C.CisTasks, ext_x: list, out: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per task: (branch hit = any sample with the true branch combination, weight mass on it)."""
    P, K = out["W"].shape
    lat, J = out["lat"], out["J"]
    xs = t.xa.sort(1).values  # run() works on sorted A; labels are order-free per point, compare sorted
    u_true = C.transform(xs, t.lat, t.J)
    u_s = C.transform(xs[:, None].expand(P, K, -1), lat, J)
    hit, mass = np.zeros(P), np.zeros(P)
    for j in range(P):
        e = ext_x[j].to(xs.device)
        bt, bs = C.branch_label(e, u_true[j]), C.branch_label(e, u_s[j])
        ok = (bs == bt[None]).all(-1)
        hit[j] = float(ok.any())
        mass[j] = float((out["W"][j] * ok).sum())
    return hit, mass


def run(tc, dev, log=print) -> dict:
    """tc: the task config (configs/task/air_cis.yaml). -> {"nll", "query", "diag", "tasks"} DataFrames."""
    from ....tasks.air_cis import _mtpfn

    spec = FT.Spec()
    T_all = C.eval_tasks(tc.n_eval, tc.eval_seed, dev)
    n_ext, ext_x = C.extrema(T_all)
    tasks = pd.DataFrame({
        "task": np.arange(tc.n_eval), "J": T_all.J.cpu().numpy(), "n_ext": n_ext, "h_id": T_all.hid.cpu().numpy(),
        "sig_a": T_all.sig_a.cpu().numpy(), "a": T_all.hp[:, 0].cpu().numpy(), "b": T_all.hp[:, 1].cpu().numpy(),
    })  # fmt: skip
    dec = _load(CD.CisDecoder(D=tc.dm, layers=tc.layers, tfeat=True).to(dev), tc.decoder_ckpt, dev)
    models = {}
    if tc.mtpfn_ckpt:
        models["mtpfn"] = (_load(_mtpfn(tc, dev), tc.mtpfn_ckpt, dev), mt.mt_batch)
    if tc.aonly_ckpt:
        models["aonly"] = (_load(_mtpfn(tc, dev), tc.aonly_ckpt, dev), mt.ao_batch)
    dec_nf = _load(CD.CisDecoder(D=tc.dm, layers=tc.layers, tfeat=False).to(dev), tc.decoder_nf_ckpt, dev) \
        if tc.get("decoder_nf_ckpt") else None  # fmt: skip
    gp = BS.fit(T_all.xb, T_all.yb)
    tab = C.prior_tables()
    tab = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in tab.items()}
    rows_nll, rows_q, rows_d = [], [], []
    xq_np = T_all.xq.cpu().numpy()

    def add(method, n, lp, cdf, extra=None):
        lp_np, cdf_np = lp.float().cpu().numpy(), cdf.float().cpu().numpy()
        r = {"method": method, "n_a": n, "task": np.arange(tc.n_eval), "nll": -lp_np.mean(1)}
        if extra:
            r.update(extra)
        rows_nll.append(pd.DataFrame(r))
        Q = lp_np.shape[1]
        rows_q.append(pd.DataFrame({
            "method": method, "n_a": n, "task": np.repeat(np.arange(tc.n_eval), Q),
            "q": np.tile(np.arange(Q), tc.n_eval),
            "x": xq_np.reshape(-1), "lp": lp_np.reshape(-1), "cdf": cdf_np.reshape(-1),
        }))  # fmt: skip

    for n in tc.n_as:
        t = T_all.head(n)
        log(f"n_A = {n}")
        t0 = time.time()
        for name, (m, adapt) in models.items():
            lp, cdf = _pfn_lp_cdf(m, adapt(t))
            add(name, n, lp, cdf)
        lp, cdf = _dec_lp_cdf(dec, t, use_b=False)
        add("null", n, lp, cdf)
        lp, cdf = _dec_lp_cdf(dec, t, use_b=True)
        add("teacher", n, lp, cdf)
        if dec_nf is not None:
            lp, cdf = _dec_lp_cdf(dec_nf, t, use_b=True)
            add("teacher_nf", n, lp, cdf)
        log(f"  PFNs and decoder heads: {time.time() - t0:.0f} s")
        t0 = time.time()
        o = O.oracle(t, spec, K=tc.k_oracle, seed=tc.eval_seed + n, chunk=tc.oracle_chunk)
        add("oracle", n, o["lp"], o["cdf"], {"mc": o["mc"].cpu().numpy()})
        log(f"  oracle: {time.time() - t0:.0f} s, MC error mean {float(o['mc'].mean()):.4f}")
        for i, Jv in enumerate(spec.js):
            rows_d.append(pd.DataFrame({"method": "oracle", "n_a": n, "task": np.arange(tc.n_eval),
                                        "J_val": Jv, "pJ": o["pJ"][:, i].cpu().numpy()}))  # fmt: skip
        target = CO.DecoderTarget(dec, t, n_perm=tc.n_perm)
        g = torch.Generator(device=dev).manual_seed(tc.eval_seed * 10 + n)
        t0 = time.time()
        ir = isref(t, target, tc.k_ref, g, spec)
        add("isref", n, ir["lp"], ir["cdf"], {"ess": ir["ess"].cpu().numpy()})
        log(f"  IS-ref (K = {tc.k_ref}): {time.time() - t0:.0f} s, ESS median {float(ir['ess'].median()):.1f}")
        runs = [(R, K, "dec") for R in tc.rounds for K in tc.ks]
        if tc.get("exact_cis", True):
            runs.append((3, 128, "exact"))
        for R, K, kind in runs:
            name = f"cis_r{R}_k{K}" if kind == "dec" else f"cis_exact_r{R}_k{K}"
            sur = gp if kind == "dec" else BS.Exact(t.g)
            t0 = time.time()
            lps, cdfs, diag = [], [], []
            for s in range(0, tc.n_eval, tc.cis_chunk):
                idx = torch.arange(s, min(s + tc.cis_chunk, tc.n_eval), device=dev)
                sub = t.sub(idx)
                tg = CO.DecoderTarget(dec, sub, n_perm=tc.n_perm) if kind == "dec" else CO.ExactTarget(sub, C.PI_ID)
                out = CO.run(sub, sur.at(idx), tg, spec, R=R, K=K, g=g, tab=tab)
                lps.append(out["lp"])
                cdfs.append(out["cdf"])
                hit, mass = _branch_stats(sub, [ext_x[int(j)] for j in idx], out)
                W = out["W"]
                d = {"task": idx.cpu().numpy(), "ess": out["ess"].cpu().numpy(), "hit": hit, "qmass": mass,
                     "log_z1": out["log_z1"].cpu().numpy(),
                     "pJ_true": (W * (out["J"] == sub.J[:, None])).sum(1).cpu().numpy()}  # fmt: skip
                for r in range(out["ess_rounds"].shape[1]):
                    d[f"ess_r{r + 1}"] = out["ess_rounds"][:, r].cpu().numpy()
                for Jv in spec.js:
                    d[f"pJ_{Jv}"] = (W * (out["J"] == Jv)).sum(1).cpu().numpy()
                diag.append(pd.DataFrame(d))
            dt = time.time() - t0
            dg = pd.concat(diag)
            dg["method"], dg["n_a"], dg["R"], dg["K"], dg["sec_per_task"] = name, n, R, K, dt / tc.n_eval
            rows_d.append(dg)
            add(name, n, torch.cat(lps), torch.cat(cdfs))
            log(f"  {name}: {dt:.0f} s, ESS/K median {float(dg.ess.median()) / K:.3f}, hit {dg.hit.mean():.3f}")
    nll = pd.concat(rows_nll, ignore_index=True)
    summ = nll.groupby(["method", "n_a"]).nll.mean().unstack()
    log(summ.round(3).to_string())
    return {"nll": nll, "query": pd.concat(rows_q, ignore_index=True), "diag": pd.concat(rows_d, ignore_index=True),
            "tasks": tasks}  # fmt: skip
