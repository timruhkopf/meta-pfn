"""AIR: amortised importance sampling over an explicit transform latent (issue #19, notebook 10;
docs/a2b_registration_proposal_2026-11-02.md). One member = one mode:
- `decoder`: train D_ψ (+ null head) on the J-mixture; artifact model.pt; table `train`;
- `proposal`: train q_φ for task.J; artifact model.pt; table `train`;
- `eval`: load the decoder (task.decoder_ckpt) and the proposals (task.proposal_ckpts, "J:path,J:path"), run the V0–V5
  evaluation; tables `nll`, `diag`, `prior` (V0: T calibration);
- `diagnose`: post hoc failure analysis on the same checkpoints; table `pfail` (experimental/air/diagnose.py).
"""

import time

import pandas as pd
import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..experimental.air import prior as Pr
from ..experimental.air import train as Tr
from ..experimental.air.decoder import Decoder
from ..experimental.air.diagnose import diagnose
from ..experimental.air.proposal import Proposal
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def air(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.time()
    if t.mode == "decoder":
        dec = Decoder(D=t.dm, layers=t.layers).to(dev)
        hist = Tr.train_decoder(dec, t.steps, dev, seed=cfg.seed, batch=t.batch, lr=t.lr, j_mix=list(t.j_mix),
                                h_kind=t.h_kind, every=t.every, metrics=run.log_metrics, log=log)  # fmt: skip
        run.log_table("train", hist)
        torch.save(dec.state_dict(), run.artifacts / "model.pt")
        score = float(hist.val_teacher_n5.iloc[-1])
    elif t.mode == "proposal":
        q = Proposal(t.J).to(dev)
        hist = Tr.train_proposal(q, t.J, t.steps, dev, seed=cfg.seed, batch=t.batch, lr=t.lr, h_kind=t.h_kind,
                                 every=t.every, metrics=run.log_metrics, log=log)  # fmt: skip
        run.log_table("train", hist)
        torch.save(q.state_dict(), run.artifacts / "model.pt")
        score = float(hist.val_nll_n5.iloc[-1])
    elif t.mode in ("eval", "diagnose"):
        dec = Decoder(D=t.dm, layers=t.layers).to(dev)
        dec.load_state_dict(torch.load(t.decoder_ckpt, map_location=dev))
        props = {}
        for item in str(t.proposal_ckpts).split(","):
            J, path = item.split(":", 1)
            q = Proposal(int(J)).to(dev)
            q.load_state_dict(torch.load(path, map_location=dev))
            props[int(J)] = q
        if t.mode == "diagnose":
            tab = diagnose(dec, props, dev, n_tasks=t.n_eval, seed=t.eval_seed, ks=tuple(t.diag_ks),
                           fid_tasks=t.fid_tasks, log=log)  # fmt: skip
            run.log_table("pfail", tab)
            run.log_summary(**{M.TIME_PREFIX + "total_s": time.time() - t0})
            return float(len(tab))
        rows = []
        for J in props:  # V0: the transform family
            g = torch.Generator().manual_seed(J)
            th = torch.randn(2000, J, generator=g)
            x = torch.linspace(0, 1, 201).expand(2000, -1)
            s = Pr.s_t_for(J)
            tx = Pr.flow(x, th, s)
            inv = Pr.flow(tx, th, s, inverse=True)
            end_err = float(tx[:, [0, -1]].sub(torch.tensor([0.0, 1.0])).abs().max())
            rows.append({"J": J, "s_t": s, "mean_shift": float((tx - x).abs().mean()),
                         "monotone": bool((tx.diff(dim=1) >= 0).all()), "end_err": end_err,
                         "inv_err": float((inv - x).abs().max())})  # fmt: skip
        run.log_table("prior", pd.DataFrame(rows))
        nll, diag = Tr.evaluate(dec, props, dev, n_tasks=t.n_eval, seed=t.eval_seed, ks=tuple(t.ks), k_ref=t.k_ref,
                                k_exact=t.k_exact, fid_tasks=t.fid_tasks, log=log)  # fmt: skip
        run.log_table("nll", nll)
        run.log_table("diag", diag)
        m = nll[(min(props) == nll.J) & (nll.n_a == 5)]
        score = float(m.loc[m.method == "qis32", "nll"].mean())
    else:
        raise ValueError(f"task.mode must be decoder, proposal, eval or diagnose, got {t.mode}")
    run.log_summary(**{M.AIR_SCORE: score, M.TIME_PREFIX + "total_s": time.time() - t0})
    return score
