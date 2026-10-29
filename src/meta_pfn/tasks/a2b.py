"""A→B architecture, S3/S4 (issue #18, notebook 09): one member = one (d, h_kind, model variant).

The model is trained first (training loss and validation metrics are logged as metrics, so they reach MLflow with
`cli mlflow-sync`), then evaluated on fixed pairs. Tables: `train`, `nll` (model per step r and references, per
stratum, pair, n_A), `reg` (location / composite errors per step r).
"""

import time

import torch
from omegaconf import DictConfig
from pfn_repro.checkpoints import load

from ..evaluation import metric_names as M
from ..experimental.a2b import train as T
from ..experimental.a2b.model import A2B
from ..experimental.a2b.prior import PriorCfg
from ..tracking.sink import Run
from ..utils.seeding import seed_everything

VARIANTS = {  # task.variant -> overrides of the task config (`sup: auto` = loc for h = id, func otherwise)
    "full": {},
    "r0": {"r": 0},  # init only: Perceiver + A-encoder + learned queries, no evidence blocks
    "k1": {"k": 1},  # one particle
    "nosup": {"sup": "none"},  # NLL only, no privileged term
    "context": {"predictive": "context"},  # last step: PFN re-encoded on B ∪ A_inB per particle
}


def a2b(cfg: DictConfig, run: Run) -> float:
    t = dict(cfg.task) | VARIANTS[cfg.task.variant]
    if t["sup"] == "auto":
        t["sup"] = "loc" if t["h_kind"] == "id" else "func"
    t = DictConfig(t)
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    pfn = load(t.pfn, device=dev)[0].eval()
    pc = PriorCfg(d=t.d, t_sd=t.t_sd, h_kind=t.h_kind, e_min=t.e_min, h_sd=t.h_sd)
    model = A2B(pfn, t.d, D=t.dm, M=t.m, K=t.k, R=t.r, learn_h=t.h_kind != "id", J=t.j, predictive=t.predictive)
    model = model.to(dev)
    t0 = time.time()
    hist = T.train(
        model, pc, t.steps, dev, seed=cfg.seed, batch=t.batch, n_b=t.n_b, n_q=t.n_q, n_a_max=t.n_a_max, lr=t.lr,
        sup=t.sup, sup_w=t.sup_w, every=t.every, metrics=run.log_metrics, log=lambda m: print(m, flush=True),
    )  # fmt: skip
    run.log_table("train", hist)
    torch.save(model.state_dict(), run.artifacts / "model.pt")
    t1 = time.time()
    nll, reg = T.evaluate(model, pfn, pc, n_pairs=t.n_eval, n_q=t.n_q, n_b=t.n_b, seed=t.eval_seed, device=dev,
                          n_hyp=t.n_hyp, refs=t.refs)  # fmt: skip
    run.log_table("nll", nll)
    run.log_table("reg", reg)
    m = nll[(nll.stratum == "warp") & (nll.n_a == 5)]
    score = float(m.loc[(m.method == "model") & (m.r == t.r), "nll"].mean())
    run.log_summary(
        **{M.A2B_NLL_N5: score, M.TIME_PREFIX + "train_s": t1 - t0, M.TIME_PREFIX + "eval_s": time.time() - t1}
    )
    return score
