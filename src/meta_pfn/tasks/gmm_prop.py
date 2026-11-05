"""GMM-token propagation (issue #23, notebook 18; experimental/gmmprop). One member = one dimension d: train the
GMM-token PFN on the GP displacement prior, then evaluate it with the closed-form references on the fixed
evaluation tasks. Artifact model.pt; tables `train`, `eval`."""

import time

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def gmm_prop(cfg: DictConfig, run: Run) -> float:
    from ..experimental.gmmprop import train as T
    from ..experimental.gmmprop.model import GMMPFN

    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.time()
    model = GMMPFN(t.d, D=t.dm, layers=t.layers, heads=t.heads, dff=t.dff, K=t.k_out).to(dev)
    log(f"GMMPFN d={t.d} parameters: {sum(p.numel() for p in model.parameters())}")
    hist = T.train(model, t.d, t.steps, dev, seed=cfg.seed, batch=t.batch, q=t.q, n_max=t.n_max, lr=t.lr,
                   warmup=t.warmup, clip=t.clip, every=t.every, metrics=run.log_metrics, log=log)  # fmt: skip
    run.log_table("train", hist)
    torch.save(model.state_dict(), run.artifacts / "model.pt")
    ev = T.evaluate(model, t.d, dev, n_tasks=t.n_eval, q=t.q, seed=t.eval_seed, log=log)
    run.log_table("eval", ev)
    piv = ev.groupby(["n", "method"]).nll.mean().unstack()
    gap = (piv["model"] - piv["oracle"]) / t.d
    for n, v in gap.items():
        run.log_summary(**{f"{M.GMM_PROP_GAP}/n{n}": float(v)})
    score = float(gap.max())
    run.log_summary(**{M.GMM_PROP_SCORE: score, M.TIME_PREFIX + "total_s": time.time() - t0})
    return score
