"""GW-only (issue #13, notebook 06): one member = one solver (`task.method` = peyre | unrolled | gru).

Learned solvers are trained first (training CE and validation landmark metrics are logged as metrics, so they reach
MLflow with `cli mlflow-sync`). Tables: `eval` (per A point: landmark errors, π-mass, coverage; for peyre also the
identity / value baselines and the best-of-K restart), `points` and `coupling` (1d examples), `train` (learned).
"""

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..experimental import gwonly as G
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def gwonly(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    model = None
    if t.method in ("unrolled", "gru"):
        model = G.LearnedGW(h=t.h, n_iter=t.n_iter, gru=t.method == "gru").to(dev)
        hist = G.train_learned(
            model, t.steps, dev, seed=cfg.seed, batch=t.batch, n_b=t.n_b, lr=t.lr,
            log=lambda m: print(m, flush=True), metrics=run.log_metrics, every=t.every,
        )  # fmt: skip
        run.log_table("train", hist)
        torch.save(model.state_dict(), run.artifacts / "model.pt")
    elif t.method != "peyre":
        raise ValueError(f"task.method must be peyre, unrolled or gru, got {t.method}")
    ev = G.evaluate(t.method, model, dev, n_pairs=t.n_eval, n_b=t.n_b, restarts=t.restarts)
    run.log_table("eval", ev)
    pts, cps = G.examples(t.method, model, dev, n_b=t.n_b)
    run.log_table("points", pts)
    run.log_table("coupling", cps)
    e = ev.loc[(ev.method == t.method) & (ev.cell == "C10") & (ev.n_a == 5) & ev.covered]
    score = float((e.err_x < 0.05).mean())
    run.log_summary(**{M.GWONLY_FRAC_N5: score})
    return score
