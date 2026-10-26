"""Field-only (issue #14, notebook 07): one member = one method (`task.method` = gd | operator | optimizer).

Learned methods are trained first (training loss and validation field metrics are logged as metrics, so they reach
MLflow with `cli mlflow-sync`). Tables: `eval` (per pair: field errors, energies, prediction error; for gd also
gd_best8 / gd_oracle8 and the start / truth references), `points` and `traj` (1d examples), `train` (learned).
"""

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..experimental import fieldonly as F
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def fieldonly(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    model = None
    if t.method in ("operator", "optimizer"):
        model = F.make_model(t.method, t.n_iter, t.dm).to(dev)
        hist = F.train_learned(
            model, t.method, t.steps, dev, seed=cfg.seed, batch=t.batch, n_b=t.n_b, lr=t.lr,
            log=lambda m: print(m, flush=True), metrics=run.log_metrics, every=t.every,
        )  # fmt: skip
        run.log_table("train", hist)
        torch.save(model.state_dict(), run.artifacts / "model.pt")
    elif t.method != "gd":
        raise ValueError(f"task.method must be gd, operator or optimizer, got {t.method}")
    ev = F.evaluate(t.method, model, dev, n_pairs=t.n_eval, n_b=t.n_b, restarts=t.restarts, gd_steps=t.gd_steps)
    run.log_table("eval", ev)
    pts, trs = F.examples(t.method, model, dev, n_b=t.n_b, gd_steps=t.gd_steps)
    run.log_table("points", pts)
    run.log_table("traj", trs)
    e = ev.loc[(ev.method == t.method) & (ev.cell == "C10") & (ev.n_a == 10) & (ev.init == "random")]
    score = float((e.err_x < 0.05).mean())
    run.log_summary(**{M.FIELDONLY_FRAC_N10: score})
    return score
