"""Example pipeline (replace it): linear regression by gradient descent on synthetic data.

Shows the pattern every pipeline follows: typed config in, results out through the run sink
(series via log_metric, scalars via log_summary, tables via log_table), nothing printed.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..data.splits import check_split
from ..evaluation import metric_names as M
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def toy_regression(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    if t.steps < 1:
        raise ValueError(f"task.steps must be >= 1, got {t.steps}")
    check_split(t.eval_split, cfg.stage)
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    w = rng.normal(size=t.dim)
    x = rng.normal(size=(2 * t.n, t.dim))
    y = x @ w + t.noise * rng.normal(size=2 * t.n)
    x_train, y_train, x_eval, y_eval = x[: t.n], y[: t.n], x[t.n :], y[t.n :]

    w_hat = np.zeros(t.dim)
    for step in range(1, t.steps + 1):
        resid = x_train @ w_hat - y_train
        w_hat -= t.lr * 2 * x_train.T @ resid / t.n
        if step % t.log_every == 0 or step == t.steps:
            run.log_metric(M.TRAIN_MSE, float(np.mean(resid**2)), step)

    eval_mse = float(np.mean((x_eval @ w_hat - y_eval) ** 2))
    run.log_summary(
        **{
            M.FINAL_TRAIN_MSE: float(np.mean((x_train @ w_hat - y_train) ** 2)),
            M.FINAL_EVAL_MSE: eval_mse,
            M.FINAL_W_ERROR: float(np.linalg.norm(w_hat - w)),
        }
    )
    run.log_table("coefficients", pd.DataFrame({"true": w, "estimate": w_hat}))
    return eval_mse
