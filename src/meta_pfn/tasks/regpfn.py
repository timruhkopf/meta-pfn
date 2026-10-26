"""Phase-1 registration experiments (docs/spec_phase1_experiments.md): one member = one training run + its evaluation.

`task.mode`:
- `aonly`: train the A-only PFN (E1) and evaluate E1 + E2 (effective-data curves) -> tables `train`, `aonly_eval`;
- `back`: train the back half with `task.seed_mode` = true (E3) or perturb (E4) and `task.n_iter` refinement
  steps, and evaluate per cell × n_A (× perturbation) -> tables `train`, `back_eval`.
The trained weights are saved as `model.pt` in the run's artifacts.
"""

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..experimental.regpfn import train as T
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def regpfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    if t.mode == "aonly":
        model, hist = T.train_aonly(t.d, t.minutes, dev, seed=cfg.seed, batch=t.batch, n_b=t.n_b, log=log)
        ev = T.eval_aonly(model, t.d, dev, n_pairs=t.n_eval, n_perm=t.n_perm)
        run.log_table("aonly_eval", ev)
        score = float(ev.loc[(ev.m == 0) & (ev.n_a == 5)].nll.mean())
    elif t.mode == "back":
        model, hist = T.train_back(
            t.d,
            t.minutes,
            dev,
            t.seed_mode,
            t.n_iter,
            t.sig_max,
            t.lupi,
            seed=cfg.seed,
            batch=t.batch,
            n_b=t.n_b,
            log=log,
            resid=t.resid,
            max_step=t.max_step,
        )
        ev = T.eval_back(model, t.d, dev, t.seed_mode, n_pairs=t.n_eval)
        run.log_table("back_eval", ev)
        score = float(ev.loc[(ev.kind == "none") & (ev.n_a == 5) & (ev.n_iter == t.n_iter)].nll.mean())
    else:
        raise ValueError(f"task.mode must be aonly or back, got {t.mode}")
    run.log_table("train", hist)
    path = run.artifacts / "model.pt"
    torch.save(model.state_dict(), path)
    run.log_summary(**{M.REGPFN_NLL_N5: score, M.TIME_PREFIX + "train_min": float(t.minutes)})
    return score
