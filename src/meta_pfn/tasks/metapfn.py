"""End-to-end transfer models (issue #12, notebook 05): one member = one training run + its evaluation.

`task.mode` = m0 (TwoCtxPFN, LUPI variant `task.priv`) or m3 (RegE2E with the frozen A-only PFN at `task.pfn_path`).
Tables: `train`, `eval` (per-pair NLL per stratum × n_A × variant), `slots` (m3: weights and field errors).
The trained weights (without the frozen PFN) are saved as `model.pt` in the run's artifacts.
"""

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..experimental.metapfn import train as T
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def metapfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    model, hist = T.train(
        t.mode,
        t.d,
        t.steps,
        dev,
        seed=cfg.seed,
        batch=t.batch,
        n_b=t.n_b,
        n_test=t.n_test,
        priv=t.priv,
        p_priv=t.p_priv,
        p_drop_b=t.p_drop_b,
        lam=t.lam,
        lupi=t.lupi,
        n_iter=t.n_iter,
        gn=t.gn,
        pfn_path=t.pfn_path,
        lr=t.lr,
        log=lambda msg: print(msg, flush=True),
    )
    run.log_table("train", hist)
    pfn = T.load_pfn(t.pfn_path, t.d, dev)
    ev, slots = T.evaluate(model, t.mode, t.d, dev, pfn, n_pairs=t.n_eval, n_b=t.n_b)
    run.log_table("eval", ev)
    if len(slots):
        run.log_table("slots", slots)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("pfn.")}, run.artifacts / "model.pt")
    m = ev.loc[(ev.n_a == 5) & ev.stratum.isin(["C00", "C10", "C01", "C11"])].groupby("variant").nll.mean()
    gain = float(m["pfn_aonly"] - m["model"])
    run.log_summary(**{M.METAPFN_GAIN_N5: gain})
    return gain
