"""AIR-CIS (issue #22, notebook 17; experimental/air/cis). One member = one mode:
- `dec_train`: the transported decoder D on the CIS prior (task.tfeat: with the T′ feature, false = C8 ablation);
  plateau rule; artifact model.pt; table `train`;
- `mt_train`: MTPFN (#20's architecture) on the CIS prior, or with task.aonly the clean A-only PFN; artifact
  model.pt; table `train`;
- `eval`: every method on the fixed evaluation tasks (see experimental/air/cis/evaluate.py); tables `nll`, `diag`,
  `tasks`.
"""

import time

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def _mtpfn(t, device):
    from ..experimental.air import decoder as AD
    from ..experimental.mtpfn.model import MTPFN

    b = torch.linspace(AD.Y_LO, AD.Y_HI, AD.N_BINS + 1, device=device)
    return MTPFN(1, b, d_model=t.mt_dm, heads=t.mt_heads, n_pairs=t.mt_pairs, d_ff=t.mt_dff, head="bar").to(device)


def air_cis(cfg: DictConfig, run: Run) -> float:
    from ..experimental.air.cis import decoder as CD
    from ..experimental.air.cis import mt
    from ..experimental.air.cis import prior as C
    from ..experimental.mtpfn import train as Tr

    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.time()
    if t.mode == "dec_train":
        dec = CD.CisDecoder(D=t.dm, layers=t.layers, tfeat=t.tfeat).to(dev)
        log(f"decoder parameters: {sum(p.numel() for p in dec.parameters())} (tfeat={t.tfeat})")
        hist = CD.train(dec, t.max_steps, dev, seed=cfg.seed, batch=t.batch, lr=t.lr, clip=t.clip, every=t.every,
                        min_steps=t.min_steps, metrics=run.log_metrics, log=log)  # fmt: skip
        run.log_table("train", hist)
        torch.save(dec.state_dict(), run.artifacts / "model.pt")
        score = float(hist.val_teacher_n5.iloc[-1])
    elif t.mode == "mt_train":
        m = _mtpfn(t, dev)
        log(f"MTPFN parameters: {sum(p.numel() for p in m.parameters())} (aonly={t.aonly})")
        vt = C.sample_tasks(C.CisCfg(), 256, 5, torch.Generator().manual_seed(cfg.seed + 4242), dev)
        val = mt.ao_batch(vt) if t.aonly else mt.mt_batch(vt)
        hist = Tr.train(m, mt.sampler(t.aonly), t.mt_steps, dev, seed=cfg.seed, batch=t.mt_batch, lr=t.mt_lr,
                        warmup=t.mt_warmup, clip=t.mt_clip, every=t.every * 2, val=val, metrics=run.log_metrics,
                        log=log)  # fmt: skip
        run.log_table("train", hist)
        torch.save(m.state_dict(), run.artifacts / "model.pt")
        score = float(hist.val_nll.iloc[-1])
    elif t.mode == "eval":
        from ..experimental.air.cis import evaluate as E

        tabs = E.run(t, dev, log=log)
        for k, v in tabs.items():
            run.log_table(k, v)
        nll = tabs["nll"]
        score = float(nll[(nll.method == "cis_r3_k128") & (nll.n_a == 3)].nll.mean())
    else:
        raise ValueError(f"unknown task.mode {t.mode}")
    run.log_summary(**{M.AIR_CIS_SCORE: score, M.TIME_PREFIX + "total_s": time.time() - t0})
    return score
