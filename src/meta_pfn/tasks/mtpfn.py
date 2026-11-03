"""MTPFN baseline (issue #20, notebook 12; experimental/mtpfn). One member = one mode:
- `icm_train`: train an MTPFN on the paper's robust ICM prior (task.p, task.head); artifact model.pt; table `train`;
- `icm_ref`: exact GP references on the fixed ICM evaluation datasets; tables `nll`, `ess`;
- `icm_eval`: evaluate MTPFN checkpoints (task.ckpts = "name:path,...") on the same datasets; table `nll`;
- `air_train`: train an MTPFN on the AIR prior (B = source, A = target); artifact model.pt; table `train`;
- `air_eval`: evaluate it on pilot 3's evaluation tasks (task.ckpts = "mtpfn:path"); table `nll`;
- `air_diag`: diagnostics on the same tasks (experimental/mtpfn/diagnose.py; notebook 13).
"""

import time

import torch
from omegaconf import DictConfig

from ..evaluation import metric_names as M
from ..experimental.air import decoder as AD
from ..experimental.mtpfn import train as Tr
from ..experimental.mtpfn.model import MTPFN
from ..tracking.sink import Run
from ..utils.seeding import seed_everything


def _borders(t, device):
    if t.prior == "air":  # AIR's bar support, so NLLs join pilot 3's
        return torch.linspace(AD.Y_LO, AD.Y_HI, AD.N_BINS + 1, device=device)
    return torch.linspace(t.y_lo, t.y_hi, t.n_bins + 1, device=device)


def _model(t, device, head=None):
    return MTPFN(t.d_max if t.prior == "icm" else 1, _borders(t, device), d_model=t.dm, heads=t.heads,
                 n_pairs=t.n_pairs, d_ff=t.d_ff, head=head or t.head).to(device)  # fmt: skip


def _load(t, device):
    out = {}
    for item in str(t.ckpts).split(","):
        name, path = item.split(":", 1)
        head = "gauss" if "gauss" in name else "bar"
        m = _model(t, device, head)
        m.load_state_dict(torch.load(path, map_location=device))
        out[name] = m
    return out


def mtpfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    seed_everything(cfg.seed)
    torch.set_num_threads(t.threads)
    dev = t.device if torch.cuda.is_available() or t.device == "cpu" else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.time()
    if t.mode in ("icm_train", "air_train"):
        m = _model(t, dev)
        log(f"MTPFN parameters: {sum(p.numel() for p in m.parameters())}")
        g = torch.Generator(device=dev).manual_seed(cfg.seed + 99_991)
        if t.mode == "icm_train":
            draw = Tr.icm_sampler(t, dev)
            val = Tr.icm_batch(Tr.I.sample_icm(256, t.t_eval, t.d_max, t.q, g, dev, p=t.p, n_tot=60, ard=t.ard))
        else:
            draw = Tr.air_sampler(t, dev)
            val = Tr.air_batch(Tr.sample_tasks(Tr.PriorCfg(J=2, g_norm=t.g_norm), 256, 5, g, dev))
        hist = Tr.train(m, draw, t.steps, dev, seed=cfg.seed, batch=t.batch, lr=t.lr, warmup=t.warmup, clip=t.clip,
                        every=t.every, val=val, metrics=run.log_metrics, log=log)  # fmt: skip
        run.log_table("train", hist)
        torch.save(m.state_dict(), run.artifacts / "model.pt")
        score = float(hist.val_nll.iloc[-1])
    elif t.mode == "icm_ref":
        nll, ess = Tr.icm_references(t, dev, log=log)
        run.log_table("nll", nll)
        run.log_table("ess", ess)
        score = float(nll[nll.method == "bayes"].nll.mean())
    elif t.mode == "icm_eval":
        models = _load(t, dev)
        nll = Tr.icm_evaluate(models, t, dev, log=log)
        run.log_table("nll", nll)
        score = float(nll[nll.method == next(iter(models))].nll.mean())
    elif t.mode == "air_diag":
        from ..experimental.mtpfn.diagnose import air_diagnose

        (m,) = _load(t, dev).values()
        ao = None
        if t.aonly_ckpt:
            ao = AD.Decoder(D=128, layers=6).to(dev)
            ao.load_state_dict(torch.load(t.aonly_ckpt, map_location=dev))
        tabs = air_diagnose(m, t, dev, aonly=ao, log=log)
        for k, v in tabs.items():
            run.log_table(k, v)
        c = tabs["ctrl"]
        score = float(c[(c.input == "b") & (c.n_a == 5)].nll.mean())
    elif t.mode == "air_eval":
        (m,) = _load(t, dev).values()
        nll = Tr.air_evaluate(m, t, dev, log=log)
        run.log_table("nll", nll)
        score = float(nll[(min(t.eval_j) == nll.J) & (nll.n_a == 5)].nll.mean())
    else:
        raise ValueError(f"unknown task.mode {t.mode}")
    run.log_summary(**{M.MTPFN_SCORE: score, M.TIME_PREFIX + "total_s": time.time() - t0})
    return score
