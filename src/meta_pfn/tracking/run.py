"""tracked_run: wraps every pipeline invocation (scratch or tracked) in a run sink with end markers."""

import json
import os
import shlex
import socket
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ..utils.env import PACKAGE, var
from .params import config_hash, semantic_params
from .provenance import DirtyTreeError, git_state
from .sink import Preempted, Run


def _check_member_provenance(sweep: dict) -> None:
    """A member must run the code its sweep was planned at, e.g. not a commit made while it was queued."""
    if sweep.get("allow_dirty"):
        return
    state = git_state()
    if state.commit != sweep["commit"] or state.dirty:
        raise DirtyTreeError(
            f"sweep {sweep['id']} was planned at {sweep['commit'][:8]}, but the repository is at "
            f"{state.commit[:8]}{' (dirty)' if state.dirty else ''}. Run members from a checkout of that commit."
        )


def _devices() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return len([d for d in visible.split(",") if d]) if visible else 0


@contextmanager
def tracked_run(cfg: DictConfig) -> Iterator[Run]:
    run = Run(Path(HydraConfig.get().runtime.output_dir))
    for marker in ("DONE", "FAILED"):  # a resumed member starts over as STARTED
        (run.dir / marker).unlink(missing_ok=True)

    sweep_dir = cfg.tracking.sweep_dir
    sweep = json.loads((Path(sweep_dir) / "sweep.json").read_text()) if sweep_dir else {}
    if sweep:
        _check_member_provenance(sweep)
    state = git_state()
    params = semantic_params(cfg)
    previous = run.read()

    run.update(
        params=params,
        status="STARTED",
        start=previous.get("start", time.time()),  # keep the first start across requeues
        command=sweep.get("command") or shlex.join([PACKAGE, *sys.argv[1:]]),
        overrides=list(HydraConfig.get().overrides.task),
        restarts=previous.get("restarts", -1) + 1,
    )
    run.set_tags(
        experiment=cfg.experiment_name,
        stage=cfg.stage,
        sweep_id=sweep.get("id", ""),
        member=cfg.tracking.member if cfg.tracking.member is not None else "",
        config_hash=config_hash(params, cfg.tracking.replicate_keys),
        git_commit=state.commit,
        git_dirty=state.dirty,
        reportable=bool(sweep) and not sweep.get("allow_dirty") and cfg.stage in ("confirm", "final"),
        setup=var("SETUP", "unknown"),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    (run.artifacts / "config").mkdir(exist_ok=True)
    OmegaConf.save(cfg, run.artifacts / "config" / "resolved.yaml", resolve=True)

    t0 = time.time()
    try:
        yield run
    except Preempted:
        run.update(status="STARTED", wall_s=run.read().get("wall_s", 0.0) + time.time() - t0)
        raise
    except BaseException:
        (run.dir / "FAILED").write_text(traceback.format_exc())
        run.update(status="FAILED", end=time.time(), wall_s=run.read().get("wall_s", 0.0) + time.time() - t0)
        raise
    wall = run.read().get("wall_s", 0.0) + time.time() - t0
    run.update(status="DONE", end=time.time(), wall_s=wall, device_h=wall / 3600 * _devices())
    (run.dir / "DONE").touch()
