"""The only entry point. `uv run <package> experiment=<name> ...` is a scratch run; tracked runs go
through `just run` / `just sweep`, which call this with the sweep's bookkeeping overrides."""

import hydra
from omegaconf import DictConfig, OmegaConf

from .tasks import TASKS
from .tracking.run import tracked_run
from .utils.env import load_dotenv, repo_root

OmegaConf.register_new_resolver("repo_root", lambda: str(repo_root()), replace=True)
# `experiment: null` would otherwise put default runs into runs/None/
OmegaConf.register_new_resolver("or_scratch", lambda x: "_scratch" if x in (None, "None", "null") else x, replace=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Dispatch to a pipeline; the return value is the optimized metric for Hydra sweepers."""
    load_dotenv(repo_root() / ".env", repo_root() / ".env.secrets")  # scratch runs too, not only the CLI
    with tracked_run(cfg) as run:
        return TASKS[cfg.task.name](cfg, run)


if __name__ == "__main__":
    main()
