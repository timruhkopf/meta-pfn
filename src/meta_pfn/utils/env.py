"""Machine settings come from environment variables, loaded from the git-ignored .env / .env.secrets."""

import os
from pathlib import Path

PACKAGE = __name__.split(".")[0]
PREFIX = PACKAGE.upper()


def repo_root() -> Path:
    """The repository containing this package, independent of the cwd (Hydra or SLURM may change it)."""
    return Path(__file__).resolve().parents[3]


def var(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"{PREFIX}_{name}", default)


def runs_root() -> Path:
    return Path(var("RUNS_ROOT") or repo_root() / "runs")


def load_dotenv(*files: Path) -> None:
    """Minimal KEY=VALUE loader; variables already set in the environment win."""
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require(*names: str) -> None:
    """Fail at startup with the full list of missing variables, not at first use."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing environment variables (set them in .env): {', '.join(missing)}")
