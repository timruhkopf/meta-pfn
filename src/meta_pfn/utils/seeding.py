import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch on CPU and all GPUs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (legacy global state, which third-party code may use)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
