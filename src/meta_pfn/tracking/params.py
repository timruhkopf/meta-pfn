"""Semantic params: everything in the config that can change results, flattened."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Cannot change results (runtime, paths) or are bookkeeping (tracking, stage). Not logged as params.
NON_SEMANTIC = ("runtime", "paths", "tracking", "hydra", "stage", "experiment_name")


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, list | tuple):
            flat[name] = json.dumps(value)  # one column per param, also for lists
        else:
            flat[name] = value
    return flat


def semantic_params(cfg: DictConfig) -> dict[str, Any]:
    tree = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(tree, dict)
    return flatten({str(k): v for k, v in tree.items() if k not in NON_SEMANTIC})


def config_hash(params: Mapping[str, Any], replicate_keys: Iterable[str] = ("seed",)) -> str:
    """Identifies a configuration up to its replicates (seeds etc.); independent of key order."""
    skip = set(replicate_keys)
    canonical = json.dumps({k: params[k] for k in sorted(params) if k not in skip}, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
