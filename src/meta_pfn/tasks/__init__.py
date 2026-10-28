"""Pipeline registry: `task.name` in the config selects the function."""

from collections.abc import Callable

from omegaconf import DictConfig

from ..tracking.sink import Run
from .fieldonly import fieldonly
from .gwonly import gwonly
from .headroom import headroom
from .ladder import ladder
from .metapfn import metapfn
from .outmap import outmap
from .regpfn import regpfn
from .sampler_check import sampler_check
from .toy_regression import toy_regression

Pipeline = Callable[[DictConfig, Run], float | None]

TASKS: dict[str, Pipeline] = {
    "fieldonly": fieldonly,
    "gwonly": gwonly,
    "headroom": headroom,
    "ladder": ladder,
    "metapfn": metapfn,
    "outmap": outmap,
    "regpfn": regpfn,
    "sampler_check": sampler_check,
    "toy_regression": toy_regression,  # the template's example; its own tests (`experiment=debug`) use it
}
