"""`just new-experiment`: an experiment is a config, a notebook (description first) and a log entry."""

import datetime as dt
import json
import re
import uuid
from pathlib import Path

from ..utils.env import PACKAGE
from .logbook import new_entry

DESCRIPTION = """\
# {title}

## Description

<!-- Written before the first run; kept up to date. -->

**Question.** What do we want to know, and why does it matter?

**Setup.** What is compared against what: method, baselines, null controls; data / instances; seeds.

**What to look for.** Which plots and numbers answer the question, and how to read them.

**GO / NO-GO.** Which result would mean GO, NO-GO or INCONCLUSIVE.

**Limits.** Where the components are valid (e.g. the training range of a frozen model).
"""

HEADER_CODE = """\
from {package}.analysis import load_snapshot, paired_diff, show_header, summarize

SNAPSHOT = "results/{experiment}/<sweep_id>"  # set after `just snapshot runs/{experiment}/<sweep_id>`
snap = load_snapshot(SNAPSHOT)
show_header(snap)  # how it was run + configuration, generated from the snapshot"""

ANALYSIS_CODE = """\
# Aggregates use DONE members only; n_failed / n_started / n_missing say what is not in them yet.
summarize(snap.runs, "<metric>")"""


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source.splitlines(keepends=True)}
    return cell | ({"outputs": [], "execution_count": None} if kind == "code" else {})


def notebook_json(title: str, experiment: str) -> dict:
    cells = [
        _cell("markdown", DESCRIPTION.format(title=title)),
        _cell("code", HEADER_CODE.format(package=PACKAGE, experiment=experiment)),
        _cell("markdown", "## Results"),
        _cell("code", ANALYSIS_CODE),
        _cell(
            "markdown",
            "## Verdict\n\nThe GO / NO-GO criterion from the description, evaluated, in one line. "
            "Record it with `just decide`.",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def new_experiment(root: Path, experiment: str, title: str) -> list[Path]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", experiment):
        raise ValueError("experiment names are snake_case: <question>[_<variant>]")
    config = root / "src" / PACKAGE / "configs" / "experiment" / f"{experiment}.yaml"
    if config.exists():
        raise FileExistsError(config)
    config.write_text(f"# @package _global_\n# {title}\ndefaults:\n  - override /task: ???\n")

    nb_dir = root / "notebooks"
    numbers = [int(m.group(1)) for p in nb_dir.glob("*.ipynb") if (m := re.match(r"(\d+)_", p.name))]
    notebook = nb_dir / f"{max(numbers, default=0) + 1:02d}_{dt.date.today().isoformat()}_{experiment}.ipynb"
    notebook.write_text(json.dumps(notebook_json(title, experiment), indent=1) + "\n")

    entry = new_entry(root / "research_log", experiment, title, f"notebooks/{notebook.name}")
    return [config, notebook, entry.path]
