"""The generated part of every notebook's header: how the experiment was run, and its configuration."""

from collections import defaultdict
from typing import Any

import pandas as pd
import yaml

from .snapshot import Snapshot

REPLICATE_KEYS = ("seed",)  # shown as replicate axes, not as configurations


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    cells = df.head(max_rows).astype(str)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows_")
    return "\n".join(lines)


def _hours(h: float) -> str:
    return f"{h:.1f} h" if h >= 1 else f"{h * 60:.1f} min"


def provenance_md(snap: Snapshot) -> str:
    m = snap.manifest
    counts = m["status_counts"]
    total = sum(counts.values())
    status = ", ".join(f"{counts[s]} {s.lower()}" for s in ("DONE", "FAILED", "STARTED", "MISSING") if counts.get(s))
    warn = ""
    if snap.preliminary:
        warn = (
            f"\n\n> **Preliminary:** {total - counts.get('DONE', 0)} of {total} members are not DONE. "
            "Aggregates use DONE members only."
        )
    if m.get("allow_dirty"):
        warn += "\n\n> **Not reportable:** launched with `--allow-dirty` (uncommitted changes)."
    rows = pd.DataFrame(
        [
            ("stage", m["stage"]),
            ("sweep", f"`{m['id']}`"),
            ("commit", f"`{m['commit'][:12]}`" + (f" (tag `{m['tag']}`)" if m.get("tag") else "")),
            ("backend / host", f"{m['backend']} / {m['host']}"),
            *[(f"editable dep `{k}`", f"`{v[:12]}`") for k, v in m.get("editable_dependencies", {}).items()],
            ("ran", f"{m.get('first_start')} → {m.get('last_end')}"),
            ("members", f"{total}: {status}"),
            ("compute", f"{_hours(m['wall_h'])} wall, {_hours(m['device_h'])} GPU"),
        ],
        columns=["", ""],
    )
    return f"## How it was run\n\n```bash\n{m['command']}\n```\n\n{_md_table(rows)}{warn}"


def _group_constants(constants: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, Any] = defaultdict(dict)
    for key, value in sorted(constants.items()):
        head, _, rest = key.partition(".")
        if rest:
            grouped[head][rest] = value
        else:
            grouped[key] = value
    return dict(grouped)


def configuration_md(snap: Snapshot) -> str:
    params = snap.runs.attrs.get("params", [])
    varying = [p for p in params if p not in REPLICATE_KEYS]
    replicates = []
    for k in (p for p in params if p in REPLICATE_KEYS):
        values = ", ".join(map(str, sorted(snap.runs[k].dropna().unique())))
        replicates.append(f"`{k}` ∈ {{{values}}}")
    configs = pd.DataFrame(snap.runs[varying]).drop_duplicates() if varying else pd.DataFrame()
    parts = [
        "## Configuration",
        f"Experiment config (`configs/experiment/{snap.manifest['experiment']}.yaml`):",
        f"```yaml\n{snap.experiment_yaml.rstrip()}\n```",
        f"**Varying** ({len(configs)} configurations):\n\n{_md_table(configs)}" if varying else "**Varying:** nothing",
        f"**Replicates:** {', '.join(replicates)}" if replicates else "**Replicates:** none",
        "**Constant:**",
        f"```yaml\n{yaml.safe_dump(_group_constants(snap.constants), sort_keys=False).rstrip()}\n```",
    ]
    return "\n\n".join(parts)


def show_header(snap: Snapshot) -> None:
    text = provenance_md(snap) + "\n\n" + configuration_md(snap)
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)
