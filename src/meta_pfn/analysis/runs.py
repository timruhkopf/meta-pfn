"""load_runs: one row per planned member, read from the run sinks, including members that failed or never ran."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from ..tracking.sink import member_status, read_run
from ..tracking.sweep import jobs, read_sweep

ID_COLUMNS = ["sweep_id", "member", "status", "config_hash", "git_commit"]


def load_runs(sweep_dirs: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Columns: ID_COLUMNS, then varying params, then summary metrics.

    Params constant across all rows are dropped from the table and kept in df.attrs["constant"];
    df.attrs["params"] / ["metrics"] name the remaining param / metric columns.
    """
    dirs = [Path(sweep_dirs)] if isinstance(sweep_dirs, str | Path) else [Path(d) for d in sweep_dirs]
    rows, param_keys, metric_keys = [], set(), set()
    for d in dirs:
        meta = read_sweep(d)
        for member, overrides in jobs(d).items():
            run = read_run(d / str(member))
            # members that never started only have their overrides as params
            params = run.get("params") or {
                k.lstrip("+"): yaml.safe_load(v) for k, v in (o.split("=", 1) for o in overrides if "=" in o)
            }
            summary = run.get("summary", {})
            tags = run.get("tags", {})
            param_keys |= params.keys()
            metric_keys |= summary.keys()
            rows.append(
                {
                    "sweep_id": meta["id"],
                    "member": member,
                    "status": member_status(d / str(member)),
                    "config_hash": tags.get("config_hash"),
                    "git_commit": tags.get("git_commit", meta["commit"]),
                    **{f"param:{k}": v for k, v in params.items()},
                    **{f"metric:{k}": v for k, v in summary.items()},
                }
            )
    df = pd.DataFrame(rows)
    started = df["status"] != "MISSING"
    if started.any():  # override-only keys (e.g. `experiment`) are not params
        param_keys = {k for k in param_keys if df.loc[started, f"param:{k}"].notna().any()}
    constant, varying = {}, []
    for key in sorted(param_keys):
        col = df[f"param:{key}"]
        values = col.dropna().astype(str).unique()
        if len(values) <= 1 and bool((col.notna() | ~started).all()):
            value = col.dropna().iloc[0] if bool(col.notna().any()) else None
            constant[key] = value.item() if value is not None and hasattr(value, "item") else value  # numpy -> python
        else:
            varying.append(key)
    df = df.drop(columns=[c for c in df.columns if c.startswith("param:") and c[6:] not in param_keys])
    df = df.rename(columns=lambda c: c.split(":", 1)[1] if c.startswith(("param:", "metric:")) else c)
    df = pd.DataFrame(df[ID_COLUMNS + varying + sorted(metric_keys)])
    df.attrs.update(constant=constant, params=varying, metrics=sorted(metric_keys))
    return df
