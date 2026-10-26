"""E0b analysis: gains over A alone (g inferred) with pair-bootstrap CIs, the Bayes learner's MC error, the verdict.

Gains are in nats per held-out A query (positive = better than A alone), all against `lower_bayes`, the A-only
learner under the same pair prior (it infers g from A's points too):
  gain_bayes  = NLL_lower_bayes - NLL_bayes      what B is worth when g must be inferred
  gain_oracle = NLL_lower_bayes - NLL_oracle_g   what B is worth when g is known
  gain_naive  = NLL_lower_bayes - NLL_naive      pooling without the output map
  G           = gain_bayes / gain_oracle         the share of the known-g headroom that survives inferring g
n_A is reported as a multiple of d (`n_mult` = n_a / d), so cells with different d line up.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import bootstrap_ci

CELL = ["d", "kind"]


def gains(nll: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    """nll: the `nll` table with the cell columns d, kind. -> one row per (cell, n_a)."""
    w = nll.pivot_table(index=[*CELL, "pair", "n_a"], columns="predictor", values="nll").reset_index()
    rows = []
    for key, g in w.groupby([*CELL, "n_a"]):
        row = dict(zip([*CELL, "n_a"], key, strict=True))
        row["n_mult"] = row["n_a"] // row["d"]
        for name, col in (("bayes", "bayes"), ("oracle", "oracle_g"), ("naive", "naive"), ("lower", "lower")):
            x = (g.lower_bayes - g[col]).to_numpy()
            lo, hi = bootstrap_ci(x, n_boot=n_boot)
            row |= {f"gain_{name}": x.mean(), f"gain_{name}_lo": lo, f"gain_{name}_hi": hi}
        row["G"] = row["gain_bayes"] / row["gain_oracle"] if row["gain_oracle"] > 0 else np.nan
        row["n_pairs"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def mc_error(mc: pd.DataFrame) -> pd.DataFrame:
    """Monte-Carlo error of the Bayes learner per (cell, n_a): mean |grid1 - grid2| / sqrt(2) over pairs."""
    e = mc.assign(err=(mc.bayes_1 - mc.bayes_2).abs() / np.sqrt(2))
    return e.groupby([*CELL, "n_a"]).err.mean().rename("mc_error").reset_index()


@dataclass
class Rule:
    gate_mult: tuple[int, ...] = (2, 4, 8)  # n_A = 2d, 4d, 8d
    identified_share: float = 0.75
    min_gain: float = 0.10
    none_below: float = 0.05
    max_mc_error: float = 0.025
    identity_recovery: float = 0.9  # sanity: on identity pairs the Bayes learner reaches 90% of the oracle gain
    harm_floor: float = -0.05  # safety: on unrelated pairs the CI lower bound stays above this


def verdict(g: pd.DataFrame, mc: pd.DataFrame, rule: Rule | None = None) -> dict[str, object]:
    """The pre-registered E0b rule (notebook 03). Returns the verdict and the checks behind it."""
    rule = rule or Rule()
    g = g.merge(mc, on=[*CELL, "n_a"], how="left")
    ident = g.loc[(g.kind == "identity") & (g.n_mult == max(rule.gate_mult))]
    sane = bool(len(ident)) and bool((ident.gain_bayes >= rule.identity_recovery * ident.gain_oracle).all())
    gate = g.loc[(g.kind == "outmap") & g.n_mult.isin(rule.gate_mult)]
    headroom = bool(len(gate)) and bool((gate.gain_oracle > 0).all())
    imprecise = int((gate.mc_error > rule.max_mc_error).sum())
    unrel = g.loc[(g.kind == "unrelated") & (g.n_a >= 1)]
    safe = bool((unrel.gain_bayes_lo > rule.harm_floor).all())
    checks = {
        "sanity_identity_recovered": sane,
        "sanity_headroom_with_g_known": headroom,
        "imprecise_gate_cells": imprecise,
        "safe_on_unrelated": safe,
    }
    if not (sane and headroom):
        out = "INVALID"
    elif imprecise:
        out = "IMPRECISE"
    elif ((gate.gain_bayes_lo > rule.min_gain) & (rule.identified_share < gate.G)).all():
        out = "IDENTIFIED"
    elif (gate.gain_bayes_hi < rule.none_below).all():
        out = "NONE"
    else:
        out = "PARTIAL"
    return {"verdict": out, **checks}
