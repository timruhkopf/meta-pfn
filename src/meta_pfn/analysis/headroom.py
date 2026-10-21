"""E0 analysis: gains over A alone with pair-bootstrap CIs, oracle precision, and the pre-registered verdict.

Gains are in nats per held-out A query (positive = better than A alone):
  gain_bayes  = NLL_lower - NLL_bayes      what B is worth to a Bayes learner that must infer the relation
  gain_oracle = NLL_lower - NLL_oracle_T   what B is worth when T is known
  gain_naive  = NLL_lower - NLL_naive      what pooling without registration gets
  G           = gain_bayes / gain_oracle   the share of the known-T headroom that survives inferring T
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import bootstrap_ci

CELL = ["d", "kind", "sigma"]


def gains(nll: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    """nll: the `nll` table with the cell columns d, kind, sigma. -> one row per (cell, n_a)."""
    w = nll.pivot_table(index=[*CELL, "pair", "n_a"], columns="predictor", values="nll").reset_index()
    rows = []
    for key, g in w.groupby([*CELL, "n_a"]):
        row = dict(zip([*CELL, "n_a"], key, strict=True))
        for name, col in (("bayes", "bayes"), ("oracle", "oracle_T"), ("naive", "naive")):
            x = (g.lower - g[col]).to_numpy()
            lo, hi = bootstrap_ci(x, n_boot=n_boot)
            row |= {f"gain_{name}": x.mean(), f"gain_{name}_lo": lo, f"gain_{name}_hi": hi}
        row["G"] = row["gain_bayes"] / row["gain_oracle"] if row["gain_oracle"] > 0 else np.nan
        row["n_pairs"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def mc_error(mc: pd.DataFrame) -> pd.DataFrame:
    """Monte-Carlo error of the Bayes oracle per (cell, n_a): mean |run1 - run2| / sqrt(2) over pairs."""
    e = mc.assign(err=(mc.bayes_1 - mc.bayes_2).abs() / np.sqrt(2))
    return e.groupby([*CELL, "n_a"]).err.mean().rename("mc_error").reset_index()


@dataclass
class Rule:
    gate_kinds: tuple[str, ...] = ("window", "warp")
    gate_n: tuple[int, ...] = (2, 4, 8)
    go_sigma: float = 0.1
    go_n: tuple[int, ...] = (2, 4)
    kill_below: float = 0.05
    go_above: float = 0.10
    go_share: float = 0.25
    max_mc_error: float = 0.025
    identity_recovery: float = 0.9  # sanity: on identity pairs the Bayes learner reaches 90% of the oracle gain
    harm_floor: float = -0.05  # safety: on unrelated pairs the CI lower bound stays above this


def verdict(g: pd.DataFrame, mc: pd.DataFrame, rule: Rule | None = None) -> dict[str, object]:
    """The pre-registered E0 rule (notebook 01). Returns the verdict and the checks behind it."""
    rule = rule or Rule()
    g = g.merge(mc, on=[*CELL, "n_a"], how="left")
    ident = g.loc[(g.kind == "identity") & (g.sigma == 0) & (g.n_a == max(rule.gate_n))]
    sane = bool(len(ident)) and bool((ident.gain_bayes >= rule.identity_recovery * ident.gain_oracle).all())
    gate = g.loc[g.kind.isin(rule.gate_kinds) & g.n_a.isin(rule.gate_n)]
    headroom = bool((gate.loc[gate.sigma == 0, "gain_oracle"] > 0).all())
    invalid_cells = gate.loc[gate.mc_error > rule.max_mc_error]
    unrel = g.loc[(g.kind == "unrelated") & (g.n_a >= 1)]
    safe = bool((unrel.gain_bayes_lo > rule.harm_floor).all())
    go = g.loc[g.kind.isin(rule.gate_kinds) & (g.sigma == rule.go_sigma) & g.n_a.isin(rule.go_n)]
    checks = {
        "sanity_identity_recovered": sane,
        "sanity_headroom_with_T_known": headroom,
        "invalid_gate_cells": len(invalid_cells),
        "safe_on_unrelated": safe,
        "go_cells": len(go),
    }
    if not (sane and headroom):
        out = "INVALID"
    elif len(gate) and (gate.gain_bayes_hi < rule.kill_below).all() and not len(invalid_cells):
        out = "KILL"
    elif len(go) and ((go.gain_bayes_lo > rule.go_above) & (rule.go_share < go.G)).all():
        out = "GO"
    else:
        out = "NARROW"
    return {"verdict": out, **checks}
