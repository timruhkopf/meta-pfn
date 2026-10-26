"""E0 follow-up analysis: how much of the importance sampler's gap to the known-T oracle an SMC learner closes.

Per (d, kind, n_a), over pairs, in nats per held-out A query (positive = better than A alone):
  gain_is, gain_smc, gain_oracle   the kind-conditional learners (importance sampling, SMC) and known T
  closed = (gain_smc - gain_is) / (gain_oracle - gain_is)   the share of the gap that was the sampler
  G_is, G_smc = gain / gain_oracle                          the share of the headroom kept when inferring T
  mc_smc = mean |kind_smc - kind_smc_rep| / √2              the SMC learner's Monte-Carlo error
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

CELL = ["d", "kind"]


def _ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def shares(nll: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    """nll: the `nll` table with columns d, kind. -> one row per (d, kind, n_a)."""
    w = nll.pivot_table(index=[*CELL, "pair", "n_a"], columns="predictor", values="nll").reset_index()
    rows = []
    for key, g in w.groupby([*CELL, "n_a"]):
        gi, gs, go = (
            (g.lower - g.kind_is).to_numpy(),
            (g.lower - g.kind_smc).to_numpy(),
            (g.lower - g.oracle_T).to_numpy(),
        )
        lo, hi = _ratio_ci(gs - gi, go - gi, n_boot)
        rows.append(
            dict(zip([*CELL, "n_a"], key, strict=True))
            | {
                "gain_is": gi.mean(),
                "gain_smc": gs.mean(),
                "gain_oracle": go.mean(),
                "gain_bayes_is": (g.lower - g.bayes_is).mean(),
                "gain_bayes_smc": (g.lower - g.bayes_smc).mean(),
                "closed": (gs - gi).mean() / (go - gi).mean(),
                "closed_lo": lo,
                "closed_hi": hi,
                "G_is": gi.mean() / go.mean(),
                "G_smc": gs.mean() / go.mean(),
                "mc_smc": float(np.mean(np.abs(g.kind_smc - g.kind_smc_rep)) / np.sqrt(2)),
                "n_pairs": len(g),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class Rule:
    n: tuple[int, ...] = (2, 4, 8)
    sampler_above: float = 0.5  # SAMPLER: the SMC learner closes ≥ half the gap (and its CI excludes 0)
    data_below: float = 0.2  # DATA: it closes < 20% of the gap
    max_mc: float = 0.10  # the SMC learner's own MC error must be small for either label


def label(row: pd.Series, rule: Rule) -> str:
    if row.mc_smc > rule.max_mc:
        return "IMPRECISE"
    if row.closed >= rule.sampler_above and row.closed_lo > 0:
        return "SAMPLER"
    if row.closed < rule.data_below:
        return "DATA"
    return "MIXED"


def verdict(s: pd.DataFrame, rule: Rule | None = None) -> dict[str, object]:
    """The pre-registered rule (notebook 02): the label per warp cell and the majority over them."""
    rule = rule or Rule()
    s = s.loc[s.n_a.isin(rule.n)].assign(label=lambda x: x.apply(label, axis=1, rule=rule))
    warp = s.loc[s.kind == "warp", "label"].value_counts()
    top = warp.index[0] if len(warp) and warp.iloc[0] > len(s.loc[s.kind == "warp"]) / 2 else "MIXED"
    return {"verdict": top, "labels": s.set_index([*CELL, "n_a"]).label.to_dict()}
