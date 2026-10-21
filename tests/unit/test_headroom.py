"""E0's invariants: reproducible pairs, forward-only warps inside the box, consistent predictors, and the
decision rule on hand-made gains."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from meta_pfn.analysis.headroom import Rule, gains, mc_error, verdict
from meta_pfn.oracle import evaluate_pair, hypotheses
from meta_pfn.pairs import WarpFamily, apply_warp, draw_pairs, function, pair_seed
from meta_pfn.tasks.headroom import headroom
from meta_pfn.tracking.sink import Run

FAM = WarpFamily()
MASS = {"unrelated": 0.25, "identity": 0.25, "window": 0.25, "warp": 0.25}


def test_pairs_are_reproducible_and_disjoint() -> None:
    a = draw_pairs(2, "warp", 0, 2, 8, 4, 4, FAM, sigma=0.05)
    b = draw_pairs(2, "warp", 0, 2, 8, 4, 4, FAM, sigma=0.05)
    torch.testing.assert_close(a.YA, b.YA)
    assert pair_seed(0, 2, "warp", 0) != pair_seed(1, 2, "warp", 0)
    assert pair_seed(0, 2, "warp", 0) != pair_seed(0, 2, "window", 0)


@pytest.mark.parametrize("kind", ["window", "warp"])
def test_warps_map_the_box_into_itself_monotonically(kind: str) -> None:
    th = FAM.sample(kind, 64, 2, torch.Generator().manual_seed(0))
    x = torch.linspace(0, 1, 101)[None, :, None].expand(64, -1, 2)
    t = apply_warp(th, x)
    assert bool((t >= 0).all() and (t <= 1).all())
    assert bool((t.diff(dim=1) >= -1e-7).all())  # monotone per axis


def test_identity_pair_shares_the_function_and_sigma_zero_is_exact() -> None:
    p = draw_pairs(1, "identity", 0, 1, 8, 4, 4, FAM)
    f = function(p.seeds[0], 1)
    x = p.XA[0]
    with torch.no_grad():
        clean = f.evaluate(x[None], noise=False)[0]
    assert float((p.YA[0] - clean).abs().max()) < 0.1  # observation noise only
    torch.testing.assert_close(function(5, 2, 0.0).W_h, function(5, 2).W_h)


def test_predictors_are_consistent_on_identity_pairs() -> None:
    pfn = load("pfn_smoke")[0].eval()
    p = draw_pairs(1, "identity", 0, 1, 8, 4, 8, FAM)
    hyp = hypotheses(1, FAM, 8, MASS, torch.Generator().manual_seed(0))
    r = evaluate_pair(pfn, p, 0, hyp, [0, 2, 4])
    np.testing.assert_allclose(r["nll"][:, 1], r["nll"][:, 2], rtol=1e-5)  # naive == oracle_T when T = id
    np.testing.assert_allclose(r["post"].sum(1), 1.0, rtol=1e-5)
    assert r["post"][0, 0] == pytest.approx(0.25, abs=1e-6)  # n_A = 0: the posterior is the prior


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="meta_pfn.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=headroom",
                "task.pfn=pfn_smoke",
                "task.n_pairs=2",
                "task.n_mc_pairs=1",
                "task.k=4",
                "task.k_by_d={}",
                "task.n_q=4",
                "task.n_b=8",
                "task.threads=2",
            ],
        )
    run = Run(tmp_path)
    headroom(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert set(tables) == {"nll", "posterior", "mc"}
    assert len(tables["nll"]) == 2 * 5 * 4  # pairs x n_A grid x predictors


def _g(kind, sigma, n_a, bayes, lo, hi, oracle, d=1):
    return {
        "d": d,
        "kind": kind,
        "sigma": sigma,
        "n_a": n_a,
        "gain_bayes": bayes,
        "gain_bayes_lo": lo,
        "gain_bayes_hi": hi,
        "gain_oracle": oracle,
        "G": bayes / oracle if oracle else np.nan,
    }


def _table(related_gain, lo, hi):
    rows = [_g("identity", 0, 8, 1.0, 0.9, 1.1, 1.05), _g("unrelated", 0, 2, 0.0, -0.02, 0.02, 0.0)]
    for kind in ("window", "warp"):
        for s in (0, 0.05, 0.1):
            for n in (2, 4, 8):
                rows.append(_g(kind, s, n, related_gain, lo, hi, 1.0))
    g = pd.DataFrame(rows)
    return g, g[["d", "kind", "sigma", "n_a"]].assign(mc_error=0.01)


def test_verdict_rule() -> None:
    assert verdict(*_table(0.5, 0.3, 0.7))["verdict"] == "GO"
    assert verdict(*_table(0.01, -0.02, 0.04))["verdict"] == "KILL"
    assert verdict(*_table(0.08, 0.02, 0.14))["verdict"] == "NARROW"
    g, mc = _table(0.01, -0.02, 0.04)
    assert verdict(g, mc.assign(mc_error=0.1))["verdict"] == "NARROW"  # imprecise oracle blocks KILL
    g.loc[g.kind == "identity", "gain_bayes"] = 0.1
    assert verdict(g, mc)["verdict"] == "INVALID"


def test_gains_and_mc_error_from_tables() -> None:
    nll = pd.DataFrame(
        {
            "d": 1,
            "kind": "window",
            "sigma": 0.0,
            "pair": np.repeat([0, 1], 4),
            "n_a": 2,
            "predictor": ["lower", "naive", "oracle_T", "bayes"] * 2,
            "nll": [1.0, 0.8, 0.0, 0.5] * 2,
        }
    )
    g = gains(nll, n_boot=100).iloc[0]
    assert g.gain_bayes == pytest.approx(0.5) and g.gain_oracle == pytest.approx(1.0) and pytest.approx(0.5) == g.G
    mc = pd.DataFrame(
        {"d": 1, "kind": "window", "sigma": 0.0, "pair": [0], "n_a": 2, "bayes_1": [0.5], "bayes_2": [0.3]}
    )
    assert mc_error(mc).mc_error.iloc[0] == pytest.approx(0.2 / np.sqrt(2))
    assert Rule().go_sigma == 0.1
