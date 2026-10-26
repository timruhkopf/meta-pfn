"""The SMC reference learner: its prior matches the pair prior, its helpers do what they claim, and the
follow-up pipeline reproduces the pilot's importance-sampling numbers on the same pairs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from meta_pfn.oracle import evaluate_pair, hypotheses
from meta_pfn.pairs import WarpFamily, draw_pairs
from meta_pfn.smc import Data, ess, log_prior, next_increment, sample_prior, smc, systematic, to_theta
from meta_pfn.tasks.sampler_check import sampler_check
from meta_pfn.tracking.sink import Run

FAM = WarpFamily()
MASS = {"unrelated": 0.25, "identity": 0.25, "window": 0.25, "warp": 0.25}


@pytest.mark.parametrize("kind", ["window", "warp"])
def test_prior_in_u_matches_the_pair_prior(kind: str) -> None:
    th = to_theta(sample_prior(20_000, 2, kind, FAM, torch.Generator().manual_seed(0)), kind, FAM)
    ref = FAM.sample(kind, 20_000, 2, torch.Generator().manual_seed(1))
    torch.testing.assert_close(th.mean((0, 1)), ref.mean((0, 1)), atol=0.02, rtol=0.02)
    torch.testing.assert_close(th.std((0, 1)), ref.std((0, 1)), atol=0.02, rtol=0.03)


def test_log_prior_is_a_normalized_density() -> None:
    # 1-d slice of the window prior: ∫ logistic(u) du = 1 per free coordinate
    u = torch.linspace(-30, 30, 200_001)
    grid = torch.zeros(len(u), 1, 2)
    grid[:, 0, 0] = u
    dens = (log_prior(grid, "window", FAM) - log_prior(torch.zeros(1, 1, 2), "window", FAM) + np.log(0.25)).exp()
    assert float(torch.trapezoid(dens, u)) == pytest.approx(1.0, abs=1e-3)


def test_tempering_step_hits_the_target_ess() -> None:
    logw = torch.full((500,), -np.log(500))
    ll = torch.randn(500, generator=torch.Generator().manual_seed(0)) * 50
    step = next_increment(logw, ll, 1.0, 250.0)
    assert step < 1.0
    assert ess(logw + step * ll) == pytest.approx(250, rel=1e-3)
    assert next_increment(logw, ll * 0, 1.0, 250.0) == 1.0


def test_systematic_resampling_follows_the_weights() -> None:
    logw = torch.log(torch.tensor([0.5, 0.25, 0.25, 0.0]))
    idx = systematic(logw, torch.Generator().manual_seed(0))
    assert sorted(idx.tolist()) == [0, 0, 1, 2]
    assert 3 not in idx.tolist()


def test_smc_is_reproducible_and_consistent_at_n0() -> None:
    pfn = load("pfn_smoke")[0].eval()
    p = draw_pairs(1, "warp", 0, 1, 8, 4, 4, FAM)
    data = Data(p.XB[0], p.YB[0], p.XA[0], p.YA[0], p.XQ[0], p.YQ[0])
    a = smc(pfn, data, "warp", FAM, 16, [0, 2, 4], torch.Generator().manual_seed(0), p.theta[0], n_moves=1)
    b = smc(pfn, data, "warp", FAM, 16, [0, 2, 4], torch.Generator().manual_seed(0), p.theta[0], n_moves=1)
    np.testing.assert_array_equal(a.logq, b.logq)
    assert a.logz[0] == 0.0 and a.stages[0] == 0  # no data, no evidence, no tempering
    assert np.isfinite(a.logq).all() and (a.t_err >= 0).all()
    assert (np.diff(a.stages) >= 1).all()


def test_kind_outputs_of_the_importance_sampler_recombine_to_bayes() -> None:
    pfn = load("pfn_smoke")[0].eval()
    p = draw_pairs(1, "window", 0, 1, 8, 4, 8, FAM)
    hyp = hypotheses(1, FAM, 8, MASS, torch.Generator().manual_seed(0))
    r = evaluate_pair(pfn, p, 0, hyp, [0, 2, 4])
    w = np.log(0.25) + r["kind_logz"]
    w = w - np.logaddexp.reduce(w, axis=1, keepdims=True)
    mix = np.logaddexp.reduce(w[:, :, None] + r["kind_logq"], axis=1)  # [grid, Q]
    np.testing.assert_allclose(-mix.mean(1), r["nll"][:, 3], rtol=1e-4)
    np.testing.assert_allclose(np.exp(w), r["post"], atol=1e-5)


def _compose(*extra: str):
    with initialize_config_module(config_module="meta_pfn.configs", version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "experiment=sampler_check",
                "task.pfn=pfn_smoke",
                "task.n_pairs=1",
                "task.k=4",
                "task.k_by_d={}",
                "task.k_smc=8",
                "task.n_moves=1",
                "task.n_a_grid=[0,2]",
                "task.n_q=4",
                "task.n_b=8",
                "task.threads=2",
                *extra,
            ],
        )


def _tables(run: Run) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    run = Run(tmp_path / "smc")
    sampler_check(_compose(), run)
    tables = _tables(run)
    assert set(tables) == {"nll", "posterior", "smc"}
    assert len(tables["nll"]) == 1 * 2 * 7  # pairs x n_A grid x predictors
    assert len(tables["smc"]) == 3  # true family twice, the other once
    post = tables["posterior"]
    np.testing.assert_allclose(post[[c for c in post if c.startswith("p_")]].sum(1), 1.0, rtol=1e-5)

    # regression: kind_is is the importance sampler's, not overwritten by the SMC values
    run_is = Run(tmp_path / "is")
    sampler_check(_compose("task.smc=false"), run_is)
    nll_is = _tables(run_is)["nll"]
    assert set(nll_is.predictor) == {"lower", "oracle_T", "bayes_is", "kind_is"}
    key = ["pair", "n_a", "predictor"]
    both = tables["nll"].merge(nll_is, on=key, suffixes=("", "_is"))
    np.testing.assert_allclose(both.nll, both.nll_is, rtol=1e-6)
    w = tables["nll"].pivot_table(index=["pair", "n_a"], columns="predictor", values="nll")
    assert not np.allclose(w.kind_is, w.kind_smc)


def _toy_nll(gap_closed: float, rep_noise: float = 0.0) -> pd.DataFrame:
    rows = []
    for kind in ("window", "warp"):
        for pair in range(8):
            for n in (2, 4, 8):
                lower, oracle = 0.0, -2.0 - 0.1 * pair
                is_ = -0.5
                smc_ = is_ + gap_closed * (oracle - is_)
                vals = {
                    "lower": lower,
                    "oracle_T": oracle,
                    "kind_is": is_,
                    "bayes_is": is_,
                    "kind_smc": smc_,
                    "bayes_smc": smc_,
                    "kind_smc_rep": smc_ + rep_noise,
                }
                rows += [dict(d=1, kind=kind, pair=pair, n_a=n, predictor=k, nll=v) for k, v in vals.items()]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("closed", "noise", "want"), [(0.9, 0, "SAMPLER"), (0.05, 0, "DATA"), (0.35, 0, "MIXED"), (0.9, 0.5, "IMPRECISE")]
)
def test_sampler_check_rule(closed: float, noise: float, want: str) -> None:
    from meta_pfn.analysis.sampler_check import shares, verdict

    s = shares(_toy_nll(closed, noise), n_boot=200)
    assert s.closed.to_numpy() == pytest.approx(closed)
    assert verdict(s)["verdict"] == want
