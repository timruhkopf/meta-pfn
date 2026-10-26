from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_module
from pfn_repro.checkpoints import load

from meta_pfn.analysis.outmap import gains, mc_error, verdict
from meta_pfn.experimental.outmap import (
    PREDICTORS,
    OutputFamily,
    draw_pairs,
    evaluate_pair,
    kum,
    kum_inv,
    log_jac_inv,
)
from meta_pfn.pairs import function, observe
from meta_pfn.tasks.outmap import outmap
from meta_pfn.tracking.sink import Run

FAM = OutputFamily()
MASS = {"identity": 1.0, "outmap": 1.0, "unrelated": 1.0}


def test_kumaraswamy_inverse_and_jacobian() -> None:
    ab = torch.tensor([[1.7, 0.6], [0.5, 2.0], [1.0, 1.0]])
    y = torch.linspace(0.01, 0.99, 50, dtype=torch.float64)
    u = kum_inv(y, ab.double())
    np.testing.assert_allclose(kum(u, ab.double()).numpy(), y.expand(3, -1).numpy(), atol=1e-9)
    h = 1e-6  # finite-difference derivative of g⁻¹
    num = (kum_inv(y + h, ab.double()) - kum_inv(y - h, ab.double())) / (2 * h)
    np.testing.assert_allclose(log_jac_inv(y, ab.double()).numpy(), num.log().numpy(), atol=1e-5)
    np.testing.assert_allclose(log_jac_inv(y, ab.double())[2].numpy(), 0.0, atol=1e-9)  # identity


def test_quantile_grid_is_equal_mass_and_centred() -> None:
    ab = FAM.grid(8, 0.5)
    assert ab.shape == (64, 2)
    np.testing.assert_allclose(ab.log().mean(0).numpy(), 0.0, atol=1e-6)


def test_pairs_are_reproducible_and_a_is_g_of_the_prior_observation() -> None:
    a = draw_pairs(1, "outmap", 0, 2, 8, 4, 4, FAM, 0.5)
    b = draw_pairs(1, "outmap", 0, 2, 8, 4, 4, FAM, 0.5)
    torch.testing.assert_close(a.YA, b.YA)
    s = a.seeds[0]
    y = observe(function(s, 1), s + 1, torch.cat([a.XA[0], a.XQ[0]]))
    torch.testing.assert_close(kum_inv(a.YA[0], a.ab[0][None])[0], y[:4], atol=1e-4, rtol=0)
    ident = draw_pairs(1, "identity", 0, 2, 8, 4, 4, FAM, 0.5)
    assert (ident.ab == 1).all()


def test_identity_pair_oracle_is_naive_and_posterior_normalized() -> None:
    pfn = load("pfn_smoke")[0].eval()
    p = draw_pairs(1, "identity", 0, 1, 8, 4, 6, FAM, 0.5)
    r = evaluate_pair(pfn, p, 0, FAM, MASS, 0.5, [0, 2, 4], 9, 3, 2, torch.Generator().manual_seed(0))
    assert r["nll"].shape == (3, len(PREDICTORS)) and np.isfinite(r["nll"]).all()
    np.testing.assert_allclose(r["nll"][:, 3], r["nll"][:, 2], rtol=1e-5)  # true g = id: oracle_g = naive
    np.testing.assert_allclose(r["post"].sum(1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(r["post"][0], [1 / 3, 1 / 3, 1 / 3], rtol=1e-5)  # n_A = 0: the prior


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    with initialize_config_module(config_module="meta_pfn.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=outmap",
                "task.pfn=pfn_smoke",
                "task.d=2",
                "task.n_pairs=2",
                "task.n_mc_pairs=1",
                "task.k_prior=4",
                "task.k_local=2",
                "task.rounds=1",
                "task.n_a_mult=[0,1,2]",
                "task.n_b_by_d={1: 4, 2: 6}",
                "task.n_q=4",
                "task.threads=2",
            ],
        )
    run = Run(tmp_path)
    outmap(cfg, run)
    tables = {p.stem: pd.read_parquet(p) for p in (run.artifacts / "tables").glob("*.parquet")}
    assert set(tables) == {"nll", "posterior", "mc"}
    assert len(tables["nll"]) == 2 * 3 * len(PREDICTORS)
    assert sorted(tables["nll"].n_a.unique()) == [0, 2, 4]  # n_A scales with d


def _toy(kind: str, bayes: float, oracle: float, spread: float = 0.01) -> list[dict]:
    rows = []
    for pair in range(10):
        for n in (1, 2, 4, 8):
            e = spread * (pair - 4.5)
            vals = {"lower_bayes": 0.0, "bayes": -bayes + e, "oracle_g": -oracle + e, "naive": 0.0, "lower": 0.0}
            rows += [dict(d=1, kind=kind, pair=pair, n_a=n, predictor=k, nll=v) for k, v in vals.items()]
    return rows


@pytest.mark.parametrize(("bayes", "want"), [(0.9, "IDENTIFIED"), (0.3, "PARTIAL"), (0.0, "NONE")])
def test_outmap_rule(bayes: float, want: str) -> None:
    nll = pd.DataFrame(_toy("identity", 1.0, 1.0) + _toy("outmap", bayes, 1.0) + _toy("unrelated", 0.0, 0.0))
    g = gains(nll, n_boot=200)
    mc = g[["d", "kind", "n_a"]].assign(mc_error=0.01)
    assert verdict(g, mc)["verdict"] == want
    assert verdict(g, mc.assign(mc_error=0.1))["verdict"] in ("IMPRECISE", "INVALID")


def test_mc_error() -> None:
    mc = pd.DataFrame({"d": 1, "kind": "outmap", "pair": [0], "n_a": 2, "bayes_1": [0.5], "bayes_2": [0.3]})
    assert mc_error(mc).mc_error.iloc[0] == pytest.approx(0.2 / np.sqrt(2))


def test_amis_importance_ratios_recover_a_prior_expectation() -> None:
    from meta_pfn.experimental.outmap import _MapPosterior

    mp = _MapPosterior(OutputFamily(0.5), 3600, 2000, torch.Generator().manual_seed(1))
    z = mp.stage1()
    ll = -((z - 0.3) ** 2).sum(1, keepdim=True).expand(-1, 2) * 10  # a peaked "likelihood"
    zn = mp.propose(z, ll, [0, 1, 2])
    z = torch.cat([z, zn])
    w = torch.softmax(mp.log_weights(z), 0)
    assert abs(float((w * z[:, 0]).sum())) < 0.03  # E_prior[z_0] = 0 from the weighted mixture draws
