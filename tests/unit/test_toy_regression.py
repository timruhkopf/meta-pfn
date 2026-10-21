"""Example project tests (replace with tests of your pipelines): hand-computed cases and properties."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from meta_pfn.data.splits import SplitEmbargoError, check_split


def test_check_split() -> None:
    check_split("test", "final")
    check_split("val", "pilot")
    try:
        check_split("test", "confirm")
    except SplitEmbargoError:
        pass
    else:
        raise AssertionError("test split must be embargoed before stage final")


@settings(max_examples=25)
@given(st.integers(2, 6), st.integers(0, 2**31 - 1))
def test_gradient_step_reduces_mse(dim: int, seed: int) -> None:
    """Property: one small gradient step on least squares never increases the training MSE."""
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(32, dim)), rng.normal(size=32)
    w = np.zeros(dim)
    lr = 0.5 / np.linalg.norm(x.T @ x / 32, 2)  # below 1/L: guaranteed descent
    w_next = w - lr * 2 * x.T @ (x @ w - y) / 32
    assert np.mean((x @ w_next - y) ** 2) <= np.mean((x @ w - y) ** 2) + 1e-12
