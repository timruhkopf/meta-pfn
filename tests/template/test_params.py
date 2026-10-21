from hypothesis import given
from hypothesis import strategies as st

from meta_pfn.tracking.params import config_hash, flatten

keys = st.text("abcdefgh", min_size=1, max_size=6)
values = st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=5), st.booleans())
params = st.dictionaries(keys, values, min_size=1, max_size=8)


@given(params, st.integers())
def test_hash_ignores_key_order_and_seed(p: dict, seed: int) -> None:
    shuffled = dict(reversed(list(p.items())))
    assert config_hash({**p, "seed": seed}) == config_hash({**shuffled, "seed": seed + 1})


@given(params, keys, values)
def test_hash_changes_with_any_semantic_value(p: dict, key: str, value: object) -> None:
    changed = {**p, key: value}
    if changed != p and key != "seed":
        assert config_hash(changed) != config_hash(p)


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": [1, 2]}, "d": None}) == {"a.b": 1, "a.c": "[1, 2]", "d": None}
