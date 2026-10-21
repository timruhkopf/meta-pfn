"""The test split is embargoed: only stage `final` may read it."""

SPLITS = ("train", "val", "test")


class SplitEmbargoError(RuntimeError):
    pass


def check_split(split: str, stage: str) -> None:
    """Call wherever data for a split is loaded."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "test" and stage != "final":
        raise SplitEmbargoError(f"the test split is read only at stage 'final' (this run: {stage!r})")
