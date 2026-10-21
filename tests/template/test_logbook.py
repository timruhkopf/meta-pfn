from pathlib import Path

import pytest

from meta_pfn.tracking import logbook
from meta_pfn.tracking.logbook import GateError


def _sweep(stage: str, sid: str = "s1") -> dict:
    return {"id": f"{stage}_{sid}", "experiment": "exp", "stage": stage, "commit": "a" * 40}


def test_gates(root: Path) -> None:
    log = root / "research_log"
    logbook.check_gate(log, "exp", "pilot")  # no gate before confirm
    with pytest.raises(GateError):
        logbook.check_gate(log, "exp", "confirm")  # no entry at all
    logbook.new_entry(log, "exp", "Title", "notebooks/01.ipynb")
    with pytest.raises(GateError):
        logbook.check_gate(log, "exp", "confirm")  # entry, but no pilot GO
    logbook.decide(log, _sweep("pilot"), {"DONE": 3}, "NO_GO", "nothing there")
    with pytest.raises(GateError):
        logbook.check_gate(log, "exp", "confirm")
    logbook.decide(log, _sweep("pilot", "s2"), {"DONE": 3}, "GO", "signal")
    logbook.check_gate(log, "exp", "confirm")
    with pytest.raises(GateError):
        logbook.check_gate(log, "exp", "final")


def test_decide_writes_header_and_keeps_body(root: Path) -> None:
    log = root / "research_log"
    entry = logbook.new_entry(log, "exp", "Title", "notebooks/01.ipynb")
    body = entry.path.read_text().split("---\n", 2)[2]
    logbook.decide(log, _sweep("pilot"), {"DONE": 8, "FAILED": 2}, "GO", "gap 0.1 (CI 0.05-0.2)")
    again = logbook.read_entry(entry.path)
    assert again.body == body
    (decided,) = again.sweeps
    assert decided["decision"] == "GO"
    assert decided["members"] == "8/10 done, 2 failed"
    assert "**GO**" in (log / "README.md").read_text()


def test_decisions_are_final(root: Path) -> None:
    log = root / "research_log"
    logbook.new_entry(log, "exp", "Title", "nb")
    logbook.decide(log, _sweep("pilot"), {"DONE": 1}, "GO", "r")
    with pytest.raises(GateError):
        logbook.decide(log, _sweep("pilot"), {"DONE": 1}, "NO_GO", "changed my mind")


def test_decide_needs_entry(root: Path) -> None:
    with pytest.raises(GateError):
        logbook.decide(root / "research_log", _sweep("pilot"), {"DONE": 1}, "GO", "r")
