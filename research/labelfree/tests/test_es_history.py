"""Sidecar trial-record invariants (T009)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import history as H         # noqa: E402


def _log(tmp_path) -> Path:
    return tmp_path / "history.jsonl"


def test_write_appears_before_any_score(tmp_path):
    p = _log(tmp_path)
    eid = H.write("cand0", {"a": 1}, "f07", "train", "cmaes", path=p)
    rows = H.read(p)
    assert len(rows) == 1
    assert rows[0]["eval_id"] == eid
    assert rows[0]["score"] is None
    assert "completed_at" not in rows[0]


def test_abandoned_evaluation_still_counts(tmp_path):
    """FR-013's whole point: the deflation counts ATTEMPTS, not successes.

    A log that only recorded completed evaluations would undercount attempts and inflate
    every downstream statistic — which is the opposite of what this feature measures.
    """
    p = _log(tmp_path)
    H.write("a", {}, "f00", "train", "cmaes", path=p)
    done = H.write("b", {}, "f00", "train", "cmaes", path=p)
    H.complete(done, {"score": 1.25}, path=p)
    rows = H.read(p)
    assert len(rows) == 2, "the abandoned attempt vanished from the record"
    assert rows[0]["score"] is None and rows[1]["score"] == 1.25


def test_append_only_survives_a_crash_between_write_and_complete(tmp_path):
    p = _log(tmp_path)
    eid = H.write("a", {}, "f00", "train", "cmaes", path=p)
    before = p.read_text()
    H.complete(eid, {"score": 0.5}, path=p)
    assert p.read_text().startswith(before), "complete() rewrote history instead of appending"


def test_range_must_be_train_or_test(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        H.write("a", {}, "f00", "validation", "cmaes", path=_log(tmp_path))


def test_audit_catches_two_test_evaluations(tmp_path):
    p = _log(tmp_path)
    H.write("a", {}, "f00", "train", "cmaes", path=p)
    H.write("b", {}, "f00", "test", "cmaes", path=p)
    H.write("c", {}, "f00", "test", "cmaes", path=p)
    ok, rep = H.audit(p, shared=tmp_path / "missing.jsonl")
    assert not ok
    assert any("expected exactly 1" in x for x in rep["problems"])


def test_audit_catches_training_after_the_test_evaluation(tmp_path):
    """The leak that matters: the search continuing once the test range has been opened."""
    import time
    p = _log(tmp_path)
    H.write("a", {}, "f00", "train", "cmaes", path=p)
    H.write("b", {}, "f00", "test", "cmaes", path=p)
    time.sleep(1.1)                                   # written_at has second resolution
    H.write("c", {}, "f00", "train", "cmaes", path=p)
    ok, rep = H.audit(p, shared=tmp_path / "missing.jsonl")
    assert not ok
    assert any("written AFTER the test evaluation" in x for x in rep["problems"])


def test_audit_requires_the_shared_summary_row(tmp_path):
    """SC-018: an audit reading only the summary would certify a record it never saw."""
    p, shared = _log(tmp_path), tmp_path / "trials.jsonl"
    H.write("a", {}, "f00", "train", "cmaes", path=p)
    H.write("b", {}, "f00", "test", "cmaes", path=p)

    ok, rep = H.audit(p, shared=shared)
    assert not ok and "missing" in rep["shared_log"]

    shared.write_text(json.dumps({"run": 1}) + "\n")
    ok, rep = H.audit(p, shared=shared)
    assert not ok and "no summary row" in rep["shared_log"]

    shared.write_text(json.dumps({"run": 1, **H.summarise(p)}) + "\n")
    ok, rep = H.audit(p, shared=shared)
    assert ok, rep["problems"]

    H.write("c", {}, "f01", "train", "cmaes", path=p)   # sidecar moves on, summary does not
    H.write("d", {}, "f01", "test", "cmaes", path=p)
    ok, rep = H.audit(p, shared=shared)
    assert not ok and "stale" in rep["shared_log"]
