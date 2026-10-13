"""031 T027: the judge refuses records that predate the pre-registration unless exempt, and
turns a zero-add arm into INCONCLUSIVE."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[3]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.held_pair import judge as J  # noqa: E402


PR = {"books": {"ARO": {"arm": "zz_hp_test", "folds": ["f00", "f08"]}},
      "folds_2022": [], "exempt_records": ["zz_hp_test:f00"]}


def _write(tmp, monkeypatch, fold, mtime):
    monkeypatch.setattr(J, "ARMS", tmp)
    p = tmp / f"zz_hp_test_{fold}_{fold.upper()}.json"
    p.write_text(json.dumps({"n_adds": 1}))
    os.utime(p, (mtime, mtime))
    return p


def test_older_non_exempt_record_is_refused(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "f08", 1000.0)
    out = J.refusals(PR, commit_ts=2000.0)
    assert out == ["record older than the pre-registration commit: zz_hp_test_f08_F08.json"]


def test_older_exempt_record_and_newer_records_are_accepted(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "f00", 1000.0)
    _write(tmp_path, monkeypatch, "f08", 3000.0)
    assert J.refusals(PR, commit_ts=2000.0) == []


def test_zero_adds_is_inconclusive():
    v = J._zero_adds({"verdict": "PASS"}, {"f00": 0, "f01": None})
    assert v["verdict"] == "INCONCLUSIVE" and v["inconclusive_reason"] == "zero adds"
    assert J._zero_adds({"verdict": "FAIL"}, {"f00": 2})["verdict"] == "FAIL"
    assert J._zero_adds({"verdict": "MISSING"}, {"f00": 0})["verdict"] == "MISSING"
