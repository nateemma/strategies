"""032 T020: the staged judge -- refusals, stage order, small margins, wording."""

from __future__ import annotations

import json
import os

import pytest

from research.candidate_training import judge as J


F = [f"f{i:02d}" for i in range(15)]
PR = {"books": {"ARO": {"arm": "co_aro", "control": "wf_aro_g24", "width": "w_aro"}},
      "folds": F, "early": F[:8], "late": F[8:],
      "thirds": {"t1": F[:5], "t2": F[5:10], "t3": F[10:]},
      "small_margin_pp": 1.0, "confirmation_route": "post-2026-09-07 folds or dry-run",
      "exempt_records": []}


def _inst(**over):
    m = {"fidelity": {"recall": 0.99, "precision": 0.99}, "fidelity_pass": True,
         "entry_violations": 0, "sell_rows_kept": 0, "trades_per_fold": {f: 10 for f in F}}
    return {"models": {"CO_ARO": m | over}}


def _auc(passed=True):
    h = {"auc_retrained": 0.6, "auc_current": 0.58, "diff": 0.02 if passed else -0.01}
    return {"models": {"ARO": {"early": h, "late": h, "S1_pass": passed}}}


def _pool(per_fold: dict):
    def f(tag):
        return {"folds_missing": [], "per_fold": [{"fold_id": k, "return_pct": per_fold[tag]}
                                                   for k in F]}
    return f


def _judge(verdict="PASS", late_gain=10.0):
    return lambda pr, a, c: {"verdict": verdict, "late_gain_pp": late_gain}


def _raise(*a, **k):
    raise AssertionError("a later stage's records were read")


def test_s1_fail_leaves_later_stages_unread():
    r = J.judge_model(PR, "ARO", _auc(False), _inst(), _raise, _raise)
    assert r["decision"] == "FAIL" and r["stopped_at"] == "S1"
    assert r["S2"] == r["S3"] == J.NR


def test_instrument_failure_is_inconclusive_and_reads_nothing():
    r = J.judge_model(PR, "ARO", _auc(), _inst(entry_violations=3), _raise, _raise)
    assert r["decision"] == "INCONCLUSIVE" and r["stopped_at"] == "instrument"
    r = J.judge_model(PR, "ARO", _auc(), _inst(trades_per_fold={**{f: 5 for f in F}, "f03": 0}),
                      _raise, _raise)
    assert r["reason"] == "a fold has zero trades"


def test_s2_fail_leaves_s3_unread():
    pool = _pool({"co_aro": 1.0, "wf_aro_g24": 2.0, "w_aro": 0.5})
    r = J.judge_model(PR, "ARO", _auc(), _inst(), pool, _raise)
    assert r["decision"] == "FAIL" and r["stopped_at"] == "S2" and r["S3"] == J.NR


def test_small_s2_margin_is_inconclusive_pending_repeat():
    # 8 early folds x 0.1 pp = 0.8 pp margin over the control
    pool = _pool({"co_aro": 2.1, "wf_aro_g24": 2.0, "w_aro": 0.0})
    r = J.judge_model(PR, "ARO", _auc(), _inst(), pool, _raise)
    assert r["decision"] == "INCONCLUSIVE" and r["S2"]["small_margin"] and r["S3"] == J.NR


def test_full_pass_is_provisional_and_never_adopted():
    pool = _pool({"co_aro": 3.0, "wf_aro_g24": 2.0, "w_aro": 1.0})
    r = J.judge_model(PR, "ARO", _auc(), _inst(), pool, _judge("PASS", 7.0))
    assert r["decision"] == "PASS (PROVISIONAL)" and r["confirmation_route"]
    assert "ADOPT" not in json.dumps(r).upper()
    assert set(r["thirds"]) == {"t1", "t2", "t3"}


def test_s3_judge_fail_and_late_width_loss_fail():
    pool = _pool({"co_aro": 3.0, "wf_aro_g24": 2.0, "w_aro": 1.0})
    assert J.judge_model(PR, "ARO", _auc(), _inst(), pool, _judge("FAIL"))["decision"] == "FAIL"
    pool2 = _pool({"co_aro": 3.0, "wf_aro_g24": 2.0, "w_aro": 1.0})
    r = J.judge_model(PR, "ARO", _auc(), _inst(), pool2, _judge("INCONCLUSIVE"))
    assert r["decision"] == "INCONCLUSIVE" and r["stopped_at"] == "S3"


def test_refuses_a_record_older_than_the_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "ARMS_DIR", tmp_path)
    p = tmp_path / "co_aro_f00_F00.json"
    p.write_text("{}")
    os.utime(p, (1000, 1000))
    assert J.refusals(PR, commit_ts=2000.0) == ["record older than the pre-registration commit: "
                                                "co_aro_f00_F00.json"]
    os.utime(p, (3000, 3000))
    assert J.refusals(PR, commit_ts=2000.0) == []
