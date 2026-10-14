"""033 T024: the staged judge -- per-arm stage order, comparator reads, small margins, refusal."""

from __future__ import annotations

import json
import os

import pytest

from research.width_relative import judge as J


F = [f"f{i:02d}" for i in range(15)]
PR = {"books": {"ARO": {"exit_off": "wr_aro_sz", "exit_on": "wr_aro_szx", "control": "wf_aro_g24",
                        "width": "w_aro"}},
      "folds": F, "early": F[:8], "late": F[8:],
      "thirds": {"t1": F[:5], "t2": F[5:10], "t3": F[10:]},
      "small_margin_pp": 1.0, "confirmation_route": "post-2026-09-07 folds or dry-run",
      "exempt_records": []}


def _inst(off=None, on=None, label=True):
    base = {"S0_pass": True, "S0_inconclusive": False, "zero_trade_folds": []}
    return {"models": {"ARO": {"label_pass": label,
                               "arms": {"exit_off": base | (off or {}),
                                        "exit_on": base | (on or {})}}}}


def _sep(passed=True):
    return {"models": {"ARO": {"S1_pass": passed, "early": {}, "current_gap_early_preregistered": 0.072}}}


def _pool(per_fold: dict, read: list | None = None):
    def f(tag):
        if read is not None:
            read.append(tag)
        return {"folds_missing": [], "per_fold": [{"fold_id": k, "return_pct": per_fold[tag]}
                                                   for k in F]}
    return f


def _judge(verdict="PASS", gain=10.0):
    return lambda pr, a, c: {"verdict": verdict, "late_gain_pp": gain}


def _raise(*a, **k):
    raise AssertionError("a later stage's records were read")


PP = {"wr_aro_sz": 3.0, "wr_aro_szx": 4.0, "wf_aro_g24": 2.0, "w_aro": 1.0}


def test_s1_fail_stops_exit_off_but_exit_on_is_judged_with_exit_off_as_comparator():
    r = J.judge_arm(PR, "ARO", "exit_off", _inst(), _sep(False), _raise, _raise)
    assert r["decision"] == "FAIL" and r["stopped_at"] == "S1" and r["S2"] == J.NR
    read = []
    r2 = J.judge_arm(PR, "ARO", "exit_on", _inst(), _sep(False), _pool(PP, read), _judge())
    assert r2["S1"] == "n/a" and "wr_aro_sz" in read
    assert r2["decision"] == "PASS (PROVISIONAL)" and r2["confirmation_route"]


def test_s0_failure_and_zero_trade_are_inconclusive_and_read_nothing():
    r = J.judge_arm(PR, "ARO", "exit_off", _inst(off={"S0_pass": False}), _sep(), _raise, _raise)
    assert r["decision"] == "INCONCLUSIVE" and r["stopped_at"] == "S0"
    r = J.judge_arm(PR, "ARO", "exit_on",
                    _inst(on={"S0_inconclusive": True, "zero_trade_folds": ["f03"]}),
                    _sep(), _raise, _raise)
    assert r["decision"] == "INCONCLUSIVE" and "f03" in r["reason"]


def test_s2_fail_leaves_s3_unjudged_and_no_thirds():
    pp = PP | {"wr_aro_sz": 1.5}  # beats width (1.0), not control (2.0)
    r = J.judge_arm(PR, "ARO", "exit_off", _inst(), _sep(), _pool(pp), _raise)
    assert r["decision"] == "FAIL" and r["stopped_at"] == "S2" and r["S3"] == J.NR
    assert "thirds" not in r


def test_small_early_margin_is_inconclusive():
    pp = PP | {"wr_aro_sz": 2.1}  # 8 early folds x 0.1 = 0.8 pp over the control
    r = J.judge_arm(PR, "ARO", "exit_off", _inst(), _sep(), _pool(pp), _raise)
    assert r["decision"] == "INCONCLUSIVE" and r["S2"]["small_margin"]


def test_full_pass_is_provisional_never_adopted_and_s3_fail_paths():
    r = J.judge_arm(PR, "ARO", "exit_off", _inst(), _sep(), _pool(PP), _judge("PASS", 7.0))
    assert r["decision"] == "PASS (PROVISIONAL)" and set(r["thirds"]) == {"t1", "t2", "t3"}
    assert "ADOPT" not in json.dumps(r).upper()
    assert J.judge_arm(PR, "ARO", "exit_off", _inst(), _sep(), _pool(PP),
                       _judge("FAIL"))["decision"] == "FAIL"
    r = J.judge_arm(PR, "ARO", "exit_on", _inst(), _sep(), _pool(PP), _judge("INCONCLUSIVE"))
    assert r["decision"] == "INCONCLUSIVE" and r["stopped_at"] == "S3"


def test_refuses_a_record_older_than_the_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "ARMS_DIR", tmp_path)
    p = tmp_path / "wr_aro_szx_f02_F02.json"
    p.write_text("{}")
    os.utime(p, (1000, 1000))
    assert J.refusals(PR, commit_ts=2000.0) == ["record older than the pre-registration commit: "
                                                "wr_aro_szx_f02_F02.json"]
    os.utime(p, (3000, 3000))
    assert J.refusals(PR, commit_ts=2000.0) == []


def test_seed2_repeat_resolves_a_small_s2_margin_both_ways():
    pp = PP | {"wr_aro_szx": 3.1}  # 8 x 0.1 = 0.8 pp over exit_off (3.0): small
    pass2 = lambda m, arm, comps: {"margins": {"vs_exit_off": 0.4, "vs_width": 5.0}}
    fail2 = lambda m, arm, comps: {"margins": {"vs_exit_off": -0.2, "vs_width": 5.0}}
    r = J.judge_arm(PR, "ARO", "exit_on", _inst(), _sep(), _pool(pp), _judge(), pass2)
    assert r["decision"] == "PASS (PROVISIONAL)" and r["S2"]["seed2"]["margins"]["vs_exit_off"] == 0.4
    r = J.judge_arm(PR, "ARO", "exit_on", _inst(), _sep(), _pool(pp), _raise, fail2)
    assert r["decision"] == "FAIL" and r["stopped_at"] == "S2" and r["S3"] == J.NR
