"""039 T013: the R7 model selection and the stage refusal."""
from __future__ import annotations

import json

from research.nnpredict import judge as J


def _arm(pp, early="UNCHANGED", late="UNCHANGED", ok=True):
    return {"instrument": {"pass": ok},
            "alone": {"arm_pp": pp, "halves": {"early": {"verdict": early}, "late": {"verdict": late}}}}


def test_highest_total_without_a_worsened_half_wins():
    arms = {"a": _arm(12.0), "b": _arm(20.0, late="WORSENED"), "c": _arm(15.0)}
    assert J.select_model(10.0, arms) == "c"


def test_baseline_keeps_ties_and_beats_lower_arms():
    assert J.select_model(10.0, {"a": _arm(10.0), "b": _arm(9.0)}) == J.BASELINE


def test_failed_instrument_cannot_be_selected():
    assert J.select_model(10.0, {"a": _arm(30.0, ok=False)}) == J.BASELINE


def test_stage_refuses_when_preregistration_is_dirty(monkeypatch, capsys):
    monkeypatch.setattr(J, "_refused", lambda path, pr: ["preregistration_models.json has uncommitted changes"])
    assert J.models() == 2
    assert "REFUSED" in capsys.readouterr().out


def test_identity_needs_equal_trades_hash_on_every_fold(monkeypatch):
    recs = {"m": {"f00": {"trades_hash": "a"}, "f01": {"trades_hash": "b"}},
            "fw0": {"f00": {"trades_hash": "a"}, "f01": {"trades_hash": "x"}}}
    import research.nnnc_switches.judge as NJ
    monkeypatch.setattr(NJ, "_recs", lambda tag, folds: recs[tag])
    r = J.identity({"folds": ["f00", "f01"]}, "m", "fw0")
    assert not r["pass"] and r["folds_differing"] == ["f01"]
    recs["fw0"]["f01"]["trades_hash"] = "b"
    assert J.identity({"folds": ["f00", "f01"]}, "m", "fw0")["pass"]


def test_screen_gap_and_decision():
    import numpy as np
    import pandas as pd
    from research.nnpredict import screen as S

    rng = np.random.default_rng(0)
    rows = []
    for f in [f"f{i:02d}" for i in range(8)]:
        sc = rng.normal(size=60)
        rows.append(pd.DataFrame({"fold": f, "score": sc, "pp": 0.1 * sc + rng.normal(0, 0.05, 60),
                                  "bb_width": rng.uniform(0.05, 0.2, 60), "p_ctl": rng.uniform(0, 1, 60)}))
    t = pd.concat(rows, ignore_index=True)
    r = S.score(t)
    assert r["gap_share50"] > 0.05 and r["partial"] > 0 and r["sign_folds"] == 8
    rule = {"screen_rule": {"min_gap_share50": 0.05, "min_sign_folds": 5}}
    assert S.decide(r, rule)["pass"]
    t["pp"] = rng.normal(0, 0.05, len(t))  # no relation -> fails
    assert not S.decide(S.score(t), rule)["pass"]
    t["pp"] = -0.1 * t["score"] + rng.normal(0, 0.05, len(t))  # inverted ranking: only "flip" passes
    assert not S.decide(S.score(t), rule)["pass"]
    assert S.decide(S.score(t.assign(score=-t["score"])), rule)["pass"]
