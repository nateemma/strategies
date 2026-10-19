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
