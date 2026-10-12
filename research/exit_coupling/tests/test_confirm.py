"""028 T005: recovery share, feasibility pass, the pre-registered decision rule and its verdict
mapping, CI units, gate-5 re-run, and the pre-registration refusals."""

from __future__ import annotations

import pandas as pd
import pytest
from research.exit_coupling import confirm as C


PREREG = {"min_control_trades": 200, "recovery_floor": 0.50}
GOOD_RISK = {"verdict_sharpe": "UNCHANGED", "verdict_maxdd": "IMPROVED"}


def _decide(**kw):
    args = dict(gain_pp=3.0, paired_verdict="UNCHANGED", risk=GOOD_RISK, early_gain=8.4,
                late_gain=6.3, ctl_trades=500, folds_changed=3, prereg=PREREG)
    args.update(kw)
    return C.decide(**args)


# ------------------------------------------------------------------------- recovery share


def test_recovery_share_counts_clean_stop_exits_only():
    df = pd.DataFrame([
        {"rec_reason": "stop", "touch_m_after": True, "ambiguous": False},
        {"rec_reason": "stop", "touch_m_after": False, "ambiguous": False},
        {"rec_reason": "stop", "touch_m_after": True, "ambiguous": False},
        {"rec_reason": "stop", "touch_m_after": True, "ambiguous": True},   # excluded
        {"rec_reason": "take_profit", "touch_m_after": True, "ambiguous": False},  # ignored
    ])
    share, n_stop, n_rec = C.recovery_share(df)
    assert (n_stop, n_rec) == (3, 2)
    assert share == pytest.approx(2 / 3)


def test_recovery_share_none_without_stop_exits():
    df = pd.DataFrame([{"rec_reason": "roi", "touch_m_after": False, "ambiguous": False}])
    assert C.recovery_share(df) == (None, 0, 0)


# ------------------------------------------------------------------------ feasibility pass


def test_floor_applies_to_stop_side_candidates():
    assert C.candidate_passes(5.0, 0.60, 0.50, True)
    assert not C.candidate_passes(5.0, 0.40, 0.50, True)


def test_share_exactly_at_floor_passes():
    assert C.candidate_passes(5.0, 0.50, 0.50, True)


def test_floor_not_applied_to_c2():
    assert C.candidate_passes(5.0, 0.10, 0.50, False)
    assert C.candidate_passes(5.0, None, 0.50, False)


def test_non_positive_replay_gain_fails():
    assert not C.candidate_passes(0.0, 0.90, 0.50, True)
    assert not C.candidate_passes(-1.0, 0.10, 0.50, False)


def test_missing_share_fails_when_floor_applies():
    assert not C.candidate_passes(5.0, None, 0.50, True)


# ------------------------------------------------------------------------- verdict mapping


@pytest.mark.parametrize("v,expected", [
    ("IMPROVED", "pass"), ("UNCHANGED", "pass"), ("WORSENED", "fail"),
    ("INCONCLUSIVE_UNDERPOWERED", "pass"), ("INCONCLUSIVE_ARM_LOSES", "pass"),
    ("INCONCLUSIVE_ABSORBED", "inconclusive"), ("UNRESOLVABLE", "inconclusive"),
])
def test_map_paired_every_value(v, expected):
    assert C.map_paired(v) == expected


@pytest.mark.parametrize("v,expected", [
    ("IMPROVED", "pass"), ("UNCHANGED", "pass"), ("WORSENED", "fail"),
    ("UNRESOLVABLE", "inconclusive"),
])
def test_map_risk_every_value(v, expected):
    assert C.map_risk(v) == expected


def test_unknown_verdict_value_raises():
    with pytest.raises(ValueError):
        C.map_paired("MAYBE")
    with pytest.raises(ValueError):
        C.map_risk("INCONCLUSIVE_ARM_LOSES")


# ---------------------------------------------------------------------------- decision rule


def test_all_conditions_hold_adopts():
    assert _decide() == ("ADOPT", [], [])


def test_unchanged_paired_with_positive_gain_adopts_not_a_significance_test():
    verdict, failed, _ = _decide(paired_verdict="UNCHANGED", gain_pp=0.5)
    assert verdict == "ADOPT" and failed == []


@pytest.mark.parametrize("kw,letter", [
    ({"gain_pp": 0.0}, "a"),
    ({"gain_pp": -2.0}, "a"),
    ({"paired_verdict": "WORSENED"}, "b"),
    ({"risk": {"verdict_sharpe": "WORSENED", "verdict_maxdd": "UNCHANGED"}}, "c"),
    ({"risk": {"verdict_sharpe": "IMPROVED", "verdict_maxdd": "WORSENED"}}, "c"),
    ({"early_gain": -0.1}, "d"),
    ({"late_gain": 0.0}, "d"),
])
def test_each_letter_failing_alone_is_not_adopted(kw, letter):
    verdict, failed, _ = _decide(**kw)
    assert verdict == "NOT ADOPTED"
    assert failed == [letter]


def test_arm_loses_and_underpowered_pass_b_with_a_flag():
    for v in ("INCONCLUSIVE_ARM_LOSES", "INCONCLUSIVE_UNDERPOWERED"):
        verdict, failed, flags = _decide(paired_verdict=v)
        assert verdict == "ADOPT" and failed == [] and flags == [v]


def test_too_few_control_trades_is_inconclusive():
    assert _decide(ctl_trades=199)[0] == "INCONCLUSIVE"
    assert _decide(ctl_trades=200)[0] == "ADOPT"


def test_zero_folds_changed_is_inconclusive():
    assert _decide(folds_changed=0)[0] == "INCONCLUSIVE"


def test_absorbed_or_unresolvable_is_inconclusive():
    assert _decide(paired_verdict="INCONCLUSIVE_ABSORBED")[0] == "INCONCLUSIVE"
    assert _decide(paired_verdict="UNRESOLVABLE")[0] == "INCONCLUSIVE"
    assert _decide(risk={"verdict_sharpe": "UNRESOLVABLE", "verdict_maxdd": "UNCHANGED"})[0] \
        == "INCONCLUSIVE"


def test_rule_a_uses_per_fold_sum_not_paired_delta():
    """The paired delta_pp is not an input: a negative paired delta cannot veto a positive
    per-fold gain, and a positive one cannot rescue a negative gain."""
    assert _decide(gain_pp=1.0)[0] == "ADOPT"
    assert _decide(gain_pp=-1.0)[0] == "NOT ADOPTED"


# ------------------------------------------------------------------------- units and gate 5


def test_ci_half_width_converts_pp_per_year_to_window_pp():
    assert C.ci_half_width_pp([-36.5, 73.0], 300 / 365) == pytest.approx(45.0)
    assert C.ci_half_width_pp(None, 1.0) is None


def test_rerun_check_equal_hash_keeps_verdict_different_turns_inconclusive():
    assert C.after_rerun("ADOPT", True) == "ADOPT"
    assert C.after_rerun("ADOPT", False) == "INCONCLUSIVE"


def test_needs_rerun_only_for_adopted_gains_under_one_pp():
    assert C.needs_rerun("ADOPT", 0.99)
    assert not C.needs_rerun("ADOPT", 1.0)
    assert not C.needs_rerun("NOT ADOPTED", 0.5)


# ----------------------------------------------------------------------------- refusals (R6)


def test_constant_differing_from_preregistration_is_refused():
    assert C.constants_mismatch({"TP_ATR_MULT": 2.9}, {"TP_ATR_MULT": 2.8970711921574543}) \
        == ["TP_ATR_MULT: class 2.9 != frozen 2.8970711921574543"]
    assert C.constants_mismatch({"TP_ATR_MULT": 2.8970711921574543},
                                {"TP_ATR_MULT": 2.8970711921574543}) == []


def test_record_older_than_preregistration_commit_is_refused():
    stale = C.stale_records({"a_E00.json": 100.0, "a_E01.json": 300.0}, commit_ts=200)
    assert stale == ["a_E00.json"]


def test_two_arms_sharing_a_class_are_refused():
    assert C.shared_classes({"rev2": "X", "c4": "Y", "c2": "X"}) == ["X: c2, rev2"]
    assert C.shared_classes({"rev2": "X", "c4": "Y"}) == []
