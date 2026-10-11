"""025 T009 (contract C5): the stage runner chooses on f00-f07 ONLY and judges on f08-f14.

`select` sees only the early-fold fields of each candidate row; `decide` adopts the choice only when
the late paired verdict is IMPROVED. A planted candidate that is best late and worst early must not
be chosen.
"""

from __future__ import annotations

import pytest
from evaluation import stage as St


def _row(tag, early, late=0.0, lost_other=0, rev_early=0.0, degenerate=False):
    return {"tag": tag, "early_sum": early, "late_sum": late, "lost_other_early": lost_other,
            "reversion_early_pp": rev_early, "degenerate": degenerate}


def test_selection_ignores_late_folds():
    rows = [_row("a", early=5.0, late=-10.0), _row("planted", early=-5.0, late=+99.0)]
    assert St.select(rows, "coexistence")["tag"] == "a"


def test_degenerate_candidates_are_excluded():
    rows = [_row("deg", early=50.0, degenerate=True), _row("ok", early=1.0)]
    assert St.select(rows, "coexistence")["tag"] == "ok"
    assert St.select([_row("deg", early=50.0, degenerate=True)], "coexistence") is None


def test_tie_breaks_are_stage_specific():
    rows = [
        _row("x", 3.0, lost_other=100, rev_early=9.0),
        _row("y", 3.0, lost_other=50, rev_early=1.0),
    ]
    assert St.select(rows, "coexistence")["tag"] == "y"      # fewer lost to the other family
    assert St.select(rows, "tune2")["tag"] == "x"            # higher reversion early contribution


def test_select_reads_no_late_field():
    class Guard(dict):
        def __getitem__(self, k):
            if k.startswith("late"):
                raise AssertionError(f"selection read {k}")
            return super().__getitem__(k)

    rows = [Guard(_row("a", 1.0)), Guard(_row("b", 2.0))]
    assert St.select(rows, "tune1")["tag"] == "b"


@pytest.mark.parametrize(
    "verdict, adopted, code",
    [("IMPROVED", True, 0), ("UNCHANGED", False, 3), ("INCONCLUSIVE_ARM_LOSES", False, 3)],
)
def test_adoption_requires_late_improved(verdict, adopted, code):
    late = {"verdict": verdict, "delta_pp": 1.0, "ci_pp_yr": [0, 1], "p": 0.5}
    d = St.decide(_row("a", 1.0), late)
    assert d["adopted"] is adopted and St.exit_code(d) == code


def test_no_eligible_candidate_is_not_adopted():
    d = St.decide(None, None)
    assert d["adopted"] is False and St.exit_code(d) == 3


def test_missing_input_exits_2(tmp_path):
    rc = St.main(["--stage", "tune1", "--previous", "nope_arm", "--candidates", "nope_a",
                  "--singles", "reversion=st_gbb,momentum=st_bgo_mx",
                  "--out", str(tmp_path / "d.md")])
    assert rc == 2


def test_decision_block_is_appended(tmp_path):
    out = tmp_path / "decisions.md"
    rows = [_row("a", 1.0, lost_other=3), _row("b", 2.0, lost_other=1)]
    d = St.decide(rows[1], {"verdict": "UNCHANGED", "delta_pp": 0.1, "ci_pp_yr": [-1, 1], "p": 0.9})
    liveness = {"a": "refusals 10", "b": "refusals 20"}
    St.write_block(out, "coexistence", "st_prev", rows, d, liveness)
    St.write_block(out, "tune1", "st_prev", rows, d, liveness)
    text = out.read_text()
    assert text.count("## Stage: ") == 2
    for needle in ("| a |", "| b |", "chosen: `b`", "UNCHANGED", "adopted: **no**", "refusals 20",
                   "PROVISIONAL"):
        assert needle in text, needle
