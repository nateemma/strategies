"""037 T036: the report renders every SC-004 heading, empty lists as "none", the SCREEN label, and a
persistence value on every candidate row."""
from __future__ import annotations

import json

from research.indicator_screen import report as R


def _flags(**k):
    return {"e0_agrees": None, "beats_width_both_halves": None, "residual_both_halves": None, **k}


def _res(col, pop="candidate", **flags):
    h = {"median_auc": 0.6, "median_strength": 0.1, "median_strat_auc": 0.58, "median_beat_width": 0.01, "n_windows": 1}
    return {"column": col, "target": "label:buy", "population": pop, "median_auc": 0.6, "median_strength": 0.1,
            "same_direction_share": 1.0, "halves": {"e0": h, "early": h, "late": h}, "width_sort_auc": 0.55,
            "candidate_flags": _flags(**flags)}


def test_empty_outputs_render_none(tmp_path):
    md = R.build(tmp_path)
    for k in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        assert f"**{k}" in md
    assert md.count("none") >= 5 and "SCREEN (descriptive)" in md


def test_lists_need_e0_agreement_and_rows_carry_persistence(tmp_path):
    sl = {"instrument": {"verdict": "PASS", "halves": ["e0", "early"], "rule": "r", "columns": {}},
          "results": [_res("good", beats_width_both_halves=True, residual_both_halves=True, e0_agrees=True),
                      _res("flip", beats_width_both_halves=True, residual_both_halves=True, e0_agrees=False)]}
    (tmp_path / "single_labels.json").write_text(json.dumps(sl))
    md = R.build(tmp_path)
    a = md.split("**(a)")[1].split("\n")[0]
    assert "`good`" in a and "`flip`" not in a
    assert "E0 agrees" in md and "E0 disagrees" in md


def test_report_is_deterministic(tmp_path):
    assert R.build(tmp_path) == R.build(tmp_path)
