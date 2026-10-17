"""037 addendum 3 (mofn.py): M is chosen on early windows only; confirmation rules."""
from __future__ import annotations

import numpy as np

from research.indicator_screen import mofn as M

PR = {"pairs": {"min_windows": 2}}


def _per(early, late, e0, wb=(1.1, 1.1, 1.0), fire=(0.01, 0.002)):
    per = {}
    for i in range(3):
        per[f"e{i}"] = {"half": "early", "lift": list(early), "fire": list(fire), "wb": wb[0], "wb_fire": 0.002,
                        "single": {"a": 1.2}}
        per[f"l{i}"] = {"half": "late", "lift": list(late), "fire": list(fire), "wb": wb[1], "wb_fire": 0.002,
                        "single": {"a": 1.2}}
    per["E0"] = {"half": "e0", "lift": list(e0), "fire": list(fire), "wb": wb[2], "wb_fire": 0.002, "single": {"a": 1.0}}
    return per


def test_choice_reads_early_only():
    a = M.choose_m(_per(early=(1.2, 1.5), late=(9.0, 0.1), e0=(9.0, 9.0)), 2, PR)
    assert a["M_best"] == 2 and a["M_match"] == 2


def test_confirm_needs_late_above_width_and_singles_and_e0():
    per = _per(early=(1.2, 1.5), late=(1.3, 1.4), e0=(1.1, 1.05))
    assert M.confirm(per, 2, ["a"])["confirmed"] == "yes"
    assert M.confirm(_per((1.2, 1.5), (1.3, 1.4), (1.1, 0.9)), 2, ["a"])["confirmed"] == "early+late only"
    assert M.confirm(_per((1.2, 1.5), (1.3, 1.15), (1.1, 1.2)), 2, ["a"])["confirmed"] == "no"
