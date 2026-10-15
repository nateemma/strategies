"""035 T019 / T021: the partial Spearman statistic, the early-only guard and the survivor rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from research.new_inputs import screen as SC
from research.new_inputs import survivors as SV
from research.new_inputs import trades as TR


RULE = {"min_coverage": 0.9, "max_rho_width": 0.5, "min_partial": 0.10, "min_sign_folds": 6,
        "max_survivors": 3}


def test_partial_spearman_by_hand_and_zero_for_a_control():
    rng = np.random.default_rng(0)
    w = rng.normal(size=500)
    pb = rng.normal(size=500)
    x = rng.normal(size=500)
    y = x + 0.5 * w + rng.normal(size=500)
    r = SC.partial_spearman(x, y, [w, pb])
    # by hand: residualise ranks on [1, rank w, rank pb], correlate
    from scipy.stats import rankdata

    X = np.column_stack([np.ones(500), rankdata(w), rankdata(pb)])

    def res(v):
        rv = rankdata(v)
        return rv - X @ np.linalg.lstsq(X, rv, rcond=None)[0]

    ref = np.corrcoef(res(x), res(y))[0, 1]
    assert r == pytest.approx(ref) and r > 0.4
    assert SC.partial_spearman(w, y, [w, pb]) == 0.0  # an input equal to bb_width adds nothing


def test_screen_refuses_late_folds():
    t = pd.DataFrame({"fold": ["f02", "f09"], "pp": [1.0, 0.0], "agree_ctl_share": [True, False]})
    with pytest.raises(RuntimeError, match="early half only"):
        TR.check(t)


def _row(cov=1.0, rw=0.1, p=0.2, sf=7):
    return {"coverage": cov, "rho_width": rw, "partial": p, "sign_folds": sf}


def test_survivor_rule():
    scr = {"inputs": {"A": _row(p=0.30), "B": _row(p=-0.25), "C": _row(p=0.20), "D": _row(p=0.15),
                      "LOW": _row(cov=0.5), "WIDE": _row(rw=0.6, p=0.4), "WEAK": _row(p=0.05),
                      "FLIP": _row(sf=5, p=0.3), "S0": {"status": "INCONCLUSIVE", "reason": "x"}}}
    r = SV.select(scr, RULE)
    assert r["survivors"] == ["A", "B", "C"]
    st = {k: v["status"] for k, v in r["inputs"].items()}
    assert st["D"] == "FAIL" and "top 3" in r["inputs"]["D"]["reason"]
    assert st["LOW"] == "INCONCLUSIVE" and st["S0"] == "INCONCLUSIVE"
    assert st["WIDE"] == st["WEAK"] == st["FLIP"] == "FAIL"
