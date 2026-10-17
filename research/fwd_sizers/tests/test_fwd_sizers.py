"""Forward sizer plans: fold f15 is the fwd1 window with f14's frozen state."""
from __future__ import annotations

import json
from pathlib import Path

FOLDS = Path(__file__).resolve().parents[3] / "evaluation" / "folds"


def test_f15_is_fwd1_window_with_f14_threshold():
    ref = json.loads((FOLDS / "plan_fwd_cs_aro.json").read_text())["folds"][0]
    for tag in ("cs_cmp", "cs_mix"):
        p = json.loads((FOLDS / f"plan_fwd_{tag}.json").read_text())
        f = p["folds"][0]
        assert f["fold_id"] == "f15" and f["strategy_class"].endswith("_f15")
        for k in ("train_timerange", "test_timerange"):
            assert f[k] == ref[k]
        f14 = json.loads((FOLDS / f"plan_{tag}.json").read_text())["thresholds"]["f14"]
        assert p["thresholds"] == {"f15": f14}
