"""040 T053: the walk-forward training window ends before the test window (foldplan PURGE_MIN).

freqtrade keeps `date <= stop`, so a train_timerange ending AT test_start trained on the first test
candle. Purged plans (`train_purge_days`) must stop >= 48 bars before test_start; legacy plans keep
their recorded windows so their records reproduce.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from evaluation import foldplan as FP  # noqa: E402

LEGACY = STRAT / "evaluation" / "folds" / "plan_wf_aro_g24.json"


def _test_start(f):
    return datetime.strptime(f["test_start"], "%Y-%m-%d")


def test_build_stops_training_at_least_48_bars_before_every_test_window():
    folds = FP.build("NNNC", "2021-01-01", "2026-09-07")
    assert len(folds) == 15
    for f in folds:
        gap = _test_start(f) - FP._train_stop(f)
        assert gap >= timedelta(minutes=15 * 48), f
        assert gap == timedelta(days=FP.DEFAULT_PURGE_DAYS)
        assert f["train_end"] < f["test_start"]
    assert not FP.validate(folds, purged=True)


def test_a_purged_plan_whose_training_reaches_the_test_window_is_refused():
    folds = FP.build("NNNC", "2021-01-01", "2026-09-07")
    folds[3]["train_timerange"] = folds[3]["train_timerange"].split("-")[0] + "-" + \
        folds[3]["test_start"].replace("-", "")
    errors = FP.validate_plan({"train_purge_days": 1, "folds": folds})
    assert len(errors) == 1 and errors[0].startswith("f03: train_timerange")


def test_legacy_plans_keep_their_windows_and_still_validate():
    plan = json.loads(LEGACY.read_text())
    assert "train_purge_days" not in plan
    f0 = plan["folds"][0]
    assert f0["train_timerange"].split("-")[1] == f0["test_start"].replace("-", "")  # the old, unpurged window
    assert not FP.validate_plan(plan)


def test_purge_makes_an_independent_twin_with_shortened_windows_and_no_old_picks():
    src = json.loads(LEGACY.read_text())
    assert src.get("thresholds")
    p = FP.purge(src, "pw")
    assert p["validated"] and p["train_purge_days"] == 1 and p["purged_from"] == src["run_id"]
    assert "thresholds" not in p and "threshold_picks" not in p
    for a, b in zip(src["folds"], p["folds"]):
        assert b["test_timerange"] == a["test_timerange"]
        assert _test_start(b) - FP._train_stop(b) == timedelta(days=1)
        assert b["train_timerange"] != a["train_timerange"]
        assert b["strategy_class"] == a["strategy_class"] + "_PW"
        assert b["model_dir"] == a["model_dir"] + "_PW"
        assert b["arm_tag"] == a["arm_tag"] + "_pw"
    assert json.loads(LEGACY.read_text()) == src  # the legacy plan is untouched


def test_purge_refuses_an_already_purged_plan_and_derive_keeps_the_field():
    p = FP.purge(json.loads(LEGACY.read_text()), "pw")
    try:
        FP.purge(p, "pw2")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    d = FP.derive(p, "SimpleTrigger", "SimpleTrigger_X", "x")
    assert d["train_purge_days"] == 1 and d["validated"]
