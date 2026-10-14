"""033 T014: the generated arms -- same weights for both variants, the width book's cuts, and a
params snapshot that differs from the control's only in the exit switch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


STRAT = Path(__file__).resolve().parents[3]
ROOT = STRAT.parent.parent
FOLDS = STRAT / "evaluation" / "folds"
for p in (STRAT, STRAT / "NNNC"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CASES = (("ARO", "wf_aro_g24"), ("GBR", "fs_gbr_gr"))


def _plan(tag):
    return json.loads((FOLDS / f"plan_{tag}.json").read_text())


def _key(plan, fold):
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cls = load_strategy_class(plan["base_strategy"])
    pinned = type(fold["strategy_class"], (cls,), {
        "MODEL_TRAIN_TIMERANGE": fold["train_timerange"], "MODEL_SEED": 1,
        "__module__": cls.__module__})
    return MK.key_hash(MK.resolve(pinned, load_config(ROOT / plan["config"])))


@pytest.mark.parametrize("m,ctl", CASES)
def test_both_variants_share_the_model_key_and_differ_from_the_comparators(m, ctl):
    sz, szx = _plan(f"wr_{m.lower()}_sz"), _plan(f"wr_{m.lower()}_szx")
    for i in (0, 14):
        k = _key(sz, sz["folds"][i])
        assert k == _key(szx, szx["folds"][i])
        assert k != _key(_plan(ctl), _plan(ctl)["folds"][i])
        assert k != _key(_plan(f"co_{m.lower()}"), _plan(f"co_{m.lower()}")["folds"][i])


@pytest.mark.parametrize("m,ctl", CASES)
def test_cuts_are_the_width_books_and_snapshots_differ_only_in_the_exit_switch(m, ctl):
    from evaluation.fingerprint import load_strategy_class

    assert (load_strategy_class(f"NNNC_WR_{m}").WIDTH_CUTS
            == load_strategy_class(f"NNNC_W_{m}").WIDTH_CUTS)
    c, sz, szx = _plan(ctl), _plan(f"wr_{m.lower()}_sz"), _plan(f"wr_{m.lower()}_szx")
    assert sz["params_snapshot"] == c["params_snapshot"] and "thresholds" not in sz
    a = json.loads((ROOT / c["params_snapshot"]).read_text())
    b = json.loads((ROOT / szx["params_snapshot"]).read_text())
    assert a["params"]["sell"].get("enable_exit_signal") is False
    b["params"]["sell"]["enable_exit_signal"] = False
    assert a == b
    assert [f["train_timerange"] for f in sz["folds"]] == [f["train_timerange"] for f in c["folds"]]
