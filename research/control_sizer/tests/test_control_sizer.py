"""Control-sizer study: the verdict rule, record refusal, and the arm's weights and plan."""

from __future__ import annotations

import json
import os
from pathlib import Path

from research.control_sizer import judge as J


PR = {"arm": "cs_aro", "folds": ["f00", "f08"], "small_margin_pp": 1.0}
OK = {"pass": True}


def _r(verdict, e, l):
    return {"verdict": verdict, "early_gain_pp": e, "late_gain_pp": l, "failed": []}


def test_pass_needs_both_comparators():
    assert J.decide(PR, OK, _r("PASS", 3, 2), _r("PASS", 2, 4))["decision"] == "PASS (PROVISIONAL)"
    assert J.decide(PR, OK, _r("PASS", 3, 2), _r("FAIL", -1, 4))["decision"] == "FAIL"
    assert J.decide(PR, OK, _r("INCONCLUSIVE", 3, 2), _r("PASS", 2, 4))["decision"] == "INCONCLUSIVE"


def test_small_margin_and_instrument():
    assert J.decide(PR, OK, _r("PASS", 3, 0.5), _r("PASS", 2, 4))["decision"] == "INCONCLUSIVE"
    r = J.decide(PR, {"pass": False}, None, None)
    assert r["decision"] == "INCONCLUSIVE" and "no P&L" in r["reason"]


def test_refuses_a_record_older_than_the_commit(tmp_path):
    p = tmp_path / "cs_aro_f08_F08.json"
    p.write_text("{}")
    os.utime(p, (1000, 1000))
    assert J.refusals(PR, commit_ts=2000.0, arms_dir=tmp_path) == [
        "record older than the pre-registration commit: cs_aro_f08_F08.json"]
    os.utime(p, (3000, 3000))
    assert J.refusals(PR, commit_ts=2000.0, arms_dir=tmp_path) == []


def test_arm_loads_the_controls_weights_and_is_the_034_sizer():
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    strat = Path(__file__).resolve().parents[3]
    cfg = load_config(strat / "config" / "config_wf.json")
    plans = {t: json.loads((strat / "evaluation" / "folds" / f"plan_{t}.json").read_text())
             for t in ("cs_aro", "wf_aro_g24")}

    def key(t, i):
        f = plans[t]["folds"][i]
        cls = load_strategy_class(plans[t]["base_strategy"])
        pinned = type(f["strategy_class"], (cls,), {
            "MODEL_TRAIN_TIMERANGE": f["train_timerange"], "MODEL_SEED": 1,
            "__module__": cls.__module__})
        return MK.key_hash(MK.resolve(pinned, cfg))

    for i in (0, 14):
        assert key("cs_aro", i) == key("wf_aro_g24", i)
    arm, ref = load_strategy_class("NNNC_CS_ARO"), load_strategy_class("NNNC_LS_CONTROL_SZ")
    assert arm.AGREE_RULE == "p_buy" and arm.AGREE_SHARE == 0.5 and arm.STAKE_WEIGHTS == (1.0, 0.5)
    assert arm.HELD_PAIR_ADD_RESOLVED == load_strategy_class("NNNC_W_ARO").HELD_PAIR_ADD_RESOLVED
    assert arm.WIDTH_CUTS == ref.WIDTH_CUTS
    assert plans["cs_aro"]["params_snapshot"] == plans["wf_aro_g24"]["params_snapshot"]


def test_picker_defaults_to_the_cs_aro_plan():
    import inspect

    from research.control_sizer import pick as P

    assert P.PLAN.name == "plan_cs_aro.json"
    for f in (P.pick, P.verify):
        assert inspect.signature(f).parameters["plan_path"].default == P.PLAN
