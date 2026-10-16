"""035 T022: survivor classes and plans -- twin key = entry key != control key, one-variable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


STRAT = Path(__file__).resolve().parents[3]
FOLDS = STRAT / "evaluation" / "folds"
for p in (STRAT, STRAT / "NNNC"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.new_inputs import arms as A  # noqa: E402
from research.new_inputs import inputs as I  # noqa: E402


TAGS = [t for t in A.survivors() if (FOLDS / f"plan_{A.plan_tag(t)}.json").is_file()]
pytestmark = pytest.mark.skipif(not TAGS, reason="no survivor arms built")


def _key(plan_tag: str, i: int):
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    p = json.loads((FOLDS / f"plan_{plan_tag}.json").read_text())
    f = p["folds"][i]
    cls = load_strategy_class(p["base_strategy"])
    pinned = type(f["strategy_class"], (cls,), {"MODEL_TRAIN_TIMERANGE": f["train_timerange"],
                                                "MODEL_SEED": 1, "__module__": cls.__module__})
    return MK.key_hash(MK.resolve(pinned, load_config(STRAT / "config" / "config_wf.json")))


@pytest.mark.parametrize("i", (0, 14))
def test_twin_key_equals_entry_key_and_differs_from_the_control(i):
    ctl = _key("wf_aro_g24", i)
    seen = set()
    for t in TAGS:
        e, s = _key(A.plan_tag(t), i), _key(A.plan_tag(t, "sz"), i)
        assert e == s, t
        assert e != ctl and e not in seen, t
        seen.add(e)


def test_one_variable_against_the_control():
    from evaluation.fingerprint import load_strategy_class

    ctl, width = load_strategy_class("NNNC_A_Raw_Open"), load_strategy_class("NNNC_W_ARO")
    for t in TAGS:
        cls = load_strategy_class(A.class_name(t))
        cols = I.columns(t)
        assert cls.include_list == ctl.include_list + cols
        assert cls.pre_normalized_columns == ctl.pre_normalized_columns + cols
        assert cls.NEW_INPUTS_SHA == I.TABLE_SHA and cls.TRAINING_TYPE == ctl.TRAINING_TYPE
        sz = load_strategy_class(f"{A.class_name(t)}_SZ")
        assert sz.HELD_PAIR_ADD_RESOLVED == width.HELD_PAIR_ADD_RESOLVED
        assert sz.AGREE_RULE == "p_buy" and sz.include_list == cls.include_list
        for role, src in (("entry", "wf_aro_g24"), ("sz", "cs_aro")):
            p = json.loads((FOLDS / f"plan_{A.plan_tag(t, role)}.json").read_text())
            q = json.loads((FOLDS / f"plan_{src}.json").read_text())
            assert p["params_snapshot"] == q["params_snapshot"]
            assert [f["train_timerange"] for f in p["folds"]] == \
                [f["train_timerange"] for f in q["folds"]]
            assert not p.get("thresholds") or set(p["thresholds"]) == set(p["threshold_picks"])


def test_train_refusals():
    from research.new_inputs import train as TRN

    assert TRN.refusals("XX", {"f00"}, ["MS_VR"], None) == ["XX is not a survivor (['MS_VR'])"]
    assert TRN.refusals("MS_VR", {"f00", "f09"}, ["MS_VR"], None)[0].startswith("MS_VR: late")
    assert TRN.refusals("MS_VR", {"f09"}, ["MS_VR"], {"s2_winner": {"tag": "MS_VR"}}) == []


def test_s2_winner_rule():
    from research.new_inputs import separation as SEP

    rows = {"A": {"gap": 0.12, "degenerate_folds": []}, "B": {"gap": 0.15, "degenerate_folds": ["f0"] * 3},
            "C": {"gap": 0.11, "degenerate_folds": []}, "D": {"gap": 0.09, "degenerate_folds": []}}
    assert SEP.judge(rows, 0.1004)["winner"] == "A"
    assert rows["B"]["S2_pass"] is False and rows["D"]["S2_pass"] is False
    assert rows["C"]["margin"] == round(0.11 - 0.1004, 4)
    assert SEP.judge({"D": {"gap": 0.09, "degenerate_folds": []}}, 0.1004)["winner"] is None


def test_null_arm_is_the_control_pipeline_with_nothing_added():
    from evaluation.fingerprint import load_strategy_class
    from research.new_inputs import train as TRN

    for null, ctl in (("NNNC_NI_NULL", "NNNC_A_Raw_Open"), ("NNNC_NI_GBR_NULL", "NNNC_GbbRaw_Open")):
        if not (STRAT / "NNNC" / f"{null}.py").is_file():
            continue
        a, b = load_strategy_class(null), load_strategy_class(ctl)
        assert a.NEW_INPUTS == () and a.include_list == b.include_list
        assert a.pre_normalized_columns == b.pre_normalized_columns
    assert TRN.refusals("NULL", {"f00"}, [], None) == []


def test_both32_adds_raw8_and_composes_the_scalers_exactly():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler

    from evaluation.fingerprint import load_strategy_class
    from research.new_inputs import both32_scaler as BS
    from research.new_inputs import train as TRN

    from _raw8_features import RAW8_COLUMNS

    if (STRAT / "NNNC" / "NNNC_NI_B32.py").is_file():
        a, b = load_strategy_class("NNNC_NI_B32"), load_strategy_class("NNNC_A_Raw_Open")
        assert a.include_list == list(b.include_list) + list(RAW8_COLUMNS)
        assert a.pre_normalized_columns == b.pre_normalized_columns
    assert TRN.refusals("B32", {"f00"}, [], None) == []
    assert TRN.refusals("B32", {"f08"}, [], None)

    rng = np.random.default_rng(0)
    xa = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    xb = pd.DataFrame(rng.normal(size=(40, 3)), columns=["c", "d", "e"])
    sa, sb = RobustScaler().fit(xa), RobustScaler().fit(xb)
    comp = BS.compose([sa, sb])
    x = pd.concat([xa, xb], axis=1)
    assert np.array_equal(comp.transform(x), np.hstack([sa.transform(xa), sb.transform(xb)]))
