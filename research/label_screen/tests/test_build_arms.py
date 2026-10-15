"""034 T004: generated classes and plans -- control keys unchanged, every candidate on its own key,
the sizer twin on the entry-source key, GFM on its existing weights, 4-dp thresholds, and feature
lists identical to the control's."""

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

from research.label_screen import candidates as C  # noqa: E402


KEYS = Path(__file__).parent / "fixtures" / "model_keys_pre034.json"
LS = [c for c in C.CANDIDATES if c.tag != "CONTROL"]


def _plan(tag):
    return json.loads((FOLDS / f"plan_{tag}.json").read_text())


def _cls(name):
    from evaluation.fingerprint import load_strategy_class

    return load_strategy_class(name)


def _cfg():
    from evaluation.fingerprint import load_config

    return load_config(STRAT / "config" / "config_wf.json")


def _key(plan, i):
    from Framework import ModelKey as MK

    f = plan["folds"][i]
    cls = _cls(plan["base_strategy"])
    pinned = type(f["strategy_class"], (cls,), {
        "MODEL_TRAIN_TIMERANGE": f["train_timerange"], "MODEL_SEED": 1,
        "__module__": cls.__module__})
    return MK.key_hash(MK.resolve(pinned, _cfg()))


@pytest.mark.parametrize("name", sorted(json.loads(KEYS.read_text())))
def test_existing_keys_unchanged(name):
    from Framework import ModelKey as MK

    assert MK.key_hash(MK.resolve(_cls(name), _cfg())) == json.loads(KEYS.read_text())[name]


@pytest.mark.parametrize("i", (0, 7))
def test_keys_distinct_twin_equal_and_gfm_reused(i):
    ctl = _key(_plan("wf_aro_g24"), i)
    seen = {}
    for c in LS:
        es = _key(_plan(C.plan_tag(c, "es")), i)
        sz = _key(_plan(C.plan_tag(c, "sz")), i)
        assert es == sz, c.tag
        if c.tag == "GFM":
            assert es == _key(_plan("sw_gfm_ctl_b030"), i)
            continue
        assert es != ctl and es not in seen.values(), c.tag
        seen[c.tag] = es


def test_thresholds_load_at_four_decimals():
    for c in LS:
        for name in (C.class_name(c), f"{C.class_name(c)}_SZ"):
            assert _cls(name).prediction_threshold.decimals == 4, name


def test_sizers_agree_on_p_buy():
    for c in C.CANDIDATES:
        assert _cls(f"{C.class_name(c)}_SZ").AGREE_RULE == "p_buy", c.tag


def test_labels_and_feature_lists():
    from Framework import TrainingSignals as T

    base = _cls("NNNC_A_Raw_Open")
    for c in LS:
        cls = _cls(C.class_name(c))
        assert int(cls.TRAINING_TYPE) == c.method
        if c.tag != "GFM":
            assert cls.LABEL_PARAMS == c.label_params and cls.LABEL_PARAMS_SELL == c.label_params_sell
        for attr in ("HORIZON", "MIN_BUY_GAIN_THRESHOLD", "MIN_SELL_LOSS_THRESHOLD", "seq_len",
                     "classifier_type"):
            assert getattr(cls, attr) == getattr(base, attr), (c.tag, attr)
        assert T.LabelMethod(c.method) is not None  # a real method id


LABEL_ONLY = {"MODEL_ENTRY_FAMILY", "TRAINING_TYPE", "LABEL_PARAMS", "LABEL_PARAMS_SELL", "prediction_threshold",
              "get_train_buy_signals", "get_train_sell_signals", "_label_params", "_filtered",
              "MODEL_KEY_ATTRIBUTES", "__doc__", "__module__", "__dict__", "__weakref__",
              "__qualname__", "__firstlineno__", "__static_attributes__"}


def test_only_label_names_differ_from_the_control():
    """One variable: every name a candidate class defines beyond the control's own MRO is a label
    name, so features, scalers, entries and exits are inherited unchanged (research R10)."""
    base = _cls("NNNC_A_Raw_Open")
    for c in LS:
        if c.tag == "GFM":
            continue
        cls = _cls(C.class_name(c))
        extra = set()
        for k in cls.__mro__:
            if k in base.__mro__:
                break
            extra |= set(vars(k))
        # names the framework writes per class at creation (__init_subclass__, ABC) must equal the
        # control's own values
        for name in sorted(extra - LABEL_ONLY):
            if name in ("_abc_impl", "__abstractmethods__", "__annotations__"):
                continue
            assert getattr(cls, name) == getattr(base, name), (c.tag, name)


def test_family_pinned_to_the_controls():
    base = _cls("NNNC_A_Raw_Open")
    assert base._model_family() == "reversion"
    width = _cls("NNNC_W_ARO")  # the width book turns the held-pair add off
    for c in LS:
        for name, ref in ((C.class_name(c), base), (f"{C.class_name(c)}_SZ", width)):
            cls = _cls(name)
            assert cls._model_family() == "reversion", name
            assert cls.HELD_PAIR_ADD_RESOLVED == ref.HELD_PAIR_ADD_RESOLVED, name
            assert cls.FAMILY_EXITS == ref.FAMILY_EXITS, name


def test_plans_copy_the_control():
    ctl = _plan("wf_aro_g24")
    for c in LS:
        for role in ("es", "sz"):
            p = _plan(C.plan_tag(c, role))
            # the control's thresholds are not copied; only pick.py writes them, with a pick record
            assert set(p.get("thresholds", {})) == set(p.get("threshold_picks", {}))
            assert p["params_snapshot"] == ctl["params_snapshot"]
            assert [f["train_timerange"] for f in p["folds"]] == \
                [f["train_timerange"] for f in ctl["folds"]]
