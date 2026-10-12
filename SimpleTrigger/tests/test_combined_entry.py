"""024 T019: a combined arm ORs its triggers, tags each row by family priority, and marks overlap.

`gbb` fires where guard_metric < 0.3 and bb_width > 0.035; `breakout_gbb` where guard_metric > 0.2
and bb_width > 0.035. Both fire where 0.2 < guard_metric < 0.3 (the overlap zone, 023 Addendum 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


#        row:        0      1      2      3      4
GM = [0.10, 0.25, 0.50, 0.25, 0.90]  # gbb only / both / bgo only / both but narrow / bgo only
BW = [0.05, 0.05, 0.05, 0.01, 0.05]


def _frame():
    return pd.DataFrame({"guard_metric": GM, "bb_width": BW})


def _arm(**attrs):
    attrs.setdefault("PRESET", "ARM_D_FAMILY")
    cls = type("Combo", (SimpleTriggerStrategy,), attrs)
    return object.__new__(cls)  # never instantiate a strategy in a unit test


def _run(s):
    mask = s.get_entry_conditions(_frame())
    return mask.tolist(), s._st_tags.tolist()


def test_mask_is_the_or_and_overlap_goes_to_reversion_by_default():
    mask, tags = _run(_arm(TRIGGERS=("gbb", "breakout_gbb")))
    assert mask == [True, True, True, False, True]
    assert tags == [
        "reversion:gbb",
        "reversion:gbb+ov",
        "momentum:breakout_gbb",
        "",
        "momentum:breakout_gbb",
    ]


def test_reversed_priority_gives_overlap_to_momentum():
    s = _arm(
        TRIGGERS=("gbb", "breakout_gbb"), FAMILY_PRIORITY=("momentum", "reversion", "unassigned")
    )
    _, tags = _run(s)
    assert tags[1] == "momentum:breakout_gbb+ov"
    assert tags[0] == "reversion:gbb"  # no overlap -> no suffix


def test_disabled_side_is_exactly_the_other_single_trigger():
    combo = _arm(TRIGGERS=("gbb", "breakout_gbb"), DISABLED_TRIGGERS=("breakout_gbb",))
    single = _arm(TRIGGER_NAME="gbb")
    assert _run(combo) == _run(single)
    combo = _arm(TRIGGERS=("gbb", "breakout_gbb"), DISABLED_TRIGGERS=("gbb",))
    single = _arm(TRIGGER_NAME="breakout_gbb")
    assert _run(combo) == _run(single)


def test_trigger_name_is_the_single_trigger_form():
    mask, tags = _run(_arm(TRIGGER_NAME="breakout_gbb"))
    assert mask == [False, True, True, False, True]
    assert "+ov" not in "".join(tags)  # one trigger: overlap is impossible


@pytest.mark.parametrize(
    "attrs, match",
    [
        ({"TRIGGER_NAME": "gbb", "TRIGGERS": ("gbb", "breakout_gbb")}, "both"),
        ({"TRIGGERS": ("gbb",), "DISABLED_TRIGGERS": ("gbb",)}, "every trigger"),
        ({"TRIGGERS": ("gbb", "x"), "DISABLED_TRIGGERS": ("y",)}, "not in TRIGGERS"),
        ({"TRIGGERS": ("gbb",), "FAMILY_PRIORITY": ("reversion", "momentum")}, "permutation"),
    ],
)
def test_bad_declarations_are_refused_at_class_creation(attrs, match):
    from SimpleTrigger.presets import PresetError

    with pytest.raises(PresetError, match=match):
        type("Bad", (SimpleTriggerStrategy,), attrs)


def test_an_unrunnable_trigger_refuses_the_whole_arm(capsys):
    # random_matched FAILS causality: a combined arm containing it must enter nothing,
    # not silently run the other side alone.
    s = _arm(TRIGGERS=("gbb", "random_matched"))
    mask = s.get_entry_conditions(_frame())
    assert not np.asarray(mask).any()
    assert "enters NO trades" in capsys.readouterr().out
