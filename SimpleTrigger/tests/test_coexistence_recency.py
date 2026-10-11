"""025 T004: rule B (momentum-recency guard on reversion entries) and the MRO merge of overrides.

Frame rows (gbb: guard_metric < 0.3 & bb_width > 0.035; breakout_gbb: guard_metric > 0.2 & bb_width
> 0.035):
    row 2 -> breakout_gbb only (guard 0.9)
    rows 3, 6, 9 -> gbb only (guard 0.1)
    row 11 -> both (guard 0.25), an overlap row
every other row fires nothing (narrow band).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from SimpleTrigger.presets import PresetError
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


def _frame(n=12):
    gm = np.full(n, 0.5)
    bw = np.full(n, 0.01)
    gm[2], bw[2] = 0.9, 0.05
    for r in (3, 6, 9):
        gm[r], bw[r] = 0.1, 0.05
    if n > 11:
        gm[11], bw[11] = 0.25, 0.05
    return pd.DataFrame({"guard_metric": gm, "bb_width": bw})


def _arm(**attrs):
    attrs.setdefault("PRESET", "ARM_D_FAMILY")
    attrs.setdefault("TRIGGERS", ("gbb", "breakout_gbb"))
    return object.__new__(type("Coex", (SimpleTriggerStrategy,), attrs))


def _run(s, df=None):
    mask = s.get_entry_conditions(_frame() if df is None else df)
    return np.flatnonzero(mask).tolist(), s._st_tags.tolist()


def test_zero_window_is_024_behaviour():
    assert _run(_arm(MOMENTUM_RECENCY_BARS=0)) == _run(_arm())
    rows, tags = _run(_arm())
    assert rows == [2, 3, 6, 9, 11]
    assert tags[11] == "reversion:gbb+ov"


def test_window_2_drops_the_reversion_entry_one_bar_after_momentum():
    rows, tags = _run(_arm(MOMENTUM_RECENCY_BARS=2))
    assert rows == [2, 6, 9, 11]
    assert tags[3] == ""


def test_window_5_reaches_back_further():
    rows, _ = _run(_arm(MOMENTUM_RECENCY_BARS=5))
    assert rows == [2, 9, 11]       # 3 and 6 are within 5 bars (current included) of row 2


def test_momentum_rows_are_never_dropped_and_overlap_goes_to_momentum():
    rows, tags = _run(_arm(MOMENTUM_RECENCY_BARS=48))
    assert 2 in rows and 11 in rows
    assert tags[2] == "momentum:breakout_gbb"
    assert tags[11] == "momentum:breakout_gbb+ov"   # G2: rule B gives overlap to momentum
    assert all(t == "" for i, t in enumerate(tags) if i not in (2, 11))


def test_rule_is_causal():
    # appending rows never changes a decision on an earlier row
    short = _run(_arm(MOMENTUM_RECENCY_BARS=5), _frame(10))
    full = _run(_arm(MOMENTUM_RECENCY_BARS=5), _frame(12))
    assert [r for r in full[0] if r < 10] == short[0]
    assert full[1][:10] == short[1]


@pytest.mark.parametrize("attrs, match", [
    ({"TRIGGERS": None, "TRIGGER_NAME": "gbb", "MOMENTUM_RECENCY_BARS": 4}, "momentum"),
    ({"MOMENTUM_RECENCY_BARS": -1}, "MOMENTUM_RECENCY_BARS"),
    ({"MOMENTUM_RECENCY_BARS": 2.5}, "MOMENTUM_RECENCY_BARS"),
])
def test_bad_rule_b_declarations_are_refused(attrs, match):
    with pytest.raises(PresetError, match=match):
        type("Bad", (SimpleTriggerStrategy,), {"PRESET": "ARM_D_FAMILY",
                                                "TRIGGERS": ("gbb", "breakout_gbb"), **attrs})


def test_sink_records_dropped_rows(monkeypatch, tmp_path):
    from Framework.BaseStrategy import BaseStrategy

    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))

    def fake_base(self, df, metadata):
        mask = self.get_entry_conditions(df)
        df["enter_tag"] = ""
        df.loc[mask, "enter_tag"] = "model_entry "
        return df

    monkeypatch.setattr(BaseStrategy, "populate_entry_trend", fake_base)
    s = _arm(MOMENTUM_RECENCY_BARS=5)
    type(s).populate_entry_trend(s, _frame(), {"pair": "SOL/USDT"})
    files = list(tmp_path.glob("*_coexist.json"))
    assert len(files) == 1
    counts = json.loads(files[0].read_text())["counts"]
    assert counts == {"coexist_recency_refused": {"pass": 2, "total": 12}}


def test_family_exit_overrides_merge_along_the_mro():
    # G1: each tuning step states only its new parameter; earlier steps' values are inherited.
    child = type("Step1", (SimpleTriggerStrategy,), {
        "PRESET": "ARM_D_FAMILY", "TRIGGERS": ("gbb", "breakout_gbb"),
        "FAMILY_EXIT_OVERRIDES": {"reversion": {"take_profit": 0.03}}})
    grandchild = type("Step2", (child,), {
        "FAMILY_EXIT_OVERRIDES": {"reversion": {"unclog_12h_hours": 6.0}}})
    assert grandchild.FAMILY_EXITS["reversion"] == {"take_profit": 0.03, "unclog_12h_hours": 6.0}
    assert grandchild.FAMILY_EXITS["momentum"] == {"take_profit": 10.0, "unclog_12h_hours": 24.0}
    assert child.FAMILY_EXITS["reversion"] == {"take_profit": 0.03}
