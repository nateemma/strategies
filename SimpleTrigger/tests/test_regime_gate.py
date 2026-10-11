"""026 T009: the reversion regime gate in SimpleTrigger (contract C3).

Rows (gbb: guard_metric < 0.3 & bb_width > 0.035; breakout_gbb: guard_metric > 0.2 & bb_width >
0.035): row 1 gbb only, row 2 breakout_gbb only, row 3 both (overlap), rows 4-5 gbb only.
The regime state is stubbed; it is "off" (>= 0.5) on rows 1, 2, 3 and NaN on row 5.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest
from SimpleTrigger.presets import PresetError
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy
from utils import RegimeState as RS


DATES = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
STATE = pd.Series([0.1, 0.9, 0.9, 0.9, 0.1, np.nan], index=DATES)
GATE = {"instrument": "volreg", "threshold": 0.5, "off_when": "ge"}


def _frame():
    gm = [0.5, 0.1, 0.9, 0.25, 0.1, 0.1]
    bw = [0.01, 0.05, 0.05, 0.05, 0.05, 0.05]
    return pd.DataFrame({"date": DATES, "guard_metric": gm, "bb_width": bw})


def _arm(gate=GATE, startup=4096, **attrs):
    attrs.setdefault("PRESET", "ARM_D_FAMILY")
    attrs.setdefault("TRIGGERS", ("gbb", "breakout_gbb"))
    cls = type(
        "Gated",
        (SimpleTriggerStrategy,),
        {
            "REVERSION_REGIME_GATE": gate,
            "ATTR_OVERRIDES": {"startup_candle_count": startup},
            **attrs,
        },
    )
    s = object.__new__(cls)
    s.dp = types.SimpleNamespace(
        current_whitelist=lambda: ["A/USDT"], get_pair_dataframe=lambda pair, timeframe: _frame()
    )
    return s


@pytest.fixture(autouse=True)
def stub_state(monkeypatch):
    monkeypatch.setattr(RS, "compute", lambda inst, frames, btc: STATE)


def _run(s):
    mask = s.get_entry_conditions(_frame())
    return np.flatnonzero(mask).tolist(), s._st_tags.tolist()


def test_no_gate_is_025_behaviour():
    assert _run(_arm(gate=None, startup=64)) == _run(_arm(gate=None))
    rows, tags = _run(_arm(gate=None))
    assert rows == [1, 2, 3, 4, 5] and tags[3] == "reversion:gbb+ov"


def test_off_state_drops_reversion_only_rows_and_keeps_momentum():
    rows, tags = _run(_arm())
    assert 1 not in rows  # gbb only, state off -> dropped
    assert 2 in rows and tags[2] == "momentum:breakout_gbb"  # momentum untouched
    assert 4 in rows and tags[4] == "reversion:gbb"  # state on
    assert 5 in rows and tags[5] == "reversion:gbb"  # NaN state -> on


def test_overlap_rows_in_off_go_to_momentum():
    rows, tags = _run(_arm())
    assert 3 in rows and tags[3] == "momentum:breakout_gbb+ov"


def test_off_when_lt_flips_the_side():
    rows, _ = _run(_arm(gate={**GATE, "off_when": "lt"}))
    assert 1 in rows and 4 not in rows


def test_sink_records_refusals_and_off_bars(monkeypatch, tmp_path):
    from Framework.BaseStrategy import BaseStrategy

    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))

    def fake_base(self, df, metadata):
        mask = self.get_entry_conditions(df)
        df["enter_tag"] = ""
        df.loc[mask, "enter_tag"] = "model_entry "
        return df

    monkeypatch.setattr(BaseStrategy, "populate_entry_trend", fake_base)
    s = _arm()
    type(s).populate_entry_trend(s, _frame(), {"pair": "A/USDT"})
    counts = json.loads(next(tmp_path.glob("*_coexist.json")).read_text())["counts"]
    assert counts["regime_gate_refused"] == {"pass": 1, "total": 6}  # row 1
    assert counts["regime_gate_retagged"] == {"pass": 1, "total": 6}  # row 3
    assert counts["regime_gate_off_bars"] == {"pass": 3, "total": 6}  # rows 1-3


@pytest.mark.parametrize(
    "gate, startup, attrs, match",
    [
        ({**GATE, "instrument": "moon"}, 4096, {}, "instrument"),
        ({**GATE, "off_when": "gt"}, 4096, {}, "off_when"),
        ({**GATE, "threshold": float("nan")}, 4096, {}, "threshold"),
        (GATE, 64, {}, "startup_candle_count"),
        (GATE, 4096, {"TRIGGERS": None, "TRIGGER_NAME": "breakout_gbb"}, "reversion"),
    ],
)
def test_bad_gate_declarations_are_refused(gate, startup, attrs, match):
    with pytest.raises(PresetError, match=match):
        type(
            "Bad",
            (SimpleTriggerStrategy,),
            {
                "PRESET": "ARM_D_FAMILY",
                "TRIGGERS": ("gbb", "breakout_gbb"),
                "REVERSION_REGIME_GATE": gate,
                "ATTR_OVERRIDES": {"startup_candle_count": startup},
                **attrs,
            },
        )


def test_btctrend_requests_btc_as_informative_pair():
    s = _arm(gate={**GATE, "instrument": "btctrend"})
    from SimpleTrigger.presets import REGIME_CONTEXT_PAIR

    assert (REGIME_CONTEXT_PAIR, s.timeframe) in s.informative_pairs()
    assert _arm().informative_pairs() == []
