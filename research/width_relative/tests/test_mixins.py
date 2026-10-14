"""033 T009/T011: WidthRelativeMixin (model key, label spec) and WidthSizerMixin (tags, stakes,
4-decimal threshold) -- contract C2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[3]
for p in (STRAT, STRAT / "NNNC", STRAT / "Framework", STRAT / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KEYS = STRAT / "Framework" / "tests" / "fixtures" / "model_keys_pre033.json"


def _cfg():
    from evaluation.fingerprint import load_config

    return load_config(STRAT / "config" / "config_wf.json")


def _key(cls):
    from Framework import ModelKey as MK

    return MK.key_hash(MK.resolve(cls, _cfg()))


def _wr(base_name, **attrs):
    from _width_relative import WidthRelativeMixin
    from evaluation.fingerprint import load_strategy_class

    base = load_strategy_class(base_name)
    return type(f"WR_{base_name}", (WidthRelativeMixin, base), dict(attrs))


# --- WidthRelativeMixin ---------------------------------------------------------------------


def test_width_relative_key_differs_and_tracks_label_settings():
    keys = json.loads(KEYS.read_text())
    k = _key(_wr("NNNC_A_Raw_Open"))
    assert k not in (keys["NNNC_A_Raw_Open"], keys["NNNC_CO_ARO"])
    assert _key(_wr("NNNC_A_Raw_Open", WR_BUCKETS=8)) != k
    assert _key(_wr("NNNC_A_Raw_Open", WR_SHARE=0.4)) != k
    assert _key(_wr("NNNC_A_Raw_Open")) == k  # deterministic


@pytest.mark.parametrize("name", sorted(json.loads(KEYS.read_text())))
def test_existing_keys_unchanged(name):
    from evaluation.fingerprint import load_strategy_class

    assert _key(load_strategy_class(name)) == json.loads(KEYS.read_text())[name]


def test_width_relative_spec_values_and_filter():
    cls = _wr("NNNC_A_Raw_Open")
    inst = cls.__new__(cls)
    spec = inst.POOLED_LABEL
    assert spec["target"] == cls.MIN_BUY_GAIN_THRESHOLD == 0.02
    assert spec["exit_loss"] == cls.MIN_SELL_LOSS_THRESHOLD == 0.02
    assert spec["horizon"] == cls.HORIZON == 48
    assert (spec["stop"], spec["exit_take_profit"]) == (-0.05, 0.022)
    assert (spec["buckets"], spec["share"], spec["reference"]) == (10, 0.5, "bb_width")
    assert cls.TRAIN_ROW_FILTER == (("%wr_keep", "gt", 0.5),)
    assert cls.HELD_PAIR_ADD is None
    from Framework.PooledLabel import validate

    validate(spec)


# --- WidthSizerMixin ------------------------------------------------------------------------


class _Base:
    """Stands in for WidthBookMixin + the model: a fixed mask, family tags, predict_buy."""

    def get_entry_conditions(self, df):
        m = df["m"].to_numpy(bool)
        self._st_tags = np.where(m, "reversion:model", "").astype(object)
        return pd.Series(m, index=df.index)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, min_stake,
                            max_stake, leverage, entry_tag, side, **kwargs):
        return min(proposed_stake, self.fillable)


def _sizer(fillable=1e9):
    from _width_sizer import WidthSizerMixin

    S = type("S", (WidthSizerMixin, _Base), {})
    s = S.__new__(S)
    s.fillable = fillable
    return s


def test_sizer_keeps_entries_and_tags_agreement():
    s = _sizer()
    df = pd.DataFrame({"m": [1, 1, 0, 1], "predict_buy": [1, 0, 1, 0]})
    mask = s.get_entry_conditions(df)
    assert mask.tolist() == [True, True, False, True]
    assert s._st_tags.tolist() == ["reversion:model:agree", "reversion:model:disagree", "",
                                   "reversion:model:disagree"]


def test_sizer_stakes_are_capital_neutral_and_capped():
    s = _sizer()
    kw = dict(pair="X/USDT", current_time=None, current_rate=1.0, min_stake=10.0, leverage=1.0,
              side="long")
    a = s.custom_stake_amount(proposed_stake=900, max_stake=1e6, entry_tag="reversion:model:agree", **kw)
    d = s.custom_stake_amount(proposed_stake=900, max_stake=1e6, entry_tag="reversion:model:disagree", **kw)
    assert a == pytest.approx(900 / 0.75) and d == pytest.approx(450 / 0.75)
    assert 0.5 * a + 0.5 * d == pytest.approx(900)
    assert s.custom_stake_amount(proposed_stake=900, max_stake=1000.0,
                                 entry_tag="reversion:model:agree", **kw) == 1000.0
    s2 = _sizer(fillable=700.0)
    assert s2.custom_stake_amount(proposed_stake=900, max_stake=1e6,
                                  entry_tag="reversion:model:agree", **kw) == 700.0


def test_sizer_threshold_has_four_decimals_and_base_keeps_two():
    from _width_sizer import WidthSizerMixin
    from Framework.BaseStrategy import BaseStrategy

    saved = (WidthSizerMixin.prediction_threshold.value, BaseStrategy.prediction_threshold.value)
    try:  # class-level parameter objects: restore them for the rest of the session
        WidthSizerMixin.prediction_threshold.value = 0.53424
        assert WidthSizerMixin.prediction_threshold.value == 0.5342
        BaseStrategy.prediction_threshold.value = 0.53424
        assert BaseStrategy.prediction_threshold.value == 0.53
        assert WidthSizerMixin.prediction_threshold.value == 0.5342  # not shared
        assert WidthSizerMixin.prediction_threshold is not BaseStrategy.prediction_threshold
    finally:
        WidthSizerMixin.prediction_threshold.value, BaseStrategy.prediction_threshold.value = saved


def test_sizer_names_leave_the_model_key_alone():
    from _width_sizer import WidthSizerMixin

    wr = _wr("NNNC_A_Raw_Open")
    sz = type("SZ", (WidthSizerMixin, wr), {})
    sz2 = type("SZ2", (WidthSizerMixin, wr), {"AGREE_SHARE": 0.3, "STAKE_WEIGHTS": (1.0, 0.2)})
    assert _key(sz) == _key(wr) == _key(sz2)
