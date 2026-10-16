"""036 T025: CostBudgetStakeMixin against the real BaseStrategy stake/entry hooks."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[3]
for p in (STRAT, STRAT / "NNNC"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils.CostBudgetStakeMixin import CostBudgetStakeMixin  # noqa: E402


T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
# quote volume per bar = close*volume = 100*10 = 1000 -> rolling 96 bars = 96,000; last bar 1,000
DF = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC"),
                   "close": 100.0, "volume": 10.0})


class _DP:
    runmode = SimpleNamespace(value="backtest")

    def get_analyzed_dataframe(self, pair, timeframe):
        return DF, None


def _model(tmp_path, choice="participation"):
    m = {"choice": choice, "tiers": {"A/USDT": 1}, "excluded_pairs": ["X/USDT"],
         "curves": {"1|buy": {"p": [0.0, 0.01, 0.02], "cost_bps": [2.0, 12.0, 22.0]},
                    "1|sell": {"p": [0.0, 0.02], "cost_bps": [2.0, 22.0]}},
         "per_pair": {"A/USDT|buy": {"p": [100, 1000], "cost_bps": [2.0, 20.0]}}}
    f = tmp_path / "cost_model.json"
    f.write_text(json.dumps(m))
    return str(f)


def _strategy(budget=None, model=None, base=BaseStrategy):
    cls = type("S", (CostBudgetStakeMixin, base),
               {"STAKE_COST_BUDGET_BPS": budget, "STAKE_COST_MODEL": model, "_cb_model": None})
    s = object.__new__(cls)
    s.dp, s.timeframe = _DP(), "15m"
    return s


def _stake(s, pair="A/USDT", proposed=900.0):
    return s.custom_stake_amount(pair, T0, 100.0, proposed, 10.0, 9000.0, 1.0, None, "long")


def _confirm(s, size, pair="A/USDT"):
    return s.confirm_trade_entry(pair, "limit", size / 100.0, 100.0, "GTC", T0, None, "long")


def test_off_is_the_parents_behaviour():
    s, b = _strategy(None), object.__new__(type("B", (BaseStrategy,), {}))
    b.dp, b.timeframe = _DP(), "15m"
    assert _stake(s) == _stake(b) == 100.0                 # 1/10 of the 1,000 candle
    for size in (50.0, 100.0, 150.0):
        assert _confirm(s, size) == _confirm(b, size)


def test_on_caps_at_the_budget_stake(tmp_path):
    s = _strategy(12.0, _model(tmp_path))                  # p_max 0.01 x 96,000 = 960
    assert _stake(s) == pytest.approx(900.0)               # proposed is below S_max
    assert _stake(s, proposed=2000.0) == pytest.approx(960.0)
    s7 = _strategy(7.0, _model(tmp_path))                  # p_max 0.005 -> 480
    assert _stake(s7) == pytest.approx(480.0)
    assert _confirm(s7, 470.0) is True and _confirm(s7, 500.0) is False


def test_min_quote_volume_floor_is_kept(tmp_path):
    s = _strategy(1000.0, _model(tmp_path))
    s.MIN_QUOTE_VOLUME = 5000                              # last candle has only 1,000
    assert _confirm(s, 10.0) is False


def test_excluded_pair_keeps_the_original_rule(tmp_path):
    s = _strategy(12.0, _model(tmp_path))
    s._cb()["tiers"]["X/USDT"] = 0
    assert _stake(s, "X/USDT") == pytest.approx(100.0)


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        _stake(_strategy(12.0, str(tmp_path / "nope.json")))
    s = _strategy(12.0, _model(tmp_path))
    with pytest.raises(ValueError, match="no cost tier"):
        _stake(s, "Z/USDT")
    bad = _strategy(12.0, _model(tmp_path))
    type(bad).STAKE_COST_MODEL_SHA = "0" * 16
    with pytest.raises(ValueError, match="sha"):
        _stake(bad)


def test_headroom_attribute_is_restored(tmp_path):
    s = _strategy(12.0, _model(tmp_path))
    _confirm(s, 100.0)
    assert "QUOTE_VOLUME_HEADROOM_MULT" not in s.__dict__
    assert s.QUOTE_VOLUME_HEADROOM_MULT == BaseStrategy.QUOTE_VOLUME_HEADROOM_MULT


def test_model_key_unchanged():
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cfg = load_config(STRAT / "config" / "config_wf.json")
    base = load_strategy_class("NNNC_A_Raw_Open")
    on = type("NNNC_CAP_T", (CostBudgetStakeMixin, base),
              {"STAKE_COST_BUDGET_BPS": 10.0, "STAKE_COST_MODEL": "/tmp/x.json"})
    assert MK.key_hash(MK.resolve(on, cfg)) == MK.key_hash(MK.resolve(base, cfg))
