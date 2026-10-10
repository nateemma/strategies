"""T026 — SimpleTriggerStrategy guarantees G1-G4, G6 and FR-005/FR-019 (023 contract C1)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from Framework.BaseStrategy import BaseStrategy
from research.outcomes import causality as C
from research.outcomes import triggers as T
from SimpleTrigger import presets as P
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy
from utils.StopInvariants import StopInvariantMixin


SRC = Path(__file__).resolve().parents[1] / "SimpleTriggerStrategy.py"


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.random(n) * 100,
            "guard_metric": rng.uniform(-1, 1, n),
            "bb_width": rng.random(n) * 0.1,
            "rvol": rng.random(n) * 3,
        }
    )


def _instance(cls):
    return object.__new__(cls)


def test_g1_no_model():
    cfg = SimpleTriggerStrategy.strategy_config
    assert cfg.needs_training is False and cfg.model_type.name == "NONE"


def test_g2_entry_is_the_registry_trigger(monkeypatch):
    monkeypatch.setattr(C, "check_runnable", lambda name, *a: (True, "PASS"))

    class _G(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"

    df = _frame()
    T.ensure_builtins()
    got = _instance(_G).get_entry_conditions(df)
    assert np.array_equal(got, T.evaluate("gbb", df).to_numpy())
    assert got.any()


@pytest.mark.parametrize(
    "name,runnable",
    [
        (None, None),
        ("no_such_trigger", (False, "absent")),
        ("random_matched", (False, "FAIL")),
        ("gbb", (False, "STALE")),
    ],
)
def test_g2_refuses_without_fallback(monkeypatch, capsys, name, runnable):
    if runnable is not None:
        monkeypatch.setattr(C, "check_runnable", lambda n, *a: runnable)

    class _R(SimpleTriggerStrategy):
        TRIGGER_NAME = name

    got = _instance(_R).get_entry_conditions(_frame())
    assert not got.any()
    assert "*** SimpleTrigger:" in capsys.readouterr().out


def test_real_verdict_file_admits_gbb_and_refuses_random_matched():
    assert C.check_runnable("gbb")[0] is True
    assert C.check_runnable("random_matched")[0] is False


def test_fr019_params_merge_and_fresh_dicts_are_refused():
    class _M(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"
        BUY_OVERRIDES = {"entry_adx_threshold": 20.0}

    assert set(BaseStrategy.buy_params) <= set(_M.buy_params)
    assert set(P.BUY_KEYS) <= set(_M.buy_params)
    assert set(BaseStrategy.sell_params) <= set(_M.sell_params)
    assert _M.buy_params["entry_adx_threshold"] == 20.0
    with pytest.raises(P.PresetError):

        class _Bad(SimpleTriggerStrategy):
            TRIGGER_NAME = "gbb"
            buy_params = {"entry_adx_threshold": 20.0}


def test_g6_no_pair_specific_code():
    src = SRC.read_text()
    assert not re.search(r"/USDT|/USD\b|pair\s*==", src)


@pytest.mark.parametrize("preset", list(P.PRESETS))
def test_fr005_no_regime_logic(preset):
    class _P(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"
        PRESET = preset

    assert _P.entry_bear_filter_enable is False and _P.entry_trend_filter_enable is False
    src = SRC.read_text()
    assert "market_regime" not in src and "import SimpleStrategies" not in src
    assert "from SimpleStrategies" not in src


def test_g4_stop_invariants_wrap_the_stop():
    mro = SimpleTriggerStrategy.__mro__
    assert mro.index(StopInvariantMixin) < mro.index(BaseStrategy)


def test_unset_attribute_cannot_be_inherited_non_none():
    class _L(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"
        PRESET = "LEGACY_TRIGGERONLY"  # sets use_exit_signal = True

    assert _L.use_exit_signal is True
    with pytest.raises(P.PresetError):

        class _Back(_L):
            PRESET = "ARM_D_OPEN"  # leaves use_exit_signal unset
