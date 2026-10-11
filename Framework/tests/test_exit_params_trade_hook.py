"""024 T002: `BaseStrategy._exit_params(trade=None)` is a NEUTRAL hook (contract C1, R1/R2).

The hook lets a strategy choose exit parameters per trade (024's trigger families). For every
strategy that does not override it, it must be invisible:
  (a) `_exit_params(trade)` returns exactly what `_exit_params()` always returned;
  (b) the stop path, now reading `use_atr_adaptive_stoploss` / `use_volume_confirmation_stoploss`
      / `stoploss` through the params object instead of the attributes, computes the same stop.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.fingerprint import load_strategy_class  # noqa: E402
from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils.ExitPolicy import atr_stop_fraction, rolling_rvol  # noqa: E402


NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Trade:
    def __init__(self, age_hours: float = 1.0, enter_tag: str | None = "model_entry "):
        self.open_rate = 100.0
        self.open_date_utc = NOW - timedelta(hours=age_hours)
        self.pair = "SOL/USDT"
        self.enter_tag = enter_tag
        self.id = 1


@pytest.mark.parametrize("name", ["SimpleTrigger_Gbb", "NNNC_A_Raw_Open", "GbbOpen"])
def test_exit_params_ignores_trade_by_default(name):
    cls = load_strategy_class(name)
    # Class attributes and Parameter defaults are all `_exit_params` reads; the class itself
    # stands in for an instance (never instantiate an NNNC class: it would load or train a model).
    assert cls._exit_params(cls, _Trade()) == cls._exit_params(cls)
    assert cls._exit_params(cls, None) == cls._exit_params(cls)


def _stub(adaptive: bool, volume_confirm: bool):
    """Only what the stop path touches; real BaseStrategy methods bound onto it."""
    s = types.SimpleNamespace()
    s.stoploss = -0.05
    s.use_atr_adaptive_stoploss = adaptive
    s.use_volume_confirmation_stoploss = volume_confirm
    s.atr_stoploss_multiplier = 2.5
    s.atr_stoploss_cap = -0.02
    s.atr_stoploss_floor = -0.12
    s.stoploss_grace_hours = 0.0
    s.timeframe = "15m"
    for attr, val in (
        ("cexit_enable_profit_checks", True),
        ("cexit_take_profit", 0.022),
        ("cexit_max_days", 10),
    ):
        setattr(s, attr, types.SimpleNamespace(value=val))
    vol = [100.0] * 30 + [400.0]  # last bar rvol > 1, so volume confirmation bites
    frame = pd.DataFrame({"atr_pct_roll": [0.01] * 31, "volume": vol})
    s.dp = types.SimpleNamespace(get_analyzed_dataframe=lambda pair, tf: (frame, None))
    for name in ("_exit_params", "_atr_bar_inputs", "_resolve_normal_stop", "custom_stoploss"):
        setattr(s, name, types.MethodType(getattr(BaseStrategy, name), s))
    return s, frame


@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("volume_confirm", [True, False])
def test_stop_path_unchanged_when_params_mirror_attributes(adaptive, volume_confirm):
    s, frame = _stub(adaptive, volume_confirm)
    params = s._exit_params()
    rvol = rolling_rvol(frame["volume"]) if volume_confirm else 0.0
    expected_normal = atr_stop_fraction(0.01, rvol, params) if adaptive else s.stoploss

    assert s._resolve_normal_stop("SOL/USDT", _Trade()) == pytest.approx(expected_normal)

    after_fill = s.custom_stoploss("SOL/USDT", _Trade(), NOW, 100.0, 0.0, after_fill=True)
    assert after_fill == (pytest.approx(expected_normal) if adaptive else 1.0)
