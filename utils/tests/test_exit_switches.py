"""028 T016: the ATR-scaled take-profit switch (C2, adopted on 2022) in the shared exit.

`ExitParams.take_profit_atr_mult` (None = off) makes the take-profit k x the signal bar's
`atr_pct_roll`, clamped to [ATR_TAKE_PROFIT_MIN, ATR_TAKE_PROFIT_MAX]. Off, every path is
unchanged. On, it is resolved per trade AFTER family overrides, exactly as 027's exploratory
`SimpleTrigger_RevOnlyVolreg2_XC2` did, so the promoted class reproduces its trades.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from utils import ExitPolicy as EP  # noqa: E402
from utils.FamilyExits import FamilyExitMixin, FamilyExitError  # noqa: E402


K = 2.8970711921574543
OPEN = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_atr_take_profit_clamps():
    assert EP.atr_take_profit(K, 0.005) == pytest.approx(K * 0.005)
    assert EP.atr_take_profit(K, 0.0001) == EP.ATR_TAKE_PROFIT_MIN == 0.01
    assert EP.atr_take_profit(K, 0.5) == EP.ATR_TAKE_PROFIT_MAX == 0.05


def test_exit_params_default_is_off_and_validates():
    assert EP.ExitParams().take_profit_atr_mult is None
    assert EP.ExitParams(take_profit_atr_mult=K).take_profit_atr_mult == K
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            EP.ExitParams(take_profit_atr_mult=bad)


def test_research_replay_uses_the_same_clamp():
    from research.exit_coupling import paths as P

    for atr in (0.0001, 0.004, 0.5):
        assert P.c2_take_profit(K, atr) == EP.atr_take_profit(K, atr)


class _Trade:
    def __init__(self, tag="reversion:gbb", id_=7):
        self.pair = "SOL/USDT"
        self.open_date_utc = OPEN
        self.enter_tag = tag
        self.id = id_


def _frame(atr_signal):
    dates = [OPEN - timedelta(minutes=15 * i) for i in (2, 1, 0)]
    return pd.DataFrame({"date": dates, "atr_pct_roll": [0.9, atr_signal, 0.8],
                         "volume": [1.0, 1.0, 1.0]})


def _attrs(s, mult, frame):
    s.stoploss, s.timeframe = -0.05, "15m"
    s.atr_stoploss_multiplier, s.atr_stoploss_cap, s.atr_stoploss_floor = 2.5, -0.02, -0.04
    s.use_atr_adaptive_stoploss, s.use_volume_confirmation_stoploss = True, True
    s.take_profit_atr_mult = mult
    s.cexit_enable_profit_checks = types.SimpleNamespace(value=True)
    s.cexit_take_profit = types.SimpleNamespace(value=0.022)
    s.cexit_max_days = types.SimpleNamespace(value=10)
    s.dp = types.SimpleNamespace(get_analyzed_dataframe=lambda pair, tf: (frame, None))
    return s


def _stub(mult, frame):
    """Attributes `_exit_params` reads, with the BaseStrategy methods bound on."""
    s = _attrs(types.SimpleNamespace(), mult, frame)
    for name in ("_exit_params", "_apply_atr_take_profit"):
        setattr(s, name, types.MethodType(getattr(BaseStrategy, name), s))
    return s


def test_off_leaves_take_profit_unchanged():
    s = _stub(None, _frame(0.004))
    assert s._exit_params(_Trade()).take_profit == 0.022
    assert s._exit_params(_Trade()) == s._exit_params(None)


def test_on_uses_the_signal_bar_atr():
    s = _stub(K, _frame(0.004))
    assert s._exit_params(_Trade()).take_profit == pytest.approx(K * 0.004)


def test_on_without_a_trade_or_visible_signal_bar_keeps_the_preset():
    s = _stub(K, _frame(0.004))
    assert s._exit_params(None).take_profit == 0.022
    empty = _stub(K, _frame(0.004).iloc[[0, 2]])  # signal bar missing
    assert empty._exit_params(_Trade()).take_profit == 0.022


def test_value_is_cached_per_trade():
    frame = _frame(0.004)
    s = _stub(K, frame)
    first = s._exit_params(_Trade()).take_profit
    frame.loc[1, "atr_pct_roll"] = 0.009  # the analysed frame moves on; the trade's TP must not
    assert s._exit_params(_Trade()).take_profit == first


def test_family_exits_can_switch_it_on_for_one_family():
    class Fam(FamilyExitMixin, BaseStrategy):
        _FAMILY_VALIDATE_ON_SUBCLASS = False
        FAMILY_EXITS = {"reversion": {"take_profit_atr_mult": K}}

    s = _attrs(object.__new__(Fam), None, _frame(0.004))  # real MRO, no freqtrade __init__
    assert s._exit_params(_Trade("reversion:gbb")).take_profit == pytest.approx(K * 0.004)
    assert s._exit_params(_Trade("momentum:breakout_gbb", id_=8)).take_profit == 0.022


def test_strategy_level_and_family_level_both_set_is_refused():
    with pytest.raises(FamilyExitError):
        class Both(FamilyExitMixin, BaseStrategy):  # noqa: F841
            take_profit_atr_mult = K
            FAMILY_EXITS = {"reversion": {"take_profit_atr_mult": K}}
