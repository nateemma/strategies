"""TP2 arm: the take-profit resolves to 2 x the label's ATR(14)/close of the signal bar, unclamped."""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]


def _strategy(df):
    from _atr14_take_profit import Atr14TakeProfitMixin

    class S(Atr14TakeProfitMixin):
        timeframe = "15m"

        def trade_clock_start(self, trade):
            return trade.open_date_utc

    s = S()
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, tf: (df, None))
    return s


def test_take_profit_is_k_times_label_atr_of_signal_bar_unclamped():
    from Framework import TrainingSignals
    from utils.ExitPolicy import ExitParams

    rng = np.random.default_rng(0)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"date": dates, "open": close, "high": close * 1.0005, "low": close * 0.9995,
                       "close": close})
    s = _strategy(df)
    trade = SimpleNamespace(id=1, pair="X/USDT", open_date_utc=dates[300])
    p = s._apply_atr_take_profit(ExitParams(take_profit_atr_mult=2.0), trade)
    want = 2.0 * TrainingSignals._atr(df.iloc[:300])[-1] / close[299]
    assert abs(p.take_profit - want) < 1e-12
    assert p.take_profit < 0.01  # below C2's 1% floor: no clamp


def test_off_or_missing_signal_bar_leaves_params():
    from utils.ExitPolicy import ExitParams

    dates = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({"date": dates, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    s = _strategy(df)
    base = ExitParams()
    assert s._apply_atr_take_profit(base, SimpleNamespace(id=1, pair="X", open_date_utc=dates[2])) == base
    on = ExitParams(take_profit_atr_mult=2.0)
    late = SimpleNamespace(id=2, pair="X", open_date_utc=dates[-1] + timedelta(hours=1))
    assert s._apply_atr_take_profit(on, late) == on


def test_arm_keeps_width_book_and_sets_reversion_k():
    import NNNC_W_ARO as W
    import NNNC_W_ARO_TP2 as T

    assert issubclass(T.NNNC_W_ARO_TP2, W.NNNC_W_ARO)
    assert T.NNNC_W_ARO_TP2.FAMILY_EXIT_OVERRIDES == {"reversion": {"take_profit_atr_mult": 2.0}}
