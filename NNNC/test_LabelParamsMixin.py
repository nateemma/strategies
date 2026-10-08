"""Tests for NNNC.LabelParamsMixin, covering the strict-key behaviour added
alongside the label-dispatcher kwarg filter fix (regime label study,
2026-10-07): LABEL_PARAMS/LABEL_PARAMS_SELL are keys an arm asks for BY NAME,
so an unsupported one must raise rather than vanish silently.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from Framework.TrainingSignals import LabelMethod
from LabelParamsMixin import LabelParamsMixin


class _Base:
    """Minimal stand-in for NNNC_MLX/NNNC_Oracle exposing only what
    LabelParamsMixin + BaseNNStrategy's shared label methods need."""

    HORIZON = 48
    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
    LABEL_RVOL_THRESHOLD = -1.0
    LABEL_FILTER_WINDOW = None
    TRAINING_TYPE = LabelMethod.gbb
    dbg_verbose = False

    def filter_peaks_by_future_performance(self, future_df, peaks, side, window):
        return peaks  # no-op passthrough: keep every candidate peak


def _synthetic_df(n: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "close": close,
            "high": close + rng.uniform(0, 0.5, n),
            "low": close - rng.uniform(0, 0.5, n),
            "open": close,
            "guard_metric": rng.uniform(-1, 1, n),
            "bb_width": rng.uniform(0.0, 0.08, n),
            "rvol": rng.uniform(0.0, 3.0, n),
        }
    )


@pytest.fixture()
def df():
    return _synthetic_df()


class TestDefaultPathUnaffected:
    def test_empty_label_params_is_a_no_op(self, df):
        class Arm(LabelParamsMixin, _Base):
            pass

        arm = Arm()
        buy = arm.get_train_buy_signals(df)
        assert len(buy) == len(df)

    def test_existing_style_guard_threshold_override_still_works(self, df):
        """Mirrors the pre-existing NNNC_Ora_G10 pattern: a supported key
        override must still work after the strict_keys change."""

        class Arm(LabelParamsMixin, _Base):
            LABEL_PARAMS = {"guard_threshold": 1.0}

        arm = Arm()
        buy = arm.get_train_buy_signals(df)
        assert len(buy) == len(df)


class TestStrictKeyRaisesOnUnsupportedLabelParam:
    def test_buy_side_raises(self, df):
        class Arm(LabelParamsMixin, _Base):
            TRAINING_TYPE = LabelMethod.triple_barrier
            LABEL_PARAMS = {"not_a_real_kwarg": 1.0}

        arm = Arm()
        with pytest.raises(ValueError, match="not_a_real_kwarg"):
            arm.get_train_buy_signals(df)

    def test_sell_side_raises_via_fallback(self, df):
        """risk_adj_future has no dedicated sell branch in
        TrainingSignals.get_train_sell_signals, so it falls back to the buy
        accessor -- which must still enforce strict_keys. (Methods WITH a
        dedicated sell branch, e.g. triple_barrier, filter to that branch's
        own hand-written `allowed` set, a pre-existing mechanism this change
        does not touch -- see the module docstring on
        TrainingSignals.get_train_sell_signals.)"""

        class Arm(LabelParamsMixin, _Base):
            TRAINING_TYPE = LabelMethod.risk_adj_future
            LABEL_PARAMS_SELL = {"not_a_real_kwarg": 1.0}

        arm = Arm()
        with pytest.raises(ValueError, match="not_a_real_kwarg"):
            arm.get_train_sell_signals(df)

    def test_pt_sl_on_triple_barrier_does_not_raise(self, df):
        class Arm(LabelParamsMixin, _Base):
            TRAINING_TYPE = LabelMethod.triple_barrier
            HORIZON = 12
            MIN_BUY_GAIN_THRESHOLD = 0.015
            MIN_SELL_LOSS_THRESHOLD = 0.015
            LABEL_PARAMS = {"pt": 0.015, "sl": 0.015}
            LABEL_PARAMS_SELL = None

        arm = Arm()
        buy = arm.get_train_buy_signals(df)
        sell = arm.get_train_sell_signals(df)
        assert len(buy) == len(df)
        assert len(sell) == len(df)
