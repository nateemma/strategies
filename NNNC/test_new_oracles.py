"""Tests for the new oracle-gate screening arms (regime label study,
2026-10-07): one subclass per label worth training-free screening.

Nothing here trains a model or reports P&L -- these are LOOKAHEAD CEILING
arms per regime/ORACLE_GATE.md and are never promotable. The tests only
check that each arm wires its TRAINING_TYPE/HORIZON/threshold correctly,
pins get_model_path to NNNC_Oracle (never trains its own weights -- see
AGENT_GUIDE's "silent failure modes"), and can produce buy/sell labels
without raising.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from Framework.TrainingSignals import LabelMethod

from NNNC_Oracle_GbbFwdMae_H48 import NNNC_Oracle_GbbFwdMae_H48
from NNNC_Oracle_FutureExpectancy import NNNC_Oracle_FutureExpectancy
from NNNC_Oracle_FutureSharpe import NNNC_Oracle_FutureSharpe
from NNNC_Oracle_FutureSortino import NNNC_Oracle_FutureSortino
from NNNC_Oracle_MfeMaeRatio import NNNC_Oracle_MfeMaeRatio
from NNNC_Oracle_RiskAdjFuture import NNNC_Oracle_RiskAdjFuture
from NNNC_Oracle_LocalMinFollowthrough import NNNC_Oracle_LocalMinFollowthrough
from NNNC_Oracle_TripleBarrier_H12 import NNNC_Oracle_TripleBarrier_H12


def _synthetic_df(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
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


def _make(cls):
    inst = cls(config={})
    inst.dp = MagicMock()
    inst.dp.runmode.value = "other"
    return inst


@pytest.fixture()
def df():
    return _synthetic_df()


CASES = [
    (NNNC_Oracle_GbbFwdMae_H48, LabelMethod.gbb_fwdmae, 48, 0.015, 0.015),
    (NNNC_Oracle_FutureExpectancy, LabelMethod.future_expectancy, 48, 0.02, 0.02),
    (NNNC_Oracle_FutureSharpe, LabelMethod.future_sharpe, 48, 0.02, 0.02),
    (NNNC_Oracle_FutureSortino, LabelMethod.future_sortino, 48, 0.02, 0.02),
    (NNNC_Oracle_MfeMaeRatio, LabelMethod.mfe_mae_ratio, 48, 0.02, 0.02),
    (NNNC_Oracle_RiskAdjFuture, LabelMethod.risk_adj_future, 48, 0.02, 0.02),
    (NNNC_Oracle_LocalMinFollowthrough, LabelMethod.local_min_followthrough, 48, 0.02, 0.02),
    (NNNC_Oracle_TripleBarrier_H12, LabelMethod.triple_barrier, 12, 0.015, 0.015),
]


class TestTrainingTypeAndHorizonWiring:
    @pytest.mark.parametrize("cls,method,horizon,min_gain,min_loss", CASES)
    def test_attrs(self, cls, method, horizon, min_gain, min_loss):
        assert cls.TRAINING_TYPE == method
        assert cls.HORIZON == horizon
        assert cls.MIN_BUY_GAIN_THRESHOLD == min_gain
        assert cls.MIN_SELL_LOSS_THRESHOLD == min_loss

    @pytest.mark.parametrize("cls,method,horizon,min_gain,min_loss", CASES)
    def test_pins_model_path_to_nnnc_mlx(self, cls, method, horizon, min_gain, min_loss):
        """Every arm must alias NNNC_MLX's weights, never train its own --
        the 're-parameterised subclass trains a fresh model' landmine
        documented in AGENT_GUIDE.md."""
        inst = _make(cls)
        assert inst.get_model_path().endswith("NNNC_MLX/NNNC_MLX.keras")


class TestLabelsComputeWithoutRaising:
    @pytest.mark.parametrize("cls,method,horizon,min_gain,min_loss", CASES)
    def test_buy_and_sell_signals(self, cls, method, horizon, min_gain, min_loss, df):
        inst = _make(cls)
        buy = inst.get_train_buy_signals(df)
        sell = inst.get_train_sell_signals(df)
        assert len(buy) == len(df)
        assert len(sell) == len(df)
        assert set(np.unique(buy.to_numpy())).issubset({0.0, 1.0})
        assert set(np.unique(sell.to_numpy())).issubset({0.0, 1.0})


class TestTripleBarrierPtSlPlumbing:
    """The one arm needing explicit plumbing: pt/sl can only be RAISED by
    min_gain/min_loss, never lowered, so reaching 1.5% needs LABEL_PARAMS
    (NNNC.LabelParamsMixin) AND a matching MIN_BUY_GAIN_THRESHOLD floor."""

    def test_label_params_set_pt_sl_directly(self):
        assert NNNC_Oracle_TripleBarrier_H12.LABEL_PARAMS == {"pt": 0.015, "sl": 0.015}

    def test_label_params_sell_mirrors_buy(self):
        assert NNNC_Oracle_TripleBarrier_H12.LABEL_PARAMS_SELL is None

    def test_floor_would_have_silently_overridden_pt_without_the_threshold_change(self):
        """Demonstrates WHY MIN_BUY_GAIN_THRESHOLD must also move: with the
        inherited default (0.02 > 0.015), the labeler's own
        max(pt, min_gain) floor would raise pt back above 1.5%."""
        from Framework.TrainingSignals import labels_triple_barrier

        close = [100, 100.5, 101.6, 101.0, 100.0, 100.0]
        df_local = pd.DataFrame({"close": close, "high": close, "low": close})
        # pt=0.015 with the OLD (un-lowered) min_gain=0.02 floors back to 0.02
        raised = labels_triple_barrier(df_local, horizon=5, pt=0.015, sl=0.015, min_gain=0.02)
        # pt=0.015 with the matching min_gain=0.015 (what this arm actually sets)
        # keeps the barrier at 0.015.
        kept = labels_triple_barrier(df_local, horizon=5, pt=0.015, sl=0.015, min_gain=0.015)
        # 101.6 is a +1.6% move: only visible as a buy once the barrier is
        # genuinely 1.5%, not re-floored to 2%.
        assert kept.iloc[0] == 1
        assert raised.iloc[0] == 0
