# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore

"""
Tests for NNNC/_raw8_features.py, NNNC/NNNC_ForwardMAE_Raw.py and
NNNC/CreateScalersRaw8.py.

Covers:
  - add_raw8_features() matches research/clfstudy/data.py::raw8 column-by-column
  - add_raw8_features() is causal (future rows cannot change a past row's value)
  - LABEL_MAX_DRAWDOWN=0.10 on the arm actually reaches labels_forward_return_mae_cap
  - the arm's seq_len / include_list / TRAINING_TYPE / HORIZON / main_scaler_name
  - the arm does NOT override get_model_path (must train its own weights)
  - CreateScalersRaw8 uses the same raw8 helper/column list and isolated scaler names
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

STRAT_DIR = str(Path(__file__).parent.parent)
NNNC_DIR = str(Path(__file__).parent)
for _p in (STRAT_DIR, NNNC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _raw8_features import add_raw8_features, RAW8_COLUMNS, RAW8_ROLL_WINDOW
from research.clfstudy.data import raw8 as study_raw8
from NNNC_ForwardMAE_Raw import NNNC_ForwardMAE_Raw
from NNNClassifierMLX import ClassifierTypeMLX
from CreateScalersRaw8 import CreateScalersRaw8
from Framework.TrainingSignals import LabelMethod
import Framework.TrainingSignals as TS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + rng.standard_normal(n) * 0.05,
            "high": close + np.abs(rng.standard_normal(n)) * 0.2,
            "low": close - np.abs(rng.standard_normal(n)) * 0.2,
            "close": close,
            "volume": rng.uniform(500, 1500, n),
        }
    )


@pytest.fixture()
def ohlcv_df():
    return _make_ohlcv(250, seed=42)


# ---------------------------------------------------------------------------
# add_raw8_features matches the study implementation
# ---------------------------------------------------------------------------

def test_raw8_matches_study_implementation(ohlcv_df):
    ours = add_raw8_features(ohlcv_df)
    theirs = study_raw8(ohlcv_df)

    for col in RAW8_COLUMNS:
        assert col in ours.columns
        np.testing.assert_allclose(
            ours[col].to_numpy(dtype=float),
            theirs[col].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"column {col!r} diverges from research/clfstudy/data.py::raw8",
        )


def test_raw8_roll_window_matches_study():
    assert RAW8_ROLL_WINDOW == 96


# ---------------------------------------------------------------------------
# Causality: a value at row t must be unchanged by mutating rows after t
# ---------------------------------------------------------------------------

def test_raw8_features_are_causal(ohlcv_df):
    t = 150
    mutated = ohlcv_df.copy()
    rng = np.random.default_rng(7)
    tail_len = len(mutated) - (t + 1)
    for col in ("open", "high", "low", "close", "volume"):
        mutated.loc[t + 1 :, col] = (
            mutated.loc[t + 1 :, col].to_numpy() + rng.standard_normal(tail_len) * 50 + 500
        )

    baseline = add_raw8_features(ohlcv_df)
    perturbed = add_raw8_features(mutated)

    for col in RAW8_COLUMNS:
        np.testing.assert_allclose(
            baseline[col].iloc[: t + 1].to_numpy(dtype=float),
            perturbed[col].iloc[: t + 1].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"column {col!r} is not causal: rows <= {t} changed "
            f"when rows > {t} were mutated",
        )

    # Sanity: the mutation actually did something downstream (logvol_z /
    # vol_ratio roll forward from row t+1 onward), otherwise this test would
    # pass vacuously.
    assert not np.allclose(
        baseline["vol_ratio"].iloc[t + 1 :].to_numpy(dtype=float),
        perturbed["vol_ratio"].iloc[t + 1 :].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# max_drawdown plumbing regression test
# ---------------------------------------------------------------------------

def test_label_max_drawdown_reaches_labeler(ohlcv_df):
    """NNNC_ForwardMAE_Raw.LABEL_MAX_DRAWDOWN = 0.10 must actually arrive at
    labels_forward_return_mae_cap's max_drawdown kwarg, not the labeler's own
    0.02 default -- the exact class of bug this repo has hit before (a
    labeler kwarg silently dropped)."""
    strat = NNNC_ForwardMAE_Raw(config={})
    strat.dp = MagicMock()
    strat.dp.runmode.value = "other"
    assert strat.LABEL_MAX_DRAWDOWN == 0.10

    captured = {}
    original = TS.labels_forward_return_mae_cap

    def spy(df, **kwargs):
        captured.update(kwargs)
        return original(df, **kwargs)

    # get_train_buy_signals() resolves the labeler via TS.METHODS.get(method),
    # so swapping the dict entry intercepts exactly the kwargs BaseNNStrategy
    # actually sent, with no other behaviour change.
    with patch.dict(TS.METHODS, {LabelMethod.forward_mae: spy}):
        strat.get_train_buy_signals(ohlcv_df)

    assert captured.get("max_drawdown") == 0.10


def test_default_strategy_never_receives_max_drawdown_key(ohlcv_df):
    """Byte-identical-default-path guard: a strategy that never sets
    LABEL_MAX_DRAWDOWN (every strategy except this arm) must not have
    "max_drawdown" injected into its labeler params."""
    from Framework.BaseNNStrategy import BaseNNStrategy

    class _PlainStrat(BaseNNStrategy):
        def get_classifier_type(self):
            return "LSTM"

        def get_classifier(self, classifier_type, pair, seq_len, num_features):
            raise NotImplementedError

    strat = _PlainStrat(config={})
    strat.dp = MagicMock()
    strat.dp.runmode.value = "other"
    assert strat.LABEL_MAX_DRAWDOWN is None

    captured_params = {}
    original_dispatch = TS.get_train_buy_signals

    def spy_dispatch(df, method, params=None):
        captured_params.update(params or {})
        return original_dispatch(df, method=method, params=params)

    with patch.object(TS, "get_train_buy_signals", side_effect=spy_dispatch):
        strat.get_train_buy_signals(ohlcv_df)

    assert "max_drawdown" not in captured_params


# ---------------------------------------------------------------------------
# Arm configuration
# ---------------------------------------------------------------------------

def test_arm_configuration():
    assert NNNC_ForwardMAE_Raw.classifier_type == ClassifierTypeMLX.LSTM_RAW
    assert NNNC_ForwardMAE_Raw.seq_len == 32
    assert NNNC_ForwardMAE_Raw.TRAINING_TYPE == LabelMethod.forward_mae
    assert NNNC_ForwardMAE_Raw.HORIZON == 12
    assert NNNC_ForwardMAE_Raw.MIN_BUY_GAIN_THRESHOLD == 0.015
    assert NNNC_ForwardMAE_Raw.MIN_SELL_LOSS_THRESHOLD == 0.015
    assert NNNC_ForwardMAE_Raw.LABEL_MAX_DRAWDOWN == 0.10
    assert NNNC_ForwardMAE_Raw.include_list == RAW8_COLUMNS
    assert NNNC_ForwardMAE_Raw.main_scaler_name == "main_scaler_fwdmae_raw8"
    # Isolation: must differ from the shared production scaler name.
    assert NNNC_ForwardMAE_Raw.main_scaler_name != "main_scaler"


def test_arm_does_not_override_get_model_path():
    """The arm legitimately needs its OWN weights (different label/features/
    seq_len from its parent), so get_model_path must NOT be overridden --
    it must resolve via BaseNNStrategy's class-name-keyed default so an
    empty saved_data/NNNC_ForwardMAE_Raw/ is visible before any backtest."""
    assert "get_model_path" not in NNNC_ForwardMAE_Raw.__dict__

    strat = NNNC_ForwardMAE_Raw(config={})
    model_path = strat.get_model_path()
    assert "NNNC_ForwardMAE_Raw" in model_path


# ---------------------------------------------------------------------------
# CreateScalersRaw8
# ---------------------------------------------------------------------------

def test_create_scalers_raw8_isolated_names():
    assert CreateScalersRaw8.include_list == RAW8_COLUMNS
    assert CreateScalersRaw8.main_scaler_name == "main_scaler_fwdmae_raw8"
    assert CreateScalersRaw8.main_scaler_name == NNNC_ForwardMAE_Raw.main_scaler_name
    assert CreateScalersRaw8.main_scaler_name != "main_scaler"
    assert CreateScalersRaw8.tensor_scaler_name != "main_tensor_scaler"


def test_create_scalers_raw8_uses_shared_feature_helper(ohlcv_df):
    scaler_builder_out = CreateScalersRaw8.add_additional_indicators(
        CreateScalersRaw8(config={}), ohlcv_df
    )
    strategy_out = add_raw8_features(ohlcv_df)
    for col in RAW8_COLUMNS:
        np.testing.assert_allclose(
            scaler_builder_out[col].to_numpy(dtype=float),
            strategy_out[col].to_numpy(dtype=float),
        )
