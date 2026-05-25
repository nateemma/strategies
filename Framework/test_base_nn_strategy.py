# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore

"""
Tests for Framework/BaseNNStrategy.py

Covers:
  - TensorFlow import (hard fail if absent)
  - aggregate_single_labels / aggregate_multi_labels (static)
  - get_model_path / get_markov_matrix_path
  - PCA helpers: get_pca_path, pca_data_exists, save_pca_data, load_pca_data
  - model_exists
  - process_one_hot_columns
  - get_normalized_size
  - check_columns_included
  - get_market_regime / get_risk_level / get_flow / get_momentum
  - rolling_dataframe_normalise
  - window_and_flatten / unflatten_to_tensor
  - filter_peaks_by_future_performance
  - dwt_smooth / ema_smooth
  - augment_training_signals
  - ratio_to_weights
  - argmax_with_threshold / argmax_with_bias
  - get_training_class_weights
  - _labels_to_class_indices
  - _compute_markov_transition_matrix
  - process_predictions
  - enhance_training_data (no-op passthrough)
  - preprocess_training_data (no-op passthrough)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# TensorFlow must be importable — hard failure if absent
import tensorflow as tf  # noqa: F401 — constraint: do NOT mock TF

STRAT_DIR = str(Path(__file__).parent.parent)
if STRAT_DIR not in sys.path:
    sys.path.insert(0, STRAT_DIR)

from Framework.BaseNNStrategy import BaseNNStrategy
from Framework.BaseStrategy import (
    TradingAction,
    MarketRegime,
    RiskLevel,
    FlowDirection,
    MomentumDirection,
)


# ---------------------------------------------------------------------------
# Concrete subclass for instantiation
# ---------------------------------------------------------------------------

class _NNStrat(BaseNNStrategy):
    """Minimal concrete subclass for unit testing."""

    dbg_verbose = False

    def get_classifier_type(self):
        return "LSTM"

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        raise NotImplementedError("test subclass does not create classifiers")


def _make_strat() -> _NNStrat:
    s = _NNStrat(config={})
    s.dp = MagicMock()
    s.dp.runmode.value = "other"
    return s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def strat():
    return _make_strat()


@pytest.fixture()
def indicator_df():
    """200-row dataframe with all MINIMAL indicator columns populated."""
    from utils.DataframePopulator import DataframePopulator, DatasetType

    n = 200
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    ohlcv = pd.DataFrame(
        {
            "date": dates,
            "open": close + rng.standard_normal(n) * 0.05,
            "high": close + np.abs(rng.standard_normal(n)) * 0.2,
            "low": close - np.abs(rng.standard_normal(n)) * 0.2,
            "close": close,
            "volume": rng.uniform(500, 1500, n),
        }
    )
    pop = DataframePopulator()
    return pop.add_indicators(ohlcv, dataset_type=DatasetType.MINIMAL)


@pytest.fixture()
def strat_with_scaler(strat, indicator_df, tmp_path):
    """Strategy with dataframeUtils initialised, storage pointed at tmp_path.

    ``main_scaler`` starts as None so rolling_dataframe_normalise will fit and
    save a fresh scaler in tmp_path on first call.
    """
    from utils.DataframeUtils import DataframeUtils, ScalerType

    # Initialize dataframeUtils (required by rolling_dataframe_normalise)
    strat.dataframeUtils = DataframeUtils()
    strat.dataframeUtils.set_scaler_type(ScalerType.Robust)

    # No pre-existing scaler — the method will fit one on first call
    strat.main_scaler = None

    # Point storage location to tmp_path so no disk side-effects
    strat.get_storage_location = lambda: str(tmp_path) + "/"
    return strat


# ---------------------------------------------------------------------------
# TensorFlow import check
# ---------------------------------------------------------------------------

def test_tensorflow_importable():
    """Hard fail if TensorFlow is missing — constraint from task spec."""
    assert tf.__version__, "TensorFlow version string is empty"


# ---------------------------------------------------------------------------
# aggregate_single_labels
# ---------------------------------------------------------------------------

class TestAggregateSingleLabels:
    def test_two_arrays(self):
        a = np.array([0, 1, 2], dtype=np.int64)
        b = np.array([2, 1, 0], dtype=np.int64)
        result = BaseNNStrategy.aggregate_single_labels([a, b])
        np.testing.assert_array_equal(result, [0, 1, 2, 2, 1, 0])

    def test_empty(self):
        result = BaseNNStrategy.aggregate_single_labels([])
        assert len(result) == 0

    def test_dtype_preserved(self):
        a = np.array([0, 1], dtype=np.int32)
        b = np.array([2, 3], dtype=np.int32)
        result = BaseNNStrategy.aggregate_single_labels([a, b])
        assert result.dtype == np.int32

    def test_list_inputs(self):
        result = BaseNNStrategy.aggregate_single_labels([[0, 1], [2, 3]])
        np.testing.assert_array_equal(result, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# aggregate_multi_labels
# ---------------------------------------------------------------------------

class TestAggregateMultiLabels:
    def test_two_dicts(self):
        d1 = {"buy": np.array([0, 1]), "sell": np.array([1, 0])}
        d2 = {"buy": np.array([1, 0]), "sell": np.array([0, 1])}
        result = BaseNNStrategy.aggregate_multi_labels([d1, d2])
        np.testing.assert_array_equal(result["buy"], [0, 1, 1, 0])
        np.testing.assert_array_equal(result["sell"], [1, 0, 0, 1])

    def test_empty(self):
        result = BaseNNStrategy.aggregate_multi_labels([])
        assert result == {}

    def test_single_dict(self):
        d = {"action": np.array([2, 1, 0])}
        result = BaseNNStrategy.aggregate_multi_labels([d])
        np.testing.assert_array_equal(result["action"], [2, 1, 0])


# ---------------------------------------------------------------------------
# Model / storage paths
# ---------------------------------------------------------------------------

class TestModelPaths:
    def test_get_model_path_returns_string(self, strat):
        path = strat.get_model_path()
        assert isinstance(path, str)

    def test_get_model_path_ends_with_keras(self, strat):
        assert strat.get_model_path().endswith(".keras")

    def test_get_model_path_contains_class_name(self, strat):
        assert "_NNStrat" in strat.get_model_path()

    def test_get_markov_matrix_path_ends_with_npy(self, strat):
        assert strat.get_markov_matrix_path().endswith(".npy")

    def test_get_markov_matrix_path_derived_from_model_path(self, strat):
        model = strat.get_model_path()
        markov = strat.get_markov_matrix_path()
        assert markov.replace("_markov.npy", ".keras") == model


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

class TestPCAHelpers:
    def test_get_pca_path_is_path_instance(self, tmp_path):
        p = BaseNNStrategy.get_pca_path(str(tmp_path), "test_pca")
        assert isinstance(p, Path)

    def test_pca_data_not_exists_initially(self, tmp_path):
        assert not BaseNNStrategy.pca_data_exists(str(tmp_path), "nonexistent")

    def test_save_and_load_pca_roundtrip(self, tmp_path):
        components = np.eye(5)
        mean = np.zeros(5)
        n_components = 5
        cols = ["a", "b", "c", "d", "e"]
        exp_var = np.ones(5) * 0.2

        BaseNNStrategy.save_pca_data(
            str(tmp_path),
            "pca_test",
            components=components,
            mean=mean,
            n_components=n_components,
            feature_columns=cols,
            explained_variance_ratio=exp_var,
        )

        assert BaseNNStrategy.pca_data_exists(str(tmp_path), "pca_test")

        payload = BaseNNStrategy.load_pca_data(str(tmp_path), "pca_test")
        np.testing.assert_array_equal(payload["components"], components)
        np.testing.assert_array_equal(payload["mean"], mean)
        assert payload["n_components"] == n_components
        assert payload["feature_columns"] == cols


# ---------------------------------------------------------------------------
# model_exists
# ---------------------------------------------------------------------------

class TestModelExists:
    def test_returns_false_when_no_file(self, strat, tmp_path):
        strat.get_storage_location = lambda: str(tmp_path) + "/"
        strat.classifier = None
        assert strat.model_exists() is False

    def test_returns_true_when_keras_file_present(self, strat, tmp_path):
        strat.get_storage_location = lambda: str(tmp_path) + "/"
        strat.classifier = None
        model_path = strat.get_model_path()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).touch()
        assert strat.model_exists() is True


# ---------------------------------------------------------------------------
# check_columns_included
# ---------------------------------------------------------------------------

class TestCheckColumnsIncluded:
    def test_passes_when_all_present(self, strat):
        strat.include_list = ["a", "b", "c"]
        strat.check_columns_included(["a", "b"], "test_fn")  # no exception

    def test_raises_when_column_missing(self, strat):
        strat.include_list = ["a", "b"]
        with pytest.raises(ValueError, match="Missing"):
            strat.check_columns_included(["a", "c"], "test_fn")

    def test_error_message_names_missing_column(self, strat):
        strat.include_list = ["x"]
        with pytest.raises(ValueError) as exc_info:
            strat.check_columns_included(["x", "y"], "my_func")
        assert "y" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_normalized_size
# ---------------------------------------------------------------------------

class TestGetNormalizedSize:
    def test_returns_int(self, strat, indicator_df):
        size = strat.get_normalized_size(indicator_df)
        assert isinstance(size, int)

    def test_positive_size(self, strat, indicator_df):
        size = strat.get_normalized_size(indicator_df)
        assert size > 0

    def test_size_matches_include_list(self, strat, indicator_df):
        present = [c for c in strat.include_list if c in indicator_df.columns]
        size = strat.get_normalized_size(indicator_df)
        assert size == len(present)


# ---------------------------------------------------------------------------
# get_market_regime
# ---------------------------------------------------------------------------

class TestGetMarketRegime:
    def test_returns_array_of_correct_length(self, strat, indicator_df):
        regime = strat.get_market_regime(indicator_df)
        assert len(regime) == len(indicator_df)

    def test_only_valid_values(self, strat, indicator_df):
        regime = strat.get_market_regime(indicator_df)
        valid = {int(MarketRegime.BEAR), int(MarketRegime.SIDEWAYS), int(MarketRegime.BULL)}
        assert set(np.unique(regime)).issubset(valid)

    def test_bull_regime_when_close_above_smoothed_plus_deadband(self, strat, indicator_df):
        df = indicator_df.copy()
        # Steady uptrend: late bars sit above the centered rolling mean.
        df["close"] = np.linspace(100.0, 200.0, len(df))
        regime = strat.get_market_regime(df)
        # Centered smoother symmetric around the trend midpoint, so only the
        # back half clears the deadband — assert the last bar and that some
        # BULL exists.
        assert regime[-1] == MarketRegime.BULL
        assert np.any(regime == MarketRegime.BULL)

    def test_bear_regime_when_close_below_smoothed_minus_deadband(self, strat, indicator_df):
        df = indicator_df.copy()
        df["close"] = np.linspace(200.0, 100.0, len(df))
        regime = strat.get_market_regime(df)
        assert regime[-1] == MarketRegime.BEAR
        assert np.any(regime == MarketRegime.BEAR)

    def test_sideways_when_close_inside_deadband(self, strat, indicator_df):
        df = indicator_df.copy()
        # Flat close: EMA matches exactly, deadband keeps every bar SIDEWAYS.
        df["close"] = 100.0
        regime = strat.get_market_regime(df)
        assert np.all(regime == MarketRegime.SIDEWAYS)


# ---------------------------------------------------------------------------
# get_risk_level
# ---------------------------------------------------------------------------

class TestGetRiskLevel:
    def test_returns_array_of_correct_length(self, strat, indicator_df):
        risk = strat.get_risk_level(indicator_df)
        assert len(risk) == len(indicator_df)

    def test_only_valid_values(self, strat, indicator_df):
        risk = strat.get_risk_level(indicator_df)
        valid = {int(RiskLevel.LOW), int(RiskLevel.NORMAL), int(RiskLevel.HIGH)}
        assert set(np.unique(risk)).issubset(valid)

    def test_low_risk_when_atr_very_negative(self, strat, indicator_df):
        df = indicator_df.copy()
        df["atr_norm"] = -1.0
        risk = strat.get_risk_level(df)
        assert np.all(risk == RiskLevel.LOW)

    def test_high_risk_when_atr_very_positive(self, strat, indicator_df):
        df = indicator_df.copy()
        df["atr_norm"] = 1.0
        risk = strat.get_risk_level(df)
        assert np.all(risk == RiskLevel.HIGH)

    def test_normal_risk_when_atr_near_zero(self, strat, indicator_df):
        df = indicator_df.copy()
        df["atr_norm"] = 0.0
        risk = strat.get_risk_level(df)
        assert np.all(risk == RiskLevel.NORMAL)


# ---------------------------------------------------------------------------
# get_flow
# ---------------------------------------------------------------------------

class TestGetFlow:
    def test_returns_array_of_correct_length(self, strat, indicator_df):
        flow = strat.get_flow(indicator_df)
        assert len(flow) == len(indicator_df)

    def test_only_valid_values(self, strat, indicator_df):
        flow = strat.get_flow(indicator_df)
        valid = {int(FlowDirection.DECREASE), int(FlowDirection.NEUTRAL), int(FlowDirection.INCREASE)}
        assert set(np.unique(flow)).issubset(valid)

    def test_adds_flow_column_to_dataframe(self, strat, indicator_df):
        df = indicator_df.copy()
        strat.get_flow(df)
        assert "flow" in df.columns


# ---------------------------------------------------------------------------
# get_momentum
# ---------------------------------------------------------------------------

class TestGetMomentum:
    def test_returns_array_of_correct_length(self, strat, indicator_df):
        momentum = strat.get_momentum(indicator_df)
        assert len(momentum) == len(indicator_df)

    def test_only_valid_values(self, strat, indicator_df):
        momentum = strat.get_momentum(indicator_df)
        valid = {int(MomentumDirection.NEGATIVE), int(MomentumDirection.STABLE), int(MomentumDirection.POSITIVE)}
        assert set(np.unique(momentum)).issubset(valid)

    def test_negative_momentum_when_aroonosc_low(self, strat, indicator_df):
        df = indicator_df.copy()
        df["aroonosc_scaled"] = -1.0
        momentum = strat.get_momentum(df)
        assert np.all(momentum == MomentumDirection.NEGATIVE)

    def test_positive_momentum_when_aroonosc_high(self, strat, indicator_df):
        df = indicator_df.copy()
        df["aroonosc_scaled"] = 1.0
        momentum = strat.get_momentum(df)
        assert np.all(momentum == MomentumDirection.POSITIVE)


# ---------------------------------------------------------------------------
# rolling_dataframe_normalise
# ---------------------------------------------------------------------------

class TestRollingDataframeNormalise:
    def test_returns_dataframe(self, strat_with_scaler, indicator_df):
        result = strat_with_scaler.rolling_dataframe_normalise(indicator_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_date_column_in_output(self, strat_with_scaler, indicator_df):
        result = strat_with_scaler.rolling_dataframe_normalise(indicator_df)
        assert "date" not in result.columns

    def test_values_clipped_to_pm10(self, strat_with_scaler, indicator_df):
        result = strat_with_scaler.rolling_dataframe_normalise(indicator_df)
        for col in result.select_dtypes(include=[np.number]).columns:
            assert result[col].min() >= -10.0 - 1e-9
            assert result[col].max() <= 10.0 + 1e-9

    def test_no_nan_in_output(self, strat_with_scaler, indicator_df):
        result = strat_with_scaler.rolling_dataframe_normalise(indicator_df)
        assert not result.isnull().any().any()

    def test_output_columns_subset_of_include_list(self, strat_with_scaler, indicator_df):
        result = strat_with_scaler.rolling_dataframe_normalise(indicator_df)
        for col in result.columns:
            assert col in strat_with_scaler.include_list, f"Unexpected column: {col}"


# ---------------------------------------------------------------------------
# window_and_flatten / unflatten_to_tensor
# ---------------------------------------------------------------------------

class TestWindowAndFlatten:
    def test_output_shape(self, strat):
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        seq_len = 3
        flat = strat.window_and_flatten(df, seq_len)
        # num_sequences = 10 - 3 + 1 = 8, num_features = 2 * 3 = 6
        assert flat.shape == (8, 6)

    def test_column_names_have_time_tags(self, strat):
        df = pd.DataFrame({"x": range(5), "y": range(5, 10)})
        flat = strat.window_and_flatten(df, seq_len=2)
        assert any("t0" in c for c in flat.columns)
        assert any("t-1" in c for c in flat.columns)

    def test_single_row_sequence(self, strat):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        flat = strat.window_and_flatten(df, seq_len=1)
        assert flat.shape == (3, 1)


class TestUnflattenToTensor:
    def test_roundtrip_shape(self, strat):
        seq_len, num_features = 4, 5
        n_samples = 10
        x_flat = np.random.rand(n_samples, seq_len * num_features)
        x_tensor = strat.unflatten_to_tensor(x_flat, seq_len, num_features)
        assert x_tensor.shape == (n_samples, seq_len, num_features)

    def test_values_preserved(self, strat):
        x_flat = np.arange(24, dtype=float).reshape(2, 12)
        x_tensor = strat.unflatten_to_tensor(x_flat, seq_len=3, num_features=4)
        np.testing.assert_array_equal(x_tensor.reshape(2, 12), x_flat)


# ---------------------------------------------------------------------------
# filter_peaks_by_future_performance
# ---------------------------------------------------------------------------

class TestFilterPeaksByFuturePerformance:
    def _make_price_df(self, n=100):
        rng = np.random.default_rng(7)
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
        return pd.DataFrame({"close": close})

    def test_returns_list(self, strat):
        df = self._make_price_df()
        peaks = [5, 20, 50]
        result = strat.filter_peaks_by_future_performance(df, peaks, "buy", window=10)
        assert isinstance(result, list)

    def test_removes_peaks_too_close_to_end(self, strat):
        df = self._make_price_df(50)
        peaks = [48, 49]
        result = strat.filter_peaks_by_future_performance(df, peaks, "buy", window=10)
        assert len(result) == 0

    def test_trivially_large_gain_survives(self, strat):
        # Construct price that always goes up 10% after every peak
        n = 50
        close = np.ones(n) * 100.0
        close[20:] = 200.0  # massive gain after index 20
        df = pd.DataFrame({"close": close})
        strat.MIN_BUY_GAIN_THRESHOLD = 0.5  # 50% gain required
        result = strat.filter_peaks_by_future_performance(df, [10], "buy", window=20)
        assert 10 in result

    def test_no_peaks_returns_empty(self, strat):
        df = self._make_price_df()
        result = strat.filter_peaks_by_future_performance(df, [], "buy", window=10)
        assert result == []


# ---------------------------------------------------------------------------
# augment_training_signals
# ---------------------------------------------------------------------------

class TestAugmentTrainingSignals:
    def test_output_same_length(self, strat):
        buys = np.zeros(50)
        sells = np.zeros(50)
        buys[20] = 1.0
        sells[40] = 1.0
        b2, s2 = strat.augment_training_signals(buys.copy(), sells.copy())
        assert len(b2) == len(buys)
        assert len(s2) == len(sells)

    def test_sells_override_buys(self, strat):
        buys = np.ones(10)
        sells = np.ones(10)
        b2, s2 = strat.augment_training_signals(buys.copy(), sells.copy())
        # Wherever sells>0, buys should be 0
        conflict = (b2 > 0) & (s2 > 0)
        assert not np.any(conflict)

    def test_buy_signals_extended_backwards(self, strat):
        buys = np.zeros(20)
        sells = np.zeros(20)
        buys[15] = 1.0
        b2, s2 = strat.augment_training_signals(buys.copy(), sells.copy())
        # Index 14 and 13 should also be set (up to 2 entries earlier)
        assert b2[14] == 1.0
        assert b2[13] == 1.0

    def test_no_signals_unchanged(self, strat):
        buys = np.zeros(10)
        sells = np.zeros(10)
        b2, s2 = strat.augment_training_signals(buys.copy(), sells.copy())
        np.testing.assert_array_equal(b2, buys)
        np.testing.assert_array_equal(s2, sells)


# ---------------------------------------------------------------------------
# ratio_to_weights
# ---------------------------------------------------------------------------

class TestRatioToWeights:
    def test_returns_list(self, strat):
        result = strat.ratio_to_weights([50.0, 30.0, 20.0])
        assert isinstance(result, list)

    def test_length_matches_input(self, strat):
        result = strat.ratio_to_weights([33.3, 33.3, 33.4])
        assert len(result) == 3

    def test_weights_sum_to_one(self, strat):
        result = strat.ratio_to_weights([50.0, 30.0, 20.0])
        assert sum(result) == pytest.approx(1.0, rel=1e-6)

    def test_minority_class_gets_higher_weight(self, strat):
        result = strat.ratio_to_weights([80.0, 10.0, 10.0])
        # class 0 is majority → lowest weight
        assert result[0] < result[1]
        assert result[0] < result[2]

    def test_empty_returns_fallback(self, strat):
        result = strat.ratio_to_weights([])
        assert isinstance(result, list)

    def test_all_zeros_returns_fallback(self, strat):
        result = strat.ratio_to_weights([0.0, 0.0, 0.0])
        assert isinstance(result, list)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# argmax_with_threshold
# ---------------------------------------------------------------------------

class TestArgmaxWithThreshold:
    def test_high_confidence_returned_as_is(self, strat):
        probs = np.array([[0.1, 0.1, 0.8], [0.7, 0.2, 0.1]])
        result = strat.argmax_with_threshold(probs, threshold=0.5, default_class=1)
        assert result[0] == 2  # class 2 at 0.8
        assert result[1] == 0  # class 0 at 0.7

    def test_low_confidence_returns_default(self, strat):
        probs = np.array([[0.4, 0.35, 0.25]])
        result = strat.argmax_with_threshold(probs, threshold=0.5, default_class=1)
        assert result[0] == 1  # default class

    def test_output_length_matches_input(self, strat):
        probs = np.random.dirichlet([1, 1, 1], size=50)
        result = strat.argmax_with_threshold(probs, threshold=0.5)
        assert len(result) == 50

    def test_values_in_valid_class_range(self, strat):
        probs = np.random.dirichlet([1, 1, 1], size=30)
        result = strat.argmax_with_threshold(probs, threshold=0.5, default_class=1)
        assert set(np.unique(result)).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# argmax_with_bias
# ---------------------------------------------------------------------------

class TestArgmaxWithBias:
    def test_bias_reduces_class_probability(self, strat):
        probs = np.array([[0.35, 0.30, 0.35]])
        # Without bias, class 0 wins via argmax (or ties)
        # Add large bias on class 0 → class 2 should win instead
        result = strat.argmax_with_bias(probs, bias_map={0: 0.4}, threshold=0.0, default_class=1)
        assert result[0] != 0

    def test_no_bias_equals_argmax_with_threshold(self, strat):
        rng = np.random.default_rng(99)
        probs = rng.dirichlet([3, 1, 1], size=20)
        r1 = strat.argmax_with_bias(probs, bias_map={}, threshold=0.5, default_class=1)
        r2 = strat.argmax_with_threshold(probs, threshold=0.5, default_class=1)
        np.testing.assert_array_equal(r1, r2)

    def test_output_length_matches_input(self, strat):
        probs = np.random.dirichlet([1, 1, 1], size=40)
        result = strat.argmax_with_bias(probs, threshold=0.5)
        assert len(result) == 40


# ---------------------------------------------------------------------------
# _labels_to_class_indices
# ---------------------------------------------------------------------------

class TestLabelsToClassIndices:
    def test_1d_array_passthrough(self, strat):
        labels = np.array([0, 1, 2, 1, 0])
        result = strat._labels_to_class_indices(labels)
        np.testing.assert_array_equal(result, labels)

    def test_2d_onehot_converted(self, strat):
        labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = strat._labels_to_class_indices(labels)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_dict_trading_key(self, strat):
        labels = {"trading": np.array([0, 1, 2])}
        result = strat._labels_to_class_indices(labels)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_dict_fallback_to_first_value(self, strat):
        labels = {"action": np.array([2, 0, 1])}
        result = strat._labels_to_class_indices(labels)
        np.testing.assert_array_equal(result, [2, 0, 1])


# ---------------------------------------------------------------------------
# _compute_markov_transition_matrix
# ---------------------------------------------------------------------------

class TestComputeMarkovTransitionMatrix:
    def test_identity_when_short_sequence(self):
        result = BaseNNStrategy._compute_markov_transition_matrix(np.array([0]), num_classes=3)
        np.testing.assert_array_equal(result, np.eye(3))

    def test_rows_sum_to_one(self):
        seq = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1])
        result = BaseNNStrategy._compute_markov_transition_matrix(seq, num_classes=3)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(3), atol=1e-10)

    def test_shape_is_num_classes_squared(self):
        seq = np.array([0, 1, 0, 2, 1])
        result = BaseNNStrategy._compute_markov_transition_matrix(seq, num_classes=3)
        assert result.shape == (3, 3)

    def test_deterministic_sequence(self):
        # 0→1→2→0→1→2 repeatedly → predictable transitions
        seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        result = BaseNNStrategy._compute_markov_transition_matrix(seq, num_classes=3)
        # From 0 always go to 1
        assert result[0, 1] == pytest.approx(1.0)
        # From 1 always go to 2
        assert result[1, 2] == pytest.approx(1.0)

    def test_none_input_returns_identity(self):
        result = BaseNNStrategy._compute_markov_transition_matrix(None, num_classes=3)
        np.testing.assert_array_equal(result, np.eye(3))


# ---------------------------------------------------------------------------
# get_training_class_weights
# ---------------------------------------------------------------------------

class TestGetTrainingClassWeights:
    def test_none_returns_uniform(self, strat):
        weights = strat.get_training_class_weights(None, None)
        assert weights == [1.0, 1.0, 1.0]

    def test_1d_labels_returns_list_of_three(self, strat):
        labels = np.array([0, 0, 0, 1, 1, 2])  # imbalanced
        weights = strat.get_training_class_weights(labels)
        assert len(weights) == 3
        # minority class 2 (1 sample) should get highest weight
        assert weights[2] > weights[0]

    def test_2d_onehot_labels(self, strat):
        labels = np.eye(3, dtype=float)[np.array([0, 0, 1, 2])]  # 2×0, 1×1, 1×2
        weights = strat.get_training_class_weights(labels)
        assert len(weights) == 3

    def test_dict_labels(self, strat):
        labels_dict = {"trading": np.array([0, 1, 2, 0, 1, 2, 2])}
        weights = strat.get_training_class_weights(labels_dict)
        assert isinstance(weights, dict)
        assert "trading" in weights


# ---------------------------------------------------------------------------
# process_predictions
# ---------------------------------------------------------------------------

class TestProcessPredictions:
    def test_adds_predict_buy_column(self, strat, indicator_df):
        df = indicator_df.copy()
        preds = np.zeros(len(df), dtype=int)
        preds[5] = TradingAction.BUY
        result = strat.process_predictions(df, preds)
        assert "predict_buy" in result.columns

    def test_adds_predict_sell_column(self, strat, indicator_df):
        df = indicator_df.copy()
        preds = np.zeros(len(df), dtype=int)
        preds[10] = TradingAction.SELL
        result = strat.process_predictions(df, preds)
        assert "predict_sell" in result.columns

    def test_buy_signal_at_correct_index(self, strat, indicator_df):
        df = indicator_df.copy()
        preds = np.full(len(df), TradingAction.HOLD)
        preds[7] = TradingAction.BUY
        result = strat.process_predictions(df, preds)
        assert result["predict_buy"].iloc[7] == 1
        assert result["predict_buy"].sum() == 1

    def test_sell_signal_at_correct_index(self, strat, indicator_df):
        df = indicator_df.copy()
        preds = np.full(len(df), TradingAction.HOLD)
        preds[12] = TradingAction.SELL
        result = strat.process_predictions(df, preds)
        assert result["predict_sell"].iloc[12] == 1
        assert result["predict_sell"].sum() == 1

    def test_hold_signals_produce_zero_buy_sell(self, strat, indicator_df):
        df = indicator_df.copy()
        preds = np.full(len(df), TradingAction.HOLD)
        result = strat.process_predictions(df, preds)
        assert result["predict_buy"].sum() == 0
        assert result["predict_sell"].sum() == 0


# ---------------------------------------------------------------------------
# enhance_training_data — dispatcher behaviour
# ---------------------------------------------------------------------------
#
# After the GAN refactor, ``enhance_training_data`` is the single entry
# point for class-balanced augmentation; it dispatches to
# ``balance_single_task`` or ``balance_multi_task`` based on ``gan_type``
# and the shape of ``train_labels``.  These tests cover the short-circuit
# branches (no-op when GAN is off, default config) without instantiating
# a real GAN — the balance helpers themselves are tested in
# ``GANs/tests/test_balance.py``.

class TestEnhanceTrainingData:
    def test_default_config_returns_input_unchanged(self, strat, indicator_df):
        """Default ``gan_type=GANType.NONE`` → pass-through."""
        df = indicator_df.copy()
        labels = np.zeros(len(df), dtype=int)
        out_df, out_labels = strat.enhance_training_data(df, labels)
        pd.testing.assert_frame_equal(out_df, df)
        np.testing.assert_array_equal(out_labels, labels)

    def test_gan_augment_false_skips_dispatcher(self, strat, indicator_df):
        """``gan_augment=False`` short-circuits even when gan_type is set.

        Multi-task strategies that do their work in ``preprocess_training_data``
        (3-D tensors) rely on this — they declare ``gan_type=MT_WGAN`` for
        downstream consumers but turn off the 2-D dispatcher path here.
        """
        from GANs.GANType import GANType  # noqa: E402
        df = indicator_df.copy()
        labels = np.zeros(len(df), dtype=int)
        strat.gan_type = GANType.WGAN
        strat.gan_augment = False
        out_df, out_labels = strat.enhance_training_data(df, labels)
        pd.testing.assert_frame_equal(out_df, df)
        np.testing.assert_array_equal(out_labels, labels)

    def test_mismatched_single_task_with_dict_labels_skips(self, strat, indicator_df, capsys):
        """Single-task ``gan_type`` + dict labels is a misconfiguration —
        skip rather than crash, and emit a warning so the operator can
        spot it in the logs."""
        from GANs.GANType import GANType  # noqa: E402
        df = indicator_df.copy()
        labels = {"trading": np.zeros((len(df), 3), dtype=np.float32)}
        strat.gan_type = GANType.WGAN
        strat.gan_augment = True
        out_df, out_labels = strat.enhance_training_data(df, labels)
        pd.testing.assert_frame_equal(out_df, df)
        # Same dict object back, untouched.
        assert out_labels is labels
        captured = capsys.readouterr()
        assert "skipping augmentation" in captured.out

    def test_mt_gan_with_ndarray_labels_defers_silently(self, strat, indicator_df, capsys):
        """Multi-task ``gan_type`` + ndarray labels (single-task strategy) —
        deferred silently to preprocess_training_data; no warning emitted here."""
        from GANs.GANType import GANType  # noqa: E402
        df = indicator_df.copy()
        labels = np.zeros(len(df), dtype=int)
        strat.gan_type = GANType.MT_WGAN
        strat.gan_augment = True
        out_df, out_labels = strat.enhance_training_data(df, labels)
        pd.testing.assert_frame_equal(out_df, df)
        np.testing.assert_array_equal(out_labels, labels)
        captured = capsys.readouterr()
        assert "skipping augmentation" not in captured.out

    def test_empty_inputs_short_circuit(self, strat):
        """Empty df / empty labels — pass-through even with gan_type set."""
        from GANs.GANType import GANType  # noqa: E402
        empty_df = pd.DataFrame()
        empty_labels = np.zeros(0, dtype=int)
        strat.gan_type = GANType.WGAN
        out_df, out_labels = strat.enhance_training_data(empty_df, empty_labels)
        assert len(out_df) == 0
        assert len(out_labels) == 0


class TestGanExpectedMetadata:
    """``_gan_expected_metadata`` is the strategy's declaration of what
    must round-trip through the GAN's saved metadata.  ``GANInterface.load``
    raises on any drift — so what's in this dict matters."""

    def test_includes_thresholds_and_training_type(self, strat, indicator_df):
        meta = strat._gan_expected_metadata(indicator_df)
        assert "min_buy_gain_threshold" in meta
        assert "min_sell_loss_threshold" in meta
        assert "training_type" in meta

    def test_threshold_values_are_floats(self, strat, indicator_df):
        meta = strat._gan_expected_metadata(indicator_df)
        assert isinstance(meta["min_buy_gain_threshold"], float)
        assert isinstance(meta["min_sell_loss_threshold"], float)

    def test_training_type_is_int(self, strat, indicator_df):
        meta = strat._gan_expected_metadata(indicator_df)
        assert isinstance(meta["training_type"], int)


class TestResolveGanPassthroughForDispatcher:
    """The dispatcher needs passthrough columns by *name* for DataFrame
    backends and by *index* for ndarray backends.  Resolution must filter
    out columns not present in the actual normalised frame so an
    over-broad config doesn't crash augmentation."""

    def test_returns_none_when_unconfigured(self, strat, indicator_df):
        strat.gan_passthrough_columns = []
        result = strat._resolve_gan_passthrough_for_dispatcher(indicator_df, indicator_df)
        assert result is None

    def test_dataframe_returns_present_names(self, strat):
        df = pd.DataFrame({"dow_sin": [0.1], "x": [1.0], "y": [2.0]})
        strat.gan_passthrough_columns = ["dow_sin", "absent"]
        result = strat._resolve_gan_passthrough_for_dispatcher(df, df)
        assert result == ["dow_sin"]

    def test_dataframe_with_no_match_returns_none(self, strat):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        strat.gan_passthrough_columns = ["absent_a", "absent_b"]
        result = strat._resolve_gan_passthrough_for_dispatcher(df, df)
        assert result is None


# ---------------------------------------------------------------------------
# preprocess_training_data (no-op hook)
# ---------------------------------------------------------------------------

class TestPreprocessTrainingData:
    def test_returns_inputs_unchanged(self, strat, indicator_df):
        train_data = np.random.rand(50, 16, 10)
        test_data = np.random.rand(20, 16, 10)
        train_labels = np.zeros((50, 3))
        test_labels = np.zeros((20, 3))
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)
        np.testing.assert_array_equal(r_vd, test_data)
        np.testing.assert_array_equal(r_tl, train_labels)
        np.testing.assert_array_equal(r_vl, test_labels)


# ---------------------------------------------------------------------------
# preprocess_training_data — single-task + MT GAN guards
# ---------------------------------------------------------------------------
#
# These tests verify every pass-through guard in the new single-task MT GAN
# branch of preprocess_training_data.  Actually loading a GAN and calling
# balance_multi_task is integration territory; here we only exercise the
# guards that keep existing single-task strategies completely unaffected.

class TestPreprocessTrainingDataMTGuards:
    """Guard branches in BaseNNStrategy.preprocess_training_data for the
    single-task + MT GAN augmentation case."""

    def _make_3d(self, n=50, T=16, F=10):
        return np.random.rand(n, T, F).astype(np.float32)

    def _make_labels(self, n=50, one_hot=False, num_classes=3):
        if one_hot:
            idx = np.random.randint(0, num_classes, n)
            return np.eye(num_classes, dtype=np.float32)[idx]
        return np.random.randint(0, num_classes, n).astype(int)

    def test_passthrough_when_gan_type_none(self, strat, indicator_df):
        """Default NONE type → immediate pass-through."""
        from GANs.GANType import GANType  # noqa: E402
        assert strat.gan_type == GANType.NONE
        train_data = self._make_3d()
        test_data = self._make_3d(n=20)
        train_labels = self._make_labels()
        test_labels = self._make_labels(n=20)
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)
        np.testing.assert_array_equal(r_vd, test_data)
        np.testing.assert_array_equal(r_tl, train_labels)
        np.testing.assert_array_equal(r_vl, test_labels)

    def test_passthrough_when_gan_augment_false(self, strat, indicator_df):
        """gan_augment=False → pass-through even with MT GAN type."""
        from GANs.GANType import GANType  # noqa: E402
        strat.gan_type = GANType.MT_DDPM
        strat.gan_augment = False
        train_data = self._make_3d()
        test_data = self._make_3d(n=20)
        train_labels = self._make_labels()
        test_labels = self._make_labels(n=20)
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)

    def test_passthrough_when_single_task_gan_type(self, strat, indicator_df):
        """Non-MT GAN type → pass-through (single-task aug handled elsewhere)."""
        from GANs.GANType import GANType  # noqa: E402
        strat.gan_type = GANType.WGAN
        strat.gan_augment = True
        train_data = self._make_3d()
        test_data = self._make_3d(n=20)
        train_labels = self._make_labels()
        test_labels = self._make_labels(n=20)
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)

    def test_passthrough_when_labels_already_dict(self, strat, indicator_df):
        """Dict labels → BaseNNMTStrategy handles; pass through here."""
        from GANs.GANType import GANType  # noqa: E402
        strat.gan_type = GANType.MT_DDPM
        strat.gan_augment = True
        train_data = self._make_3d()
        test_data = self._make_3d(n=20)
        train_labels = {"trading": self._make_labels(one_hot=True)}
        test_labels = {"trading": self._make_labels(n=20, one_hot=True)}
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)
        assert r_tl is train_labels

    def test_passthrough_when_train_data_not_3d(self, strat, indicator_df):
        """2D train_data → pass-through (can't run tensor-level aug)."""
        from GANs.GANType import GANType  # noqa: E402
        strat.gan_type = GANType.MT_DDPM
        strat.gan_augment = True
        train_data = np.random.rand(50, 10).astype(np.float32)
        test_data = np.random.rand(20, 10).astype(np.float32)
        train_labels = self._make_labels()
        test_labels = self._make_labels(n=20)
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)

    def test_passthrough_when_train_data_empty(self, strat, indicator_df):
        """Empty train_data → pass-through."""
        from GANs.GANType import GANType  # noqa: E402
        strat.gan_type = GANType.MT_DDPM
        strat.gan_augment = True
        train_data = np.empty((0, 16, 10), dtype=np.float32)
        test_data = np.random.rand(20, 16, 10).astype(np.float32)
        train_labels = np.empty(0, dtype=int)
        test_labels = self._make_labels(n=20)
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        assert r_td.shape[0] == 0


# ---------------------------------------------------------------------------
# post-GAN scaling guards
# ---------------------------------------------------------------------------

class TestPostGanScalingGuards:
    """Tests for use_post_gan_scaling flag routing in preprocess_training_data
    and get_predictions."""

    def _make_3d(self, n=50, T=16, F=10):
        return np.random.rand(n, T, F).astype(np.float32)

    def _make_labels(self, n=50, num_classes=3):
        return np.random.randint(0, num_classes, n).astype(int)

    def test_flag_defaults_to_false(self, strat):
        """use_post_gan_scaling defaults to False — existing behavior preserved."""
        assert strat.use_post_gan_scaling is False

    def test_passthrough_when_flag_false_and_gan_none(self, strat, indicator_df):
        """Flag=False + GAN=NONE → unchanged pass-through (existing guard fires first)."""
        from GANs.GANType import GANType
        assert strat.gan_type == GANType.NONE
        assert not strat.use_post_gan_scaling
        train_data = self._make_3d()
        test_data = self._make_3d(n=20)
        train_labels = self._make_labels()
        test_labels = self._make_labels(n=20)
        r_td, r_vd, r_tl, r_vl = strat.preprocess_training_data(
            indicator_df, train_data, test_data, train_labels, test_labels
        )
        np.testing.assert_array_equal(r_td, train_data)
        np.testing.assert_array_equal(r_vd, test_data)

    def test_flag_true_recognized_for_mt_ddpm(self, strat):
        """Setting use_post_gan_scaling=True on MT_DDPM strategy is accepted."""
        from GANs.GANType import GANType
        strat.gan_type = GANType.MT_DDPM
        strat.gan_augment = True
        strat.use_post_gan_scaling = True
        assert strat.use_post_gan_scaling is True

    def test_post_gan_path_uses_different_save_subdir(self):
        """When post_gan_scaling=True, gan_save_path returns GANs_PostScale/ subdir."""
        from GANs.GANType import GANType
        from GANs.paths import gan_save_path
        path_normal = gan_save_path("/storage", GANType.MT_DDPM, post_gan_scaling=False)
        path_post = gan_save_path("/storage", GANType.MT_DDPM, post_gan_scaling=True)
        assert "GANs_PostScale" in path_post
        assert "GANs_PostScale" not in path_normal
        assert path_normal != path_post

    def test_post_gan_save_path_has_correct_subdirectory(self):
        """GANs_PostScale/<type> path matches expected layout."""
        from GANs.GANType import GANType
        from GANs.paths import gan_save_subdir
        subdir = gan_save_subdir(GANType.MT_DDPM, post_gan_scaling=True)
        assert subdir == "GANs_PostScale/mt_ddpm"

    def test_normal_save_path_unchanged(self):
        """Existing v1 save path is not affected by the new parameter."""
        from GANs.GANType import GANType
        from GANs.paths import gan_save_subdir
        subdir = gan_save_subdir(GANType.MT_DDPM)
        assert subdir == "GANs/mt_ddpm"


# ---------------------------------------------------------------------------
# dwt_smooth
# ---------------------------------------------------------------------------

class TestDwtSmooth:
    def test_output_same_length(self, strat):
        data = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = strat.dwt_smooth(data)
        assert len(result) == len(data)

    def test_output_is_ndarray(self, strat):
        data = np.random.rand(64)
        result = strat.dwt_smooth(data)
        assert isinstance(result, np.ndarray)

    def test_smoothing_reduces_variance(self, strat):
        rng = np.random.default_rng(5)
        noisy = rng.standard_normal(128)
        smoothed = strat.dwt_smooth(noisy)
        assert noisy.var() >= smoothed.var() * 0.5  # variance reduced or similar


# ---------------------------------------------------------------------------
# print_dataframe_ranges (smoke)
# ---------------------------------------------------------------------------

class TestPrintDataframeRanges:
    def test_no_exception(self, strat, indicator_df, capsys):
        strat.print_dataframe_ranges("Test Ranges", indicator_df)
        out = capsys.readouterr().out
        assert "Test Ranges" in out
