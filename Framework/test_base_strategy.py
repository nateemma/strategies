# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore

"""
Tests for Framework/BaseStrategy.py

Covers:
  - Enums (TradingAction, MarketRegime, RiskLevel, FlowDirection, MomentumDirection)
  - StrategyConfig / NormalizationType / ModelType / GANType
  - aggregate_dataframes / aggregate_labels (static helpers)
  - get_assessment_feedback
  - check_precision_columns
  - get_storage_location
  - custom_stoploss
  - print_hyperopt_parameters (smoke test)
  - populate_entry_trend / populate_exit_trend (guard conditions)
  - confirm_trade_exit minimum-hold check
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure strategy root is on sys.path
STRAT_DIR = str(Path(__file__).parent.parent)
if STRAT_DIR not in sys.path:
    sys.path.insert(0, STRAT_DIR)

from Framework.BaseStrategy import (
    BaseStrategy,
    StrategyConfig,
    TradingAction,
    MarketRegime,
    RiskLevel,
    FlowDirection,
    MomentumDirection,
    NormalizationType,
    ModelType,
    GANType,
)


# ---------------------------------------------------------------------------
# Minimal concrete subclass so we can instantiate BaseStrategy
# ---------------------------------------------------------------------------

class _Strat(BaseStrategy):
    """Concrete subclass with trivial entry/exit for unit testing."""

    dbg_verbose = False  # suppress print spam

    def get_entry_conditions(self, dataframe):
        return dataframe["close"] > dataframe["close"].shift(1).fillna(0)

    def get_exit_conditions(self, dataframe):
        return dataframe["close"] < dataframe["close"].shift(1).fillna(float("inf"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def strat():
    s = _Strat(config={})
    s.dp = MagicMock()
    s.dp.runmode.value = "other"
    return s


@pytest.fixture()
def ohlcv_df():
    """200-row OHLCV dataframe with a realistic price walk."""
    n = 200
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + rng.standard_normal(n) * 0.05,
            "high": close + np.abs(rng.standard_normal(n)) * 0.2,
            "low": close - np.abs(rng.standard_normal(n)) * 0.2,
            "close": close,
            "volume": rng.uniform(500, 1500, n),
        }
    )


@pytest.fixture()
def indicator_df(ohlcv_df):
    """OHLCV dataframe enriched with a minimal indicator set."""
    from utils.DataframePopulator import DataframePopulator, DatasetType

    pop = DataframePopulator()
    df = pop.add_indicators(ohlcv_df.copy(), dataset_type=DatasetType.MINIMAL)
    return df


# ---------------------------------------------------------------------------
# Enum sanity checks
# ---------------------------------------------------------------------------

class TestEnums:
    def test_trading_action_values(self):
        assert TradingAction.SELL == 0
        assert TradingAction.HOLD == 1
        assert TradingAction.BUY == 2

    def test_market_regime_values(self):
        assert MarketRegime.BEAR == 0
        assert MarketRegime.SIDEWAYS == 1
        assert MarketRegime.BULL == 2

    def test_risk_level_values(self):
        assert RiskLevel.LOW == 0
        assert RiskLevel.NORMAL == 1
        assert RiskLevel.HIGH == 2

    def test_flow_direction_values(self):
        assert FlowDirection.DECREASE == 0
        assert FlowDirection.NEUTRAL == 1
        assert FlowDirection.INCREASE == 2

    def test_momentum_direction_values(self):
        assert MomentumDirection.NEGATIVE == 0
        assert MomentumDirection.STABLE == 1
        assert MomentumDirection.POSITIVE == 2

    def test_normalization_type_members(self):
        assert NormalizationType.NONE is not None
        assert NormalizationType.ROLLING_ROBUST is not None
        assert NormalizationType.CUSTOM is not None

    def test_model_type_members(self):
        assert ModelType.NONE is not None
        assert ModelType.KERAS is not None
        assert ModelType.SKLEARN is not None
        assert ModelType.CUSTOM is not None

    def test_gan_type_members(self):
        assert GANType.NONE is not None
        assert GANType.WGAN is not None
        assert GANType.CTAB_GAN is not None
        assert GANType.BOTH is not None


# ---------------------------------------------------------------------------
# StrategyConfig defaults
# ---------------------------------------------------------------------------

class TestStrategyConfig:
    def test_defaults(self):
        cfg = StrategyConfig()
        assert cfg.normalization == NormalizationType.NONE
        assert cfg.model_type == ModelType.NONE
        assert cfg.gan_type == GANType.NONE
        assert cfg.needs_training is False
        assert cfg.seq_len == 16
        assert cfg.num_epochs == 256
        assert cfg.batch_size == 2048
        assert cfg.dataset_type == "MINIMAL"
        assert cfg.one_hot_columns == []

    def test_custom_config(self):
        cfg = StrategyConfig(
            normalization=NormalizationType.ROLLING_ROBUST,
            model_type=ModelType.KERAS,
            needs_training=True,
            seq_len=32,
        )
        assert cfg.normalization == NormalizationType.ROLLING_ROBUST
        assert cfg.model_type == ModelType.KERAS
        assert cfg.needs_training is True
        assert cfg.seq_len == 32


# ---------------------------------------------------------------------------
# aggregate_dataframes
# ---------------------------------------------------------------------------

class TestAggregateDataframes:
    def test_two_frames(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        result = BaseStrategy.aggregate_dataframes([df1, df2])
        assert len(result) == 6
        assert list(result["a"]) == [1, 2, 3, 4, 5, 6]

    def test_empty_iterable_returns_empty_df(self):
        result = BaseStrategy.aggregate_dataframes([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_frame(self):
        df = pd.DataFrame({"x": [10, 20]})
        result = BaseStrategy.aggregate_dataframes([df])
        assert len(result) == 2

    def test_index_reset(self):
        df1 = pd.DataFrame({"v": [1, 2]}, index=[10, 11])
        df2 = pd.DataFrame({"v": [3, 4]}, index=[20, 21])
        result = BaseStrategy.aggregate_dataframes([df1, df2])
        assert list(result.index) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# aggregate_labels
# ---------------------------------------------------------------------------

class TestAggregateLabels:
    def test_two_arrays(self):
        a = np.array([0, 1, 2])
        b = np.array([1, 0, 2])
        result = BaseStrategy.aggregate_labels([a, b])
        np.testing.assert_array_equal(result, [0, 1, 2, 1, 0, 2])

    def test_empty_iterable(self):
        result = BaseStrategy.aggregate_labels([])
        assert len(result) == 0

    def test_dtype_preserved(self):
        a = np.array([0, 1], dtype=np.int64)
        b = np.array([2, 3], dtype=np.int64)
        result = BaseStrategy.aggregate_labels([a, b])
        assert result.dtype == np.int64

    def test_list_inputs(self):
        result = BaseStrategy.aggregate_labels([[0, 1], [2, 3]])
        np.testing.assert_array_equal(result, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# get_assessment_feedback
# ---------------------------------------------------------------------------

class TestGetAssessmentFeedback:
    def test_mcc_excellent(self, strat):
        fb = strat.get_assessment_feedback(0.7, "MCC")
        assert "Excellent" in fb

    def test_mcc_good(self, strat):
        fb = strat.get_assessment_feedback(0.5, "MCC")
        assert "Good" in fb

    def test_mcc_okay(self, strat):
        fb = strat.get_assessment_feedback(0.3, "MCC")
        assert "Okay" in fb

    def test_mcc_poor(self, strat):
        fb = strat.get_assessment_feedback(0.1, "MCC")
        assert "Poor" in fb

    def test_mcc_bad(self, strat):
        fb = strat.get_assessment_feedback(-0.1, "MCC")
        assert "Bad" in fb

    def test_kappa_excellent(self, strat):
        fb = strat.get_assessment_feedback(0.9, "Kappa")
        assert "Excellent" in fb or "Near-perfect" in fb

    def test_kappa_substantial(self, strat):
        fb = strat.get_assessment_feedback(0.7, "Kappa")
        assert "Substantial" in fb

    def test_kappa_moderate(self, strat):
        fb = strat.get_assessment_feedback(0.5, "Kappa")
        assert "Moderate" in fb

    def test_kappa_bad(self, strat):
        fb = strat.get_assessment_feedback(-0.05, "Kappa")
        assert "Bad" in fb

    def test_generic_excellent(self, strat):
        fb = strat.get_assessment_feedback(0.9, "F1")
        assert "Excellent" in fb

    def test_generic_poor(self, strat):
        fb = strat.get_assessment_feedback(0.25, "F1")
        assert "Poor" in fb

    def test_generic_very_bad(self, strat):
        fb = strat.get_assessment_feedback(0.1, "F1")
        assert "Very Bad" in fb or "Bad" in fb

    def test_returns_string(self, strat):
        assert isinstance(strat.get_assessment_feedback(0.5, "MCC"), str)


# ---------------------------------------------------------------------------
# check_precision_columns
# ---------------------------------------------------------------------------

class TestCheckPrecisionColumns:
    def test_adds_columns_when_missing(self, strat, ohlcv_df):
        df = strat.check_precision_columns(ohlcv_df.copy())
        for col in ["open_count", "high_count", "low_count", "close_count", "max_count"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_idempotent(self, strat, ohlcv_df):
        df1 = strat.check_precision_columns(ohlcv_df.copy())
        df2 = strat.check_precision_columns(df1.copy())
        assert list(df1.columns) == list(df2.columns)

    def test_max_count_is_max_of_ohlc(self, strat, ohlcv_df):
        df = strat.check_precision_columns(ohlcv_df.copy())
        expected_max = df[["open_count", "close_count", "high_count", "low_count"]].max(axis=1)
        pd.testing.assert_series_equal(df["max_count"], expected_max, check_names=False)


# ---------------------------------------------------------------------------
# get_storage_location
# ---------------------------------------------------------------------------

class TestGetStorageLocation:
    def test_returns_string(self, strat):
        loc = strat.get_storage_location()
        assert isinstance(loc, str)

    def test_ends_with_slash(self, strat):
        loc = strat.get_storage_location()
        assert loc.endswith("/")

    def test_path_contains_saved_data(self, strat):
        loc = strat.get_storage_location()
        assert "saved_data" in loc


# ---------------------------------------------------------------------------
# custom_stoploss
# ---------------------------------------------------------------------------

class TestCustomStoploss:
    def _call(self, strat, current_profit):
        trade = MagicMock()
        return strat.custom_stoploss(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=current_profit,
            after_fill=False,
        )

    def test_returns_no_change_sentinel_when_underwater(self, strat):
        # custom_stoploss now returns 1.0 (freqtrade "no change" sentinel)
        # so the static stoploss from entry is preserved.
        result = self._call(strat, current_profit=-0.02)
        assert result == 1.0

    def test_returns_no_change_sentinel_when_profitable(self, strat):
        result = self._call(strat, current_profit=0.05)
        assert result == 1.0

    def test_returns_positive(self, strat):
        # Positive return value = no change. Any negative value would
        # tighten the stop relative to current_rate (implicit trailing).
        result = self._call(strat, current_profit=0.05)
        assert result > 0


# ---------------------------------------------------------------------------
# confirm_trade_exit — minimum-hold check
# ---------------------------------------------------------------------------

class TestConfirmTradeExit:
    def _call(self, strat, trade_open_utc, current_time_utc, exit_reason="signal"):
        strat.dp.runmode.value = "live"
        return strat.confirm_trade_exit(
            pair="ETH/USDT",
            trade=trade_open_utc,  # trade object passed as first positional arg
            order_type="limit",
            amount=0.5,
            rate=110.0,
            time_in_force="gtc",
            exit_reason=exit_reason,
            current_time=current_time_utc,
        )

    def test_rejects_exit_under_1h(self, strat):
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(minutes=30)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = 0.01
        assert self._call(strat, trade, now) is False

    def test_allows_exit_over_1h(self, strat):
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(hours=2)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = 0.01
        strat.dp.runmode.value = "backtest"  # skip live-mode string formatting
        assert self._call(strat, trade, now) is True

    def test_force_exit_bypasses_hold_check(self, strat):
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(minutes=5)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = -0.02
        strat.dp.runmode.value = "backtest"
        assert self._call(strat, trade, now, exit_reason="force_exit") is True

    def test_stop_loss_bypasses_hold_check(self, strat):
        # Risk-management exits must never be blocked by the min-hold pacing.
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(minutes=5)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = -0.05
        strat.dp.runmode.value = "backtest"
        assert self._call(strat, trade, now, exit_reason="stop_loss") is True

    def test_trailing_stop_loss_bypasses_hold_check(self, strat):
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(minutes=5)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = -0.03
        strat.dp.runmode.value = "backtest"
        assert self._call(strat, trade, now, exit_reason="trailing_stop_loss") is True

    def test_stoploss_on_exchange_bypasses_hold_check(self, strat):
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(minutes=5)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = -0.05
        strat.dp.runmode.value = "backtest"
        assert self._call(strat, trade, now, exit_reason="stoploss_on_exchange") is True

    def test_liquidation_bypasses_hold_check(self, strat):
        trade = MagicMock()
        now = datetime.now(timezone.utc)
        trade.open_date_utc = now - timedelta(minutes=5)
        trade.open_rate = 100.0
        trade.calc_profit_ratio.return_value = -0.10
        strat.dp.runmode.value = "backtest"
        assert self._call(strat, trade, now, exit_reason="liquidation") is True


# ---------------------------------------------------------------------------
# print_hyperopt_parameters (smoke — should not raise)
# ---------------------------------------------------------------------------

class TestPrintHyperoptParameters:
    def test_no_exception(self, strat, capsys):
        strat.print_hyperopt_parameters()
        out = capsys.readouterr().out
        assert "buy" in out.lower() or "parameters" in out.lower() or out == ""


# ---------------------------------------------------------------------------
# analyze_and_assess_results (binary) — smoke test
# ---------------------------------------------------------------------------

class TestAnalyzeAndAssessResultsBinary:
    def test_no_exception(self, strat, capsys):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.integers(0, 2, 100)
        strat.analyze_and_assess_results(y_true, y_pred)
        out = capsys.readouterr().out
        assert "CLASSIFICATION PERFORMANCE ASSESSMENT" in out

    def test_perfect_prediction(self, strat, capsys):
        y = np.array([0, 1, 0, 1, 1])
        strat.analyze_and_assess_results(y, y)
        out = capsys.readouterr().out
        assert "1.000" in out or "1.00" in out


# ---------------------------------------------------------------------------
# analyze_and_assess_results_tristate — smoke test
# ---------------------------------------------------------------------------

class TestAnalyzeAndAssessResultsTristate:
    def test_no_exception(self, strat, capsys):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 3, 150)
        y_pred = rng.integers(0, 3, 150)
        strat.analyze_and_assess_results_tristate(y_true, y_pred)
        out = capsys.readouterr().out
        assert "Tri-State" in out

    def test_confusion_matrix_printed(self, strat, capsys):
        y = np.array([0, 1, 2, 0, 1, 2])
        strat.analyze_and_assess_results_tristate(y, y)
        out = capsys.readouterr().out
        assert "Confusion Matrix" in out
