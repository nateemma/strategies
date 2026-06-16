# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
BaseStrategy - Universal base class for ALL trading strategies.

Provides:
 - Common enums (TradingAction, MarketRegime, etc.)
 - Freqtrade boilerplate (ROI, stoploss, trailing, timeframe)
 - Shared hyperopt parameters (guards, custom exit, prediction threshold)
 - Standard callbacks (custom_stoploss, custom_exit, confirm_trade_entry/exit)
 - Template populate_entry_trend / populate_exit_trend
 - Minimal indicator population via DataframePopulator
 - Debug/logging utilities
 - Classification assessment/reporting

Subclasses (or intermediate bases) add family-specific logic:
 - the neural-net family → training, GAN augmentation, normalization
 - the simple-strategy family → signal-based entry/exit
 - the time-series family → wavelet / time-series regression
"""

# --------------------------------
# Top level imports
# --------------------------------
from datetime import datetime
from typing import Optional, List, Any, Dict, Iterable, Union
from functools import reduce
from dataclasses import dataclass, field
from enum import IntEnum, Enum, auto

import numpy as np
import pandas as pd
from pandas import DataFrame

import os
import sys
from pathlib import Path
import logging

from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)

from utils.DataframePopulator import DataframePopulator, DatasetType
from utils.DataframeUtils import DataframeUtils, ScalerType
from utils.Environment import Environment

# --------------------------------
# Global setup
# --------------------------------
pd.options.mode.chained_assignment = None  # default='warn'

log = logging.getLogger(__name__)

# set path such that python can find other directories
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)


# =========================================================================
# Enums
# =========================================================================


class TradingAction(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2


class MarketRegime(IntEnum):
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


class RiskLevel(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


class FlowDirection(IntEnum):
    DECREASE = 0
    NEUTRAL = 1
    INCREASE = 2


class MomentumDirection(IntEnum):
    NEGATIVE = 0
    STABLE = 1
    POSITIVE = 2


# =========================================================================
# Strategy Configuration
# =========================================================================


class NormalizationType(Enum):
    NONE = auto()  # no normalization
    ROLLING_ROBUST = auto()  # most model-based strategies
    CUSTOM = auto()  # custom/nonstandard scaling


class ModelType(Enum):
    NONE = auto()  # no ML model
    KERAS = auto()  # Keras NN families
    SKLEARN = auto()  # sklearn family
    CUSTOM = auto()  # custom regressor pipeline


# GANType lives in the GAN subsystem so it stays independent of strategy code.
# Re-exported here so existing `from Framework.BaseStrategy import GANType` imports
# continue to work without modification.
from GANs.GANType import GANType  # noqa: E402


@dataclass
class StrategyConfig:
    """Declares the capabilities and requirements of a strategy family."""

    # Data processing
    normalization: NormalizationType = NormalizationType.NONE
    norm_data: bool = True
    scale_results: bool = True
    use_pca_reduction: bool = False

    # Model
    model_type: ModelType = ModelType.NONE
    model_per_pair: bool = False
    combine_models: bool = False
    aggregate_pairs: bool = True

    # Training
    needs_training: bool = False
    expanding_window: bool = False
    seq_len: int = 16
    num_epochs: int = 256
    batch_size: int = 2048

    # Training-signal augmentation (peak-finding / wavelet smoothing /
    # synthetic buy-sell pairs).  Independent of GAN augmentation —
    # strategies that GAN-augment often set ``augment_training_data =
    # False`` because the GAN already provides synthetic samples and
    # they only want real signals as the basis.
    augment_training_data: bool = True

    # GAN augmentation — concrete strategies opt in by setting ``gan_type``
    # to anything other than NONE.  ``gan_target_ratio`` is intentionally
    # a Union: single-task strategies set a float, multi-task strategies
    # may set a float (broadcast across tasks), a Dict[task, float], or a
    # nested Dict[task, Dict[class_idx, float]] — same shape as
    # ``balance_multi_task`` accepts.  The strategy never has to know
    # which concrete GAN backend it's calling, only whether the target
    # set is single- or multi-task.
    gan_type: GANType = GANType.NONE
    gan_augment: bool = True
    gan_target_ratio: Any = 0.8
    gan_run_diagnostics: bool = False

    # Feature set
    dataset_type: str = "MINIMAL"  # maps to DatasetType enum

    # One-hot encoded columns (empty = none)
    one_hot_columns: list = field(default_factory=list)


# =========================================================================
# BaseStrategy
# =========================================================================


class BaseStrategy(IStrategy):
    # Strategy configuration (dataclass)
    strategy_config = StrategyConfig()

    # --------------------------------
    # freqtrade controlling parameters
    # --------------------------------

    # Common plot configuration
    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "%train_buy": {"color": "lightgreen"},
                "predict_buy": {"color": "green"},
                "%train_sell": {"color": "orange"},
                "predict_sell": {"color": "red"},
            },
        },
    }

    # Common timeframes
    timeframe = "15m"
    inf_timeframe = "15m"

    # Common strategy flags
    use_custom_stoploss = True
    use_entry_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = True

    # Common startup parameters
    startup_candle_count: int = 64  # must be power of 2
    process_only_new_candles = True

    # --------------------------------
    # hyperopt parameters
    # --------------------------------

    # # Buy hyperspace params:
    # buy_params = {
    #     "entry_adx_threshold": 20.0,
    #     "entry_atr_pct": 0.02,
    #     "entry_bb_width_threshold": 0.014,  # raw 1.4% bandwidth — preserves pre-unbug effective filter
    #     "entry_close_norm_threshold": 0.0,
    #     "entry_enable_guards": True,
    #     "entry_guard_threshold": -0.5,
    #     "entry_rvol_threshold": 1.0,
    #     "prediction_threshold": 0.8,
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "cexit_enable_profit_checks": True,
    #     "cexit_max_days": 2,
    #     "cexit_take_profit": 0.02,
    #     "enable_exit_signal": True,
    #     "exit_close_norm_threshold": 0.0,
    #     "exit_guard_threshold": 0.5,
    # }

    # # Buy parameters:
    # buy_params = {
    #     "entry_adx_threshold": 11.0,
    #     "entry_atr_pct": 0.005,
    #     "entry_bb_width_threshold": 0.04,
    #     "entry_close_norm_threshold": 0.7,
    #     "entry_guard_threshold": -0.1,
    #     "entry_rvol_threshold": 1.2,
    #     "entry_enable_guards": True,  # value loaded from strategy
    #     "min_buy_gain_threshold": 0.007,  # value loaded from strategy
    #     "prediction_threshold": 0.8,  # value loaded from strategy
    #     "training_type": 17,  # value loaded from strategy
    # }

    # # Sell parameters:
    # sell_params = {
    #     "cexit_enable_profit_checks": True,
    #     "cexit_max_days": 4,
    #     "cexit_take_profit": 0.009,
    #     "exit_close_norm_threshold": 0.9,
    #     "exit_guard_threshold": 0.2,
    #     "enable_exit_signal": True,  # value loaded from strategy
    #     "min_sell_loss_threshold": 0.007,  # value loaded from strategy
    # }

    # Buy parameters:
    buy_params = {
        "entry_adx_threshold": 10.0,
        "entry_atr_pct": 0.013,
        "entry_bb_width_threshold": 0.094,
        "entry_close_norm_threshold": 0.6,
        "entry_guard_threshold": 0.1,
        "entry_rvol_threshold": 1.9,
        "prediction_threshold": 0.29,
        "entry_enable_guards": True,  # value loaded from strategy
    }

    # Sell parameters:
    sell_params = {
        "cexit_max_days": 28,
        "cexit_take_profit": 0.04,
        "exit_close_norm_threshold": -0.9,
        "exit_guard_threshold": 0.7,
        "cexit_enable_profit_checks": True,  # value loaded from strategy
        "enable_exit_signal": True,  # value loaded from strategy
    }

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    # Common ROI and stoploss
    minimal_roi = {"0": 0.03}
    # minimal_roi = {"0": 0.025, "60": 0.015, "180": 0.005, "360": 0}
    stoploss = -0.05

    # ATR-adaptive initial stoploss (opt-in).
    # When True, custom_stoploss sets the per-trade initial stop at
    # after_fill to -atr_stoploss_multiplier * atr_pct_roll, clamped to
    # [atr_stoploss_floor, atr_stoploss_cap]. Volatile pairs (high
    # ATR%) get looser stops, calm pairs get tighter stops — pair-
    # agnostic. Falls back to the static `stoploss` if the column is
    # missing or zero, or if the flag is False.
    #
    # Default False here so non-NN strategies retain the no-op
    # custom_stoploss behaviour. The NN base flips this on so every NN variant
    # inherits adaptive stops by default.
    use_atr_adaptive_stoploss = False
    atr_stoploss_multiplier = 2.5
    atr_stoploss_floor = -0.04  # loosest stop allowed (most negative)
    atr_stoploss_cap = -0.02    # tightest stop allowed (closest to zero)

    # Volume-confirmation stoploss tightening (layered on ATR-adaptive).
    # When True (and use_atr_adaptive_stoploss is True), the after-fill stop
    # is tightened by 1/sqrt(rvol) for entries where current-candle volume
    # exceeds the 20-bar rolling mean. rvol ≤ 1 leaves the ATR-derived stop
    # unchanged. Clamps tightest at -0.02. Default True because the mechanism
    # is no-op for strategies that don't enable ATR-adaptive stops, and on
    # NNNC it produced +0.28pp profit / -1.17pp DD / +65% Calmar vs ATR-only.
    use_volume_confirmation_stoploss = True

    opt_base_params = True # flag that allws subclasses to disable framework optimisation

    prediction_threshold = DecimalParameter(
        0.2, 0.9, default=0.5, decimals=2, space="buy", load=True, optimize=opt_base_params
    )

    enable_exit_signal = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=opt_base_params
    )

    entry_enable_guards = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=opt_base_params
    )

    # Bear-market entry gate — causal (uses only past data, unlike the
    # training-label regime which uses a centered rolling mean). Blocks
    # entries when close is sustained below a long EMA by ``entry_bear_deadband``.
    # Independent of the bull side: we only want to skip entries during bear
    # legs, not skip them during sideways or transition periods.
    entry_bear_filter_enable: bool = False
    entry_bear_ema_period: int = 200      # ~200 candles back-look; 8 days on 1h
    entry_bear_deadband: float = 0.02     # close must be 2% below EMA to count as bear

    # Uptrend-only entry gate — stricter than the bear filter. Requires both
    # close > EMA AND EMA rising. Designed to skip range-bound / mean-reverting
    # regimes that pass the bear filter but still bleed via stop_loss because
    # the model's directional signal doesn't translate to capturable moves
    # without a tailwind. See project_gbb_labeler_exhausted.md.
    entry_trend_filter_enable: bool = False
    entry_trend_ema_period: int = 200       # ~50h on 15m — long-trend bias
    entry_trend_slope_window: int = 20      # ~5h slope check on 15m

    entry_guard_threshold = DecimalParameter(
        -1.0, 0.5, default=-0.7, decimals=1, space="buy", load=True, optimize=opt_base_params
    )

    entry_close_norm_threshold = DecimalParameter(
        -0.5, 1.0, default=0.0, decimals=1, space="buy", load=True, optimize=opt_base_params
    )

    entry_adx_threshold = DecimalParameter(
        10.0, 90.0, default=50.0, decimals=0, space="buy", load=True, optimize=opt_base_params
    )

    entry_bb_width_threshold = DecimalParameter(
        0.000, 0.100, default=0.015, decimals=3, space="buy", load=True, optimize=opt_base_params,
    )

    entry_rvol_threshold = DecimalParameter(
        0.0, 5.0, default=2.0, decimals=1, space="buy", load=True, optimize=opt_base_params
    )

    entry_atr_pct = DecimalParameter(
        0.000,
        0.060,
        default=0.010,
        decimals=3,
        space="buy",
        load=True,
        optimize=opt_base_params,
    )

    exit_guard_threshold = DecimalParameter(
        -0.5, 1.0, default=0.7, decimals=1, space="sell", load=True, optimize=opt_base_params
    )

    exit_close_norm_threshold = DecimalParameter(
        -1.0, 1.0, default=0.0, decimals=1, space="sell", load=True, optimize=opt_base_params
    )

    cexit_enable_profit_checks = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=False
    )

    cexit_take_profit = DecimalParameter(
        0.005, 0.04, default=0.008, decimals=3, space="sell", load=True, optimize=opt_base_params
    )

    cexit_max_days = IntParameter(
        1, 30, default=21, space="sell", load=True, optimize=opt_base_params
    )

    # --------------------------------
    # Strategy class-global state
    # --------------------------------

    curr_pair = ""
    custom_trade_info = {}

    # Utilities
    dataframeUtils = None
    dataframePopulator = None
    scaler_type = ScalerType.Robust  # scaler type used for normalisation

    # Debug flags
    first_time = True  # mostly for debug
    first_run = True  # used to identify first time through buy/sell populate funcs
    dbg_verbose = True  # controls debug output
    dbg_curr_df: DataFrame = None  # for debugging of current dataframe

    # Common performance filtering parameters
    PEAK_WINDOW = 6
    # Volume gate at trade-entry time. The candle's quote volume must be
    # at least QUOTE_VOLUME_HEADROOM_MULT × the trade's own quote size
    # (10× ≈ ≤10% market impact target). MIN_QUOTE_VOLUME is an absolute
    # floor for the case where stake is tiny — keeps us out of dust pairs.
    MIN_QUOTE_VOLUME = 1000
    QUOTE_VOLUME_HEADROOM_MULT = 10.0

    # --------------------------------
    # Strategy configuration (override in subclass)
    # --------------------------------

    strategy_config = StrategyConfig()  # default: no model, no normalization

    # =========================================================================
    # Debug / Utility Methods
    # =========================================================================

    def debug_print(self, msg: str):
        """Print debug message if in backtest/plot mode"""
        if self.dbg_verbose and (self.dp.runmode.value in ("backtest", "plot")):
            print(msg)

    def get_storage_location(self) -> str:
        """Determine the root directory for saved_data"""
        from pathlib import Path

        root_dir = str(Path(__file__).parent.parent / "saved_data") + "/"
        return root_dir

    @staticmethod
    def aggregate_dataframes(dataframes: Iterable[DataFrame]) -> DataFrame:
        """Concatenate multiple dataframes, resetting indices to avoid duplicates."""
        import pandas as pd
        frames = [df.reset_index(drop=True) for df in dataframes]
        if not frames:
            return DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def aggregate_labels(
        labels: Iterable[Union[np.ndarray, List[Any]]],
    ) -> np.ndarray:
        """Concatenate label arrays, preserving original dtype."""
        arrays = [np.asarray(lbl) for lbl in labels]
        if not arrays:
            return np.array([])
        return np.concatenate(arrays, axis=0)

    def print_strategy_info(self):
        """Print strategy information - to be overridden by subclasses"""
        print("")
        print("Strategy Parameters/Flags")
        print("")

    def print_hyperopt_parameters(self):
        """Dynamically print all hyperopt parameter values for any strategy"""
        print("\n    Current Hyperopt Parameters:")

        # Access through buy_params and sell_params (most reliable method)
        if hasattr(self, "buy_params") and self.buy_params:
            print("      Buy Parameters:")
            for key, value in self.buy_params.items():
                print(f"        {key}: {value}")

        if hasattr(self, "sell_params") and self.sell_params:
            print("\n      Sell Parameters:")
            for key, value in self.sell_params.items():
                print(f"        {key}: {value}")

        if hasattr(self, "protection_params") and self.protection_params:
            print("\n      Protection Parameters:")
            for key, value in self.protection_params.items():
                print(f"        {key}: {value}")

    # =========================================================================
    # Assessment / Reporting Methods
    # =========================================================================

    def get_assessment_feedback(self, score: float, metric_type: str) -> str:
        """Provides qualitative feedback for a single metric score."""
        if metric_type == "MCC":
            if score >= 0.6:
                return "Excellent. Strong positive correlation."
            if score >= 0.4:
                return "Good. Reliable positive correlation."
            if score >= 0.2:
                return "Okay. Weak but meaningful correlation."
            if score >= 0.05:
                return "Poor. Correlation is barely better than random."
            return "Bad. Correlation is near zero or negative."

        if metric_type == "Kappa":
            # Landis & Koch 1977 guidelines
            if score >= 0.81:
                return "Excellent. Near-perfect agreement."
            if score >= 0.61:
                return "Substantial. Strong agreement."
            if score >= 0.41:
                return "Moderate. Meaningful agreement."
            if score >= 0.21:
                return "Fair. Weak agreement."
            if score >= 0.0:
                return "Slight. Agreement is barely above chance."
            return "Bad. Agreement is poor or non-existent."

        # Generic score for Precision/Recall/F1
        if score >= 0.8:
            return "Excellent. Very strong performance."
        if score >= 0.6:
            return "Good. Reliable performance."
        if score >= 0.4:
            return "Okay. Acceptable, but needs improvement."
        if score >= 0.2:
            return "Poor. Significant room for improvement."
        return "Very Bad. Performance is extremely low."

    def _print_assessment_header(
        self, title: str = "CLASSIFICATION PERFORMANCE ASSESSMENT"
    ) -> None:
        """Print the header section for assessment reports."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def _print_metrics_table_header(self) -> None:
        """Print the header for the metrics table."""
        COL_1_WIDTH = 25
        COL_2_WIDTH = 7
        COL_3_WIDTH = 40

        print("\n--- Qualitative Assessment of Key Metrics ---")
        header = f"{'Metric':<{COL_1_WIDTH}} {'Score':<{COL_2_WIDTH}} {'Assessment':<{COL_3_WIDTH}} {'Trading Context'}"
        print(header)
        print(
            "-" * COL_1_WIDTH
            + "   "
            + "-" * COL_2_WIDTH
            + "   "
            + "-" * COL_3_WIDTH
            + "   "
            + "-" * 30
        )

    def _print_metric_row(
        self, metric_name: str, score: float, metric_type: str, context: str
    ) -> None:
        """Print a single row in the metrics table."""
        COL_1_WIDTH = 25
        COL_2_WIDTH = 7
        COL_3_WIDTH = 40

        assessment = self.get_assessment_feedback(score, metric_type)
        print(
            f"{metric_name:<{COL_1_WIDTH}} {score:<{COL_2_WIDTH}.3f} {assessment:<{COL_3_WIDTH}} {context}"
        )

    def analyze_and_assess_results(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Calculates essential metrics and provides qualitative feedback with fixed-width formatting.
        Binary classification version (focuses on Buy signals).

        Args:
            y_true: The ground truth labels (1D array, binary: 0/1).
            y_pred: The predicted labels (1D array, binary: 0/1).
        """

        # Calculate all metrics
        report = classification_report(
            y_true, y_pred, digits=3, zero_division=0, output_dict=True
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # FIX: Use string key '1' for the classification report dictionary lookup
        class_1_metrics = report.get("1", {})

        # Extract metrics for the positive class (1.0)
        precision_1 = class_1_metrics.get("precision", 0.0)
        recall_1 = class_1_metrics.get("recall", 0.0)
        f1_1 = class_1_metrics.get("f1-score", 0.0)

        # ------------------ PRINTING ------------------
        self._print_assessment_header("CLASSIFICATION PERFORMANCE ASSESSMENT (Binary)")

        # 1. Classification Report
        print("--- Detailed Classification Report ---")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))

        # 2. Key Metric Analysis Table
        self._print_metrics_table_header()

        # Print analysis for the classification report metrics
        self._print_metric_row(
            "Precision (1.0)",
            precision_1,
            "F1",
            "Trading objective: Avoid False Alarms",
        )
        self._print_metric_row(
            "Recall (1.0)", recall_1, "F1", "Trading objective: Find All Opportunities"
        )
        self._print_metric_row(
            "F1-Score (1.0)", f1_1, "F1", "Balanced performance on the Buy signal"
        )

        # Print analysis for the summary metrics
        self._print_metric_row(
            "MCC", mcc, "MCC", "Correlation between prediction and reality"
        )
        self._print_metric_row(
            "Cohen's Kappa", kappa, "Kappa", "Agreement better than random chance"
        )

        # 3. Final Summary Recommendation
        if precision_1 < 0.4:
            recommendation = f"🚨 WARNING: Precision (Buy Signal) is only {precision_1:.3f}. The model predicts a 'Buy' signal but is **wrong {int((1 - precision_1) * 100)}% of the time**. This is dangerous for trading. **Priority: Improve Precision.**"
        elif recall_1 < 0.5:
            recommendation = f"⚠️ CONCERN: Recall (Buy Signal) is only {recall_1:.3f}. The model is missing over {int((1 - recall_1) * 100)}% of the available 'Buy' signals. **Priority: Improve Recall and overall F1.**"
        elif f1_1 < 0.6:
            recommendation = f"✅ ACCEPTABLE: Performance is okay, but the F1-Score of {f1_1:.3f} needs to be higher for a robust trading strategy. **Focus: Fine-tune for higher F1/MCC.**"
        else:
            recommendation = f"🌟 GOOD PERFORMANCE: The model shows strong balance (F1-Score {f1_1:.3f}) and reliable correlation (MCC {mcc:.3f})."

        print("\n--- Summary and Recommendation ---")
        print(recommendation)
        print("=" * 80)

    def analyze_and_assess_results_tristate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Calculates essential metrics and provides qualitative feedback with fixed-width formatting.
        Tri-state classification version (Sell=0, Hold=1, Buy=2).

        Args:
            y_true: The ground truth labels (1D array, tri-state: 0=Sell, 1=Hold, 2=Buy).
            y_pred: The predicted labels (1D array, tri-state: 0=Sell, 1=Hold, 2=Buy).
        """

        # Calculate all metrics
        report = classification_report(
            y_true, y_pred, digits=3, zero_division=0, output_dict=True
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Get macro and micro averages
        macro_avg = report.get("macro avg", {})
        micro_avg = report.get("weighted avg", {})

        macro_precision = macro_avg.get("precision", 0.0)
        macro_recall = macro_avg.get("recall", 0.0)
        macro_f1 = macro_avg.get("f1-score", 0.0)

        micro_precision = micro_avg.get("precision", 0.0)
        micro_recall = micro_avg.get("recall", 0.0)
        micro_f1 = micro_avg.get("f1-score", 0.0)

        # Extract metrics for each class
        class_0_metrics = report.get("0.0", report.get("0", {}))  # SELL
        class_1_metrics = report.get("1.0", report.get("1", {}))  # HOLD
        class_2_metrics = report.get("2.0", report.get("2", {}))  # BUY

        precision_sell = class_0_metrics.get("precision", 0.0)
        recall_sell = class_0_metrics.get("recall", 0.0)
        f1_sell = class_0_metrics.get("f1-score", 0.0)
        support_sell = class_0_metrics.get("support", 0)

        precision_hold = class_1_metrics.get("precision", 0.0)
        recall_hold = class_1_metrics.get("recall", 0.0)
        f1_hold = class_1_metrics.get("f1-score", 0.0)
        support_hold = class_1_metrics.get("support", 0)

        precision_buy = class_2_metrics.get("precision", 0.0)
        recall_buy = class_2_metrics.get("recall", 0.0)
        f1_buy = class_2_metrics.get("f1-score", 0.0)
        support_buy = class_2_metrics.get("support", 0)

        # ------------------ PRINTING ------------------
        self._print_assessment_header(
            "CLASSIFICATION PERFORMANCE ASSESSMENT (Tri-State)"
        )

        # 1. Classification Report
        print("--- Detailed Classification Report ---")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\n--- Confusion Matrix ---")
        print("Rows = True, Columns = Predicted")
        print(f"{'':<8} {'Sell':>8} {'Hold':>8} {'Buy':>8}")
        print(f"{'Sell':>8} {cm[0,0]:>8} {cm[0,1]:>8} {cm[0,2]:>8}")
        print(f"{'Hold':>8} {cm[1,0]:>8} {cm[1,1]:>8} {cm[1,2]:>8}")
        print(f"{'Buy':>8} {cm[2,0]:>8} {cm[2,1]:>8} {cm[2,2]:>8}")

        # 3. Per-Class Metrics Table
        self._print_metrics_table_header()

        # SELL class metrics
        self._print_metric_row(
            f"Precision (Sell)",
            precision_sell,
            "F1",
            f"Avoid false sell signals (support: {support_sell})",
        )
        self._print_metric_row(
            f"Recall (Sell)",
            recall_sell,
            "F1",
            f"Find all sell opportunities (support: {support_sell})",
        )
        self._print_metric_row(
            f"F1-Score (Sell)",
            f1_sell,
            "F1",
            f"Balanced performance on Sell signal (support: {support_sell})",
        )

        # HOLD class metrics
        self._print_metric_row(
            f"Precision (Hold)",
            precision_hold,
            "F1",
            f"Correctly identify neutral periods (support: {support_hold})",
        )
        self._print_metric_row(
            f"Recall (Hold)",
            recall_hold,
            "F1",
            f"Find all neutral periods (support: {support_hold})",
        )
        self._print_metric_row(
            f"F1-Score (Hold)",
            f1_hold,
            "F1",
            f"Balanced performance on Hold signal (support: {support_hold})",
        )

        # BUY class metrics
        self._print_metric_row(
            f"Precision (Buy)",
            precision_buy,
            "F1",
            f"Avoid false buy signals (support: {support_buy})",
        )
        self._print_metric_row(
            f"Recall (Buy)",
            recall_buy,
            "F1",
            f"Find all buy opportunities (support: {support_buy})",
        )
        self._print_metric_row(
            f"F1-Score (Buy)",
            f1_buy,
            "F1",
            f"Balanced performance on Buy signal (support: {support_buy})",
        )

        # 4. Overall Metrics
        print("\n--- Overall Performance Metrics ---")
        self._print_metrics_table_header()

        self._print_metric_row(
            "Macro Avg Precision",
            macro_precision,
            "F1",
            "Average precision across all classes",
        )
        self._print_metric_row(
            "Macro Avg Recall", macro_recall, "F1", "Average recall across all classes"
        )
        self._print_metric_row(
            "Macro Avg F1-Score", macro_f1, "F1", "Average F1-score across all classes"
        )
        self._print_metric_row(
            "Weighted Avg Precision",
            micro_precision,
            "F1",
            "Support-weighted average precision",
        )
        self._print_metric_row(
            "Weighted Avg Recall", micro_recall, "F1", "Support-weighted average recall"
        )
        self._print_metric_row(
            "Weighted Avg F1-Score", micro_f1, "F1", "Support-weighted average F1-score"
        )
        self._print_metric_row(
            "MCC", mcc, "MCC", "Correlation between prediction and reality"
        )
        self._print_metric_row(
            "Cohen's Kappa", kappa, "Kappa", "Agreement better than random chance"
        )

        # 5. Final Summary Recommendation
        worst_class = None
        worst_f1 = min(f1_sell, f1_hold, f1_buy)
        if worst_f1 == f1_sell:
            worst_class = "Sell"
            worst_precision = precision_sell
            worst_recall = recall_sell
        elif worst_f1 == f1_hold:
            worst_class = "Hold"
            worst_precision = precision_hold
            worst_recall = recall_hold
        else:
            worst_class = "Buy"
            worst_precision = precision_buy
            worst_recall = recall_buy

        if worst_precision < 0.4:
            recommendation = (
                f"🚨 WARNING: {worst_class} class precision is only {worst_precision:.3f}. "
                f"The model predicts '{worst_class}' but is **wrong {int((1 - worst_precision) * 100)}% of the time**. "
                f"This is dangerous for trading. **Priority: Improve {worst_class} Precision.**"
            )
        elif worst_recall < 0.5:
            recommendation = (
                f"⚠️ CONCERN: {worst_class} class recall is only {worst_recall:.3f}. "
                f"The model is missing over {int((1 - worst_recall) * 100)}% of available '{worst_class}' signals. "
                f"**Priority: Improve {worst_class} Recall and overall F1.**"
            )
        elif macro_f1 < 0.6:
            recommendation = (
                f"✅ ACCEPTABLE: Performance is okay, but the macro F1-Score of {macro_f1:.3f} "
                f"needs to be higher for a robust trading strategy. "
                f"**Focus: Fine-tune for higher F1/MCC, especially for {worst_class} class.**"
            )
        else:
            recommendation = (
                f"🌟 GOOD PERFORMANCE: The model shows strong balance "
                f"(Macro F1-Score {macro_f1:.3f}, Weighted F1-Score {micro_f1:.3f}) "
                f"and reliable correlation (MCC {mcc:.3f}). "
                f"All classes performing reasonably well."
            )

        print("\n--- Summary and Recommendation ---")
        print(recommendation)
        print("=" * 80)

    def print_probability_stats(
        self, task: str, name: str, probabilities: np.ndarray, threshold: float = None
    ):
        """Utility function to print probability statistics and distribution"""

        num_nans = np.sum(np.isnan(probabilities))  # Should be 0 after replacement
        if num_nans > 0:
            print(
                f"    *** WARNING: {num_nans} NaN(s) in raw predictions for {name} ***"
            )
            probabilities = np.nan_to_num(probabilities, nan=0.0)

        self.debug_print(f"    {task} - {name}:")
        self.debug_print(
            f"        min: {probabilities.min():.3f} max: {probabilities.max():.3f}, mean: {probabilities.mean():.3f}"
        )

        if threshold is not None:
            signals_above_threshold = np.sum(probabilities > threshold)
            signal_percentage = 100.0 * signals_above_threshold / len(probabilities)
            self.debug_print(
                f"        signals > {threshold:.3f}: {signals_above_threshold} ({signal_percentage:.1f}%)"
            )

        # Show probability distribution as compact arrays
        bins = np.bincount(
            (probabilities * 10).astype(int), minlength=11
        )  # 0.0 to 1.0 in 0.1 steps
        percentages = bins / len(probabilities)
        cumulative = np.cumsum(percentages)
        self.debug_print(f"        counts: {bins.tolist()}")
        self.debug_print(f"        percentages: {percentages.round(2).tolist()}")
        self.debug_print(f"        cumulative: {cumulative.round(2).tolist()}")

    def print_distribution_compact(self, name: str, distribution: np.ndarray) -> None:
        counts = np.bincount(distribution, minlength=3)
        percentages = counts / len(distribution) * 100
        percent_str = (
            f"[{percentages[0]:.1f}%, {percentages[1]:.1f}%, {percentages[2]:.1f}%]"
        )
        self.debug_print(f"      {name} distribution: {counts} {percent_str}")

    # =========================================================================
    # Dataframe Utility Methods
    # =========================================================================

    def check_precision_columns(self, dataframe: DataFrame):
        """Add precision columns that are normally only added during backtesting."""
        precision_columns = [
            "open_count",
            "high_count",
            "low_count",
            "close_count",
            "max_count",
        ]
        missing_columns = [
            col for col in precision_columns if col not in dataframe.columns
        ]

        if missing_columns:
            for col in ["open", "high", "low", "close"]:
                dataframe[f"{col}_count"] = (
                    dataframe[col]
                    .round(14)
                    .apply("{:.15f}".format)
                    .str.extract(r"\.(\d*[1-9])")[0]
                    .str.len()
                )
            dataframe["max_count"] = dataframe[
                ["open_count", "close_count", "high_count", "low_count"]
            ].max(axis=1)
        return dataframe

    # =========================================================================
    # bot_start — one-time initialisation (freqtrade lifecycle hook)
    # =========================================================================

    def bot_start(self, **kwargs) -> None:
        """
        Called once after the strategy is instantiated and the data provider
        has been attached.  Do all per-bot one-time setup here so that
        ``iteration_init`` (called per ``populate_indicators`` cycle) stays
        cheap.

        Subclasses that override this MUST call ``super().bot_start(**kwargs)``
        so the base setup runs.
        """
        self.debug_print("")
        self.debug_print("----------------------")
        self.debug_print(self.__class__.__name__)
        self.debug_print("----------------------")
        self.debug_print("")

        if self.dp is not None and self.dp.runmode.value in ("util_no_exchange"):
            print(f"    run mode: {self.dp.runmode.value}")

        Environment().print_environment()
        self.print_hyperopt_parameters()

        # One-shot construction of the shared utility helpers.  The
        # ``reset_scaler`` call lives in ``iteration_init`` because it must
        # happen at the start of every populate_indicators() cycle.
        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(self.scaler_type)

        if self.dataframePopulator is None:
            self.dataframePopulator = DataframePopulator()

        # Mark the one-time block as complete so anything still checking
        # ``self.first_time`` (e.g. archived strategies) sees the right state.
        self.first_time = False

    # =========================================================================
    # Iteration Init — per-populate_indicators setup (lightweight)
    # =========================================================================

    def iteration_init(self):
        """Called at the start of each populate_indicators() cycle.

        Only per-iteration state belongs here — the bulk of one-time setup
        lives in :meth:`bot_start`.  Defensive instantiation of the utility
        helpers is preserved in case a subclass invokes populate_indicators
        without going through ``ft_bot_start`` (e.g. unit tests).
        """
        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(self.scaler_type)
        else:
            self.dataframeUtils.reset_scaler()

        if self.dataframePopulator is None:
            self.dataframePopulator = DataframePopulator()

    # =========================================================================
    # Virtual Methods (override in subclass / intermediate base)
    # =========================================================================

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Add strategy/family-specific indicators. Override in subclasses."""
        return dataframe

    def add_debug_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Add debug-only indicators (e.g. hidden columns for plotting). Override in subclasses."""
        return dataframe

    def get_entry_conditions(self, dataframe: DataFrame):
        """Return a boolean Series/array for entry signals. Must be overridden."""
        return None

    def get_exit_conditions(self, dataframe: DataFrame):
        """Return a boolean Series/array for exit signals. Must be overridden."""
        return None

    # =========================================================================
    # populate_indicators — base version
    # =========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Common indicator population using DataframePopulator minimal set.

        Subclasses should override this (calling super()) to add their own logic
        (e.g. a training loop, or signal generation).
        """

        curr_pair = metadata["pair"]
        self.curr_pair = curr_pair
        self.dbg_curr_df = dataframe

        self.iteration_init()

        if self.dbg_verbose:
            self.debug_print(f"    {curr_pair} - adding indicators...")

        dataframe = self.check_precision_columns(dataframe)
        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
        )
        dataframe = self.add_additional_indicators(dataframe)
        dataframe = self.add_debug_indicators(dataframe)

        self.dbg_curr_df = dataframe

        return dataframe

    # =========================================================================
    # Freqtrade Callbacks — populate_entry_trend / populate_exit_trend
    # =========================================================================

    def is_bear_market(self, dataframe: DataFrame):
        """Causal bear-market detector for entry gating.

        Returns a boolean Series of length len(dataframe), True where the
        market is in a sustained bear regime relative to its trailing EMA.

        Uses a standard (causal) EMA — only past data — because this is
        evaluated at entry time when the future doesn't exist. Distinct from
        the *centered* rolling mean (looks both ways) used to build training
        labels. Same idea, different causality requirement.

        Bear when ``close < EMA(entry_bear_ema_period) * (1 - entry_bear_deadband)``.
        Deadband prevents flipping on small crosses; EMA gives multi-day
        persistence on hourly bars.
        """
        close = dataframe["close"]
        ema = close.ewm(span=self.entry_bear_ema_period, adjust=False).mean()
        return close < ema * (1.0 - self.entry_bear_deadband)

    def is_uptrend(self, dataframe: DataFrame):
        """Causal uptrend detector for entry gating.

        Returns True where close > EMA(entry_trend_ema_period).

        The slope check (EMA rising over a fixed window) was removed
        2026-05-30 because it cut entries during normal mid-uptrend
        pullbacks — SOL went from +4.63% to -1.19% under the slope variant.
        ``close > EMA`` alone catches the regime gate without false negatives.
        """
        close = dataframe["close"]
        ema = close.ewm(span=self.entry_trend_ema_period, adjust=False).mean()
        return close > ema

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Common entry trend population - calls strategy-specific method for custom conditions"""
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""
        curr_pair = metadata["pair"]

        self.curr_pair = curr_pair

        if self.first_run:
            self.first_run = False

        # Call strategy-specific method to add custom conditions
        model_conditions = self.get_entry_conditions(dataframe)
        conditions.append(model_conditions)
        # Set entry tags
        dataframe.loc[model_conditions, "enter_tag"] += "model_entry "

        # # DEBUG
        # entry_count = np.sum(model_conditions)
        # self.debug_print(f"BaseStrategy entry_count: {entry_count}")

        # Bear-market gate — applies even when entry guards are off because
        # avoiding bear-leg entries is a separate concern from the volatility/
        # volume guards. Blocks new longs while close is below the long EMA.
        if getattr(self, "entry_bear_filter_enable", False):
            conditions.append(~self.is_bear_market(dataframe))

        # Uptrend-only gate — stricter than the bear filter. Requires close
        # above a long EMA AND that EMA rising. Trips before stop_loss can
        # eat the model's directional signal in range-bound regimes.
        if getattr(self, "entry_trend_filter_enable", False):
            conditions.append(self.is_uptrend(dataframe))

        # Common guard conditions
        if self.entry_enable_guards.value:
            conditions.append(dataframe["volume"] > 0.0)
            conditions.append(dataframe["rvol"] > self.entry_rvol_threshold.value)
            conditions.append(dataframe["atr_pct_roll"] > self.entry_atr_pct.value)

            conditions.append(
                dataframe["guard_metric"] < self.entry_guard_threshold.value
            )
            conditions.append(
                dataframe["close_norm"] < self.entry_close_norm_threshold.value
            )
            conditions.append(dataframe["adx"] > self.entry_adx_threshold.value)
            conditions.append(
                dataframe["bb_width"] > self.entry_bb_width_threshold.value
            )

        # Apply conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1
        else:
            dataframe["enter_long"] = 0

        if self.dp.runmode.value in ("backtest", "plot"):
            if self.strategy_config.model_type != ModelType.NONE:
                if "%train_buy" in dataframe.columns:
                    # run comparison of predict_buy and %train_buy
                    self.debug_print(f"\n{curr_pair}")
                    self.debug_print(f"    Comparing actual vs predicted signals")

                    if self.enable_exit_signal.value:
                        # tri-state version:
                        y_true = np.ones(len(dataframe))
                        y_true = np.where(dataframe["%train_buy"] > 0.5, 2, y_true)
                        y_true = np.where(dataframe["%train_sell"] > 0.5, 0, y_true)
                        y_pred = np.ones(len(dataframe))
                        y_pred = np.where(dataframe["predict_buy"] > 0.5, 2, y_pred)
                        y_pred = np.where(dataframe["predict_sell"] > 0.5, 0, y_pred)
                        self.analyze_and_assess_results_tristate(y_true, y_pred)
                    else:
                        # Binary version
                        y_true = np.where(dataframe["%train_buy"] > 0.5, 1, 0)
                        y_pred = np.where(dataframe["predict_buy"] > 0.5, 1, 0)
                        self.analyze_and_assess_results(y_true, y_pred)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Common exit trend population - calls strategy-specific method for custom conditions"""
        conditions = []
        dataframe.loc[:, "exit_tag"] = ""
        dataframe["exit_long"] = 0

        if not self.enable_exit_signal.value:
            return dataframe

        curr_pair = metadata["pair"]

        # Call strategy-specific method to add custom conditions
        model_conditions = self.get_exit_conditions(dataframe)
        conditions.append(model_conditions)
        dataframe.loc[model_conditions, "exit_tag"] += "model_exit "

        # Add common conditions

        if self.entry_enable_guards.value:
            conditions.append(dataframe["rvol"] > 2.0)

            # common guard conditions
            conditions.append(
                dataframe["guard_metric"] > self.exit_guard_threshold.value
            )

        # Apply conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
        else:
            dataframe["exit_long"] = 0

        return dataframe

    # =========================================================================
    # Custom Stoploss
    # =========================================================================

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        # ATR-adaptive initial stop (opt-in via use_atr_adaptive_stoploss).
        # Only the after_fill call returns a non-1.0 value, so freqtrade
        # locks the volatility-adjusted stop at entry and leaves it static
        # for the rest of the trade — no trailing semantics.
        if self.use_atr_adaptive_stoploss and after_fill:
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if not dataframe.empty:
                    atr_pct = float(dataframe.iloc[-1].get("atr_pct_roll", 0.0))
                    if atr_pct > 0:
                        stop = -self.atr_stoploss_multiplier * atr_pct
                        stop = max(min(stop, self.atr_stoploss_cap), self.atr_stoploss_floor)
                        if self.use_volume_confirmation_stoploss:
                            vol_mean = float(
                                dataframe["volume"].rolling(20, min_periods=5).mean().iloc[-1]
                            )
                            cur_vol = float(dataframe.iloc[-1].get("volume", 0.0))
                            if vol_mean > 0:
                                rvol = cur_vol / vol_mean
                                if rvol > 1.0:
                                    stop = min(stop / (rvol ** 0.5), -0.02)
                        return stop
            except Exception:
                pass

        # No trailing — preserve the initial static stoploss from entry. The
        # freqtrade convention is that a positive return value means
        # "no change to the current stoploss", so the trade keeps the entry-
        # time stop (self.stoploss = -0.05 by default) until exit.
        #
        # The previous implementation returned ``self.stoploss`` here, which
        # freqtrade interprets relative to current_rate — ratcheting the stop
        # up as price rises. That created an implicit trailing stop even with
        # trailing_stop=False, and caught a large fraction of winners on
        # normal volatility (see backtest 2024-04 to 2026-04: 950 trades
        # exited via trailing_stop_loss at -4.24% avg).
        return 1.0

        """
        # Alternative: trail only when profitable. Restore by replacing the
        # ``return 1.0`` above with the block below.

        if current_profit < self.cexit_take_profit.value:
            return self.stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = -(current_profit / 2)  # Make it negative!

        return desired_stoploss
        """

    # =========================================================================
    # Custom Exit
    # =========================================================================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        last_candle = dataframe.iloc[-1].squeeze()

        # if not self.use_custom_stoploss:
        #     return None

        if trade.is_short:
            print("    short trades not yet supported in custom_exit()")
            return None

        if self.cexit_enable_profit_checks.value:
            # Currently in profit - check for exit conditions
            if current_profit > 0.0:

                # Enhanced RSI conditions
                if "rsi" in last_candle:
                    current_rsi = last_candle["rsi"]

                    # Strong sell: RSI > 80 and declining
                    if current_rsi > 80 and len(dataframe) > 1:
                        prev_rsi = dataframe.iloc[-2]["rsi"]
                        if prev_rsi > current_rsi:
                            return "rsi_strong_sell"

                    # Moderate sell: RSI > 75 and declining with high profit
                    elif (
                        current_rsi > 75
                        and current_profit > 0.02
                        and len(dataframe) > 1
                    ):
                        prev_rsi = dataframe.iloc[-2]["rsi"]
                        if prev_rsi > current_rsi:
                            return "rsi_moderate_sell"

                    # Conservative sell: RSI > 70 and declining with any profit
                    elif (
                        current_rsi > 70
                        and current_profit > 0.005
                        and len(dataframe) > 1
                    ):
                        prev_rsi = dataframe.iloc[-2]["rsi"]
                        if prev_rsi > current_rsi:
                            return "rsi_conservative_sell"

                # strong sell signal, in profit
                if "guard_metric" in last_candle:
                    if last_candle["guard_metric"] > 0.98:
                        return "metric_overbought"

                if current_profit > self.cexit_take_profit.value:
                    return "take_profit"

        # Time-based exits (apply to both profitable and losing trades)
        time_delta = current_time - trade.open_date_utc
        num_hours = time_delta.total_seconds() / 3600
        num_days = time_delta.days

        # Exit if trade has been open too long
        if (num_hours >= 12) & (current_profit > 0.005):  # 12 hours with some profit
            return "unclog_12h"

        if (num_days >= 1) & (current_profit >= 0):  # 1 day with any profit
            return "unclog_1d"

        if num_days >= self.cexit_max_days.value:  # max hold
            return "max_hold"

        # Strategy-specific exit hook — last chance to exit before
        # falling through to None (which delegates to the static stoploss
        # / trailing-stop / minimal_roi config). Subclasses override
        # ``strategy_custom_exit`` to add model-prediction-based bailouts,
        # regime-shift exits, etc. without re-implementing the time/profit
        # checks above. The default implementation returns None.
        reason = self.strategy_custom_exit(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            dataframe=dataframe,
            last_candle=last_candle,
            **kwargs,
        )
        if reason is not None:
            return reason

        return None

    def strategy_custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        dataframe,
        last_candle,
        **kwargs,
    ) -> Optional[str]:
        """Strategy-specific exit hook called from ``custom_exit`` after the
        standard profit-based and time-based checks have declined to exit.

        Return an exit-reason string to trigger an exit, or None to defer
        to the static stoploss / trailing-stop. ``dataframe`` and
        ``last_candle`` are passed in so subclasses can read indicator or
        model-prediction columns without re-fetching the analyzed frame.

        Default implementation is a no-op (returns None). Subclasses
        like BaseNNMTStrategy override to inspect the model's per-bar
        task predictions and bail out on adverse signal flips.
        """
        return None

    # =========================================================================
    # Confirm Trade Entry / Exit
    # =========================================================================

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:

        # Skip volume check only in plot / other modes (no trading happens).
        # Run in backtest + hyperopt for live-parity; verbose logging is
        # suppressed there since per-trade prints flood the output.
        if self.dp.runmode.value in ("plot", "other"):
            return True

        is_live = self.dp.runmode.value in ("live", "dry_run")
        if is_live:
            self.debug_print("")
            self.debug_print(f"    Trade Entry: {pair}, rate: {round(rate, 4)}")

        # check volume — require headroom over the trade's own quote size
        # so our order doesn't dominate the candle (slippage protection).
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        last_candle = dataframe.iloc[-1].squeeze()
        quote_volume = last_candle["volume"] * last_candle["close"]
        trade_quote_size = amount * rate
        required_volume = max(
            self.MIN_QUOTE_VOLUME,
            self.QUOTE_VOLUME_HEADROOM_MULT * trade_quote_size,
        )
        if quote_volume < required_volume:
            if is_live:
                print(
                    f"    *** Reject Trade: {pair}, volume: {last_candle['volume']}, "
                    f"quote volume: {quote_volume:.2f}, trade size: {trade_quote_size:.2f}, "
                    f"required: {required_volume:.2f}"
                )
            return False

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:

        # Reject exit if trade has been open for less than 1 hour
        # (Exception for emergency exits)
        if exit_reason not in ["force_exit", "emergency_exit"]:
            # Ensure timezone awareness for comparison
            from datetime import timezone
            t_current = current_time
            if t_current.tzinfo is None:
                 t_current = t_current.replace(tzinfo=timezone.utc)
            
            t_open = trade.open_date_utc
            if t_open.tzinfo is None:
                 t_open = t_open.replace(tzinfo=timezone.utc)

            duration_hours = (t_current - t_open).total_seconds() / 3600.0
            if duration_hours < 1.0:
                return False

        # remaining logic for live logging
        if self.dp.runmode.value in ("backtest", "plot", "hyperopt", "other"):
            return True

        s_entry = str(round(trade.open_rate, 4))
        s_exit = str(round(rate, 4))
        s_profit = str(round(trade.calc_profit_ratio(rate), 4))
        pstr = "*    Trade Exit: " + pair + "  entry:" + s_entry + "  exit:" + s_exit + " profit:" + s_profit + " reason: " + exit_reason  # type: ignore
        print(pstr, flush=True)

        return True
