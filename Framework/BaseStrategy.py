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

from Framework.StrategyDiagnostics import StrategyDiagnostics

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

    # Buy/sell LABEL CONFLICT RESOLUTION. Renamed from ``apply_label_conflict_resolution``
    # 2026-09-09: that name, and the comment that stood here, were wrong and
    # actively caused a misconfiguration.
    #
    # This flag no longer gates any augmentation. "Trick 1" (2-bar dilation) was
    # removed 2026-09-08, so ``augment_training_signals`` now performs ONLY:
    #   Trick 2 — a sell overrides a coincident buy
    #   Trick 3 — buys in the N bars before a sell are cleared (they cannot reach
    #             target before the reversal; measured, 69.4% of them at N=2)
    #
    # It is NOT a substitute for GAN augmentation and setting it False because
    # "the GAN already provides synthetic samples" is a CATEGORY ERROR — a GAN
    # supplies synthetic FEATURES, not resolved LABELS. Measured cost of getting
    # this wrong, NNNC_DDPM_MLX over 6,814 trades: buy precision 0.465 against
    # recall 0.990 (2.13x too many signals), ATR stops outnumbering take-profits
    # 2,955 to 2,338, and -23.97% with a 43.6% drawdown — while MCC read 0.630.
    #
    # Leave True unless the labels are not binary buy/sell (NNPredict) or you
    # deliberately want raw unresolved labels (the Debug/* tools).
    apply_label_conflict_resolution: bool = True

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


class BaseStrategy(StrategyDiagnostics, IStrategy):
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

    # # Buy parameters:
    # buy_params = {
    #     "entry_adx_threshold": 10.0,
    #     "entry_atr_pct": 0.013,
    #     "entry_bb_width_threshold": 0.094,
    #     "entry_close_norm_threshold": 0.6,
    #     "entry_guard_threshold": 0.1,
    #     "entry_rvol_threshold": 1.9,
    #     "prediction_threshold": 0.29,
    #     "entry_enable_guards": True,  # value loaded from strategy
    # }

    # # Sell parameters:
    # sell_params = {
    #     "cexit_max_days": 28,
    #     "cexit_take_profit": 0.04,
    #     "exit_close_norm_threshold": -0.9,
    #     "exit_guard_threshold": 0.7,
    #     "cexit_enable_profit_checks": True,  # value loaded from strategy
    #     "enable_exit_signal": True,  # value loaded from strategy
    # }

    # Buy parameters:
    # NOTE: multi-task-only parameters do NOT belong here. apply_task_filters,
    # bias_profit_high/low and bias_trading_buy/sell are declared as Parameters
    # solely in NNMTStrategy, so entries here were dead config for every other
    # strategy — carried into SimpleStrategy, TSPredict, NNAnomalyStrategy and
    # DebugNNStrategy via `{**BaseStrategy.buy_params, ...}` with no Parameter
    # to receive them. They arrived as hyperopt-dump leftovers (hence the
    # "value loaded from strategy" markers) and were actively misleading: they
    # made apply_task_filters look enabled base-wide when NNMT never read them.
    # 2026-09-06: the six QUALITY guards now default OPEN.
    #
    # The guard chain selected ~99% of entries, which made the model irrelevant:
    # a PERFECT model was worth only 0.68pp more than the real one, so no model,
    # GAN or architecture work could ever pay. Opening them moves selection to
    # the model -- ~25pp of headroom and 4.5x the trades.
    #
    # They are opened by NEUTRALISING each threshold rather than by
    # entry_enable_guards=False, because that flag ALSO disables the
    # `volume > 0` data-sanity check which lives in the same block. Keep
    # entry_enable_guards True. Out-of-range values are honoured, not clamped.
    #
    # NOT a higher-return setting: across seeds it is indistinguishable from the
    # old point on summed return and LOSES the W1 acceptance window 3 times out
    # of 3. Chosen for headroom and measurability. See
    # regime/CANDIDATE_STATUS.md and regime/GUARD_ABLATION.md.
    buy_params = {
        "entry_adx_threshold": -1.0,  # open
        "entry_atr_pct": -1.0,  # open
        "entry_bb_width_threshold": -1.0,  # open
        "entry_close_norm_threshold": 1.1,  # open
        "entry_enable_guards": True,  # keeps `volume > 0` live
        "entry_guard_threshold": 1.1,  # open
        "entry_rvol_threshold": -1.0,  # open
        "prediction_threshold": 0.5,  # value loaded from strategy
    }

    # Sell parameters:
    sell_params = {
        "cexit_max_days": 21,
        # 2026-09-06: 0.032 -> 0.022. Swept on the REAL NNNC model (label 0.020,
        # guards open, thr 0.80) at two seeds. EVERY alternative beat the 0.024
        # NNNC setting on summed return at both seeds -- 0.024 ranked LAST of the
        # four values tried -- and 0.020/0.022 beat it on the acceptance window
        # 4 times out of 4, while 0.028 lost at both seeds. 0.022 minimises
        # worst-case regret between 0.020 and 0.022 and is the more consistent of
        # the two on W1.
        #
        # NOTE the oracle could NOT answer this: at the ceiling, moving the
        # take-profit was worth +0.91pp, because a perfect model only enters bars
        # that reach the target anyway. A real model enters bars that do not, and
        # an earlier exit banks them before they reverse. The gate bounds what a
        # LABEL can pay, not how the EXIT should harvest it.
        #
        # Effect sizes (0.5-5.6pp) sit inside the 6.38pp seed spread
        # individually; the signal is the CONSISTENCY of the ordering, not any
        # single figure. See regime/LABEL_EXIT_ALIGNMENT.md.
        "cexit_take_profit": 0.022,
        "enable_exit_signal": True,
        "exit_close_norm_threshold": 0.6,
        "exit_guard_threshold": -0.2,
        # 2026-09-06: was a HARDCODED `rvol > 2.0` literal in
        # populate_exit_trend, unreachable by any setting. Now a parameter,
        # because the rvol optimum is known to differ by family (NNNC 2.0,
        # NNMT 0.25). Default 2.0 preserves the previous behaviour exactly.
        "exit_rvol_threshold": 2.0,
        "stoploss_grace_hours": 2.6,
        "stoploss_grace_level": -0.17,
        "cexit_enable_profit_checks": True,  # value loaded from strategy
    }

    # Trailing stop: DISABLED framework-wide (2026-09-02).
    #
    # freqtrade's ft_stoploss_adjust runs the native trailing block IN ADDITION
    # to custom_stoploss, not instead of it, and `trailing_only_offset_is_reached`
    # is False by default here - so the guard never skips and the block ran on
    # every candle of every trade, raising the stop above the configured
    # `stoploss`. Every family in this repo sets `use_custom_stoploss = True`,
    # so every family had the double-application.
    #
    # This completes a fix that was already half-made: see the comment in
    # custom_stoploss below recording that returning self.stoploss there
    # "created an implicit trailing stop even with trailing_stop=False" and
    # caught a large fraction of winners. That half was fixed with `return 1.0`;
    # the native trailing stop was left on, doing the same thing from the other
    # side.
    #
    # Measured on NNNC_MLX across three pinned windows, no retrain:
    #   summed wallet 11.43% -> 15.95% (+4.53pp), better in ALL THREE windows on
    #   return, profit factor, Calmar and win rate. Exit mix: stops 83 -> 40,
    #   take-profit 42 -> 73 at a higher mean - trades formerly cut on noise now
    #   survive to reach their targets.
    #
    # trailing_stop_positive / _offset are left in place but are unreachable
    # while trailing_stop is False. Note they were ALSO unreachable before, since
    # they require profit > offset (0.03) and minimal_roi exits at 0.03 - which
    # is why sweeping trailing_stop_positive gave byte-identical backtests.
    #
    # A strategy family wanting the old behaviour can set trailing_stop = True
    # on its own class, but should measure first: this was validated on NNNC_MLX
    # only, and the per-family check for NNMT is recorded in
    # regime/exit/README.md.
    trailing_stop = False
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

    # Stop-loss GRACE window. For the first ``stoploss_grace_hours`` of a trade
    # the stop is held wide (``stoploss_grace_level``) so entry-noise doesn't
    # stop the trade out; once the window expires it tightens to the normal
    # stop (ATR-adaptive if enabled, else ``self.stoploss``). This restores the
    # "don't stop out immediately" effect the confirm_trade_exit min-hold used
    # to provide as a side effect — but SAFELY: the stop still fires if hit,
    # it's just wider early, rather than the exit being rejected outright.
    # freqtrade only lets a stop tighten over a trade's life, so the wide grace
    # stop is set at entry (after_fill) and the tighten happens once the window
    # passes; the tightened stop is anchored to the entry price so it does not
    # trail winners. Disabled by default (0.0) — behaviour is unchanged until a
    # subclass sets ``stoploss_grace_hours`` > 0.
    # Exposed as hyperopt params (sell space) so the window + level can be
    # tuned alongside the roi/trailing exits. Default OFF (0.0 hours);
    # NNNC/NNMT override the default to 3.0h. The custom_stoploss reads them
    # robustly, so a subclass may also set them as plain floats.
    stoploss_grace_hours = DecimalParameter(
        0.0, 6.0, default=0.0, decimals=1, space="sell", load=True, optimize=True
    )
    stoploss_grace_level = DecimalParameter(
        -0.30, -0.05, default=-0.15, decimals=2, space="sell", load=True, optimize=True
    )

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

    # NOTE: every DecimalParameter `default=` below MUST match the value in
    # `buy_params` above. They had drifted apart (guard -0.7 vs 0.3, rvol 2.0 vs
    # 0.5, adx 50 vs 38, bb_width 0.015 vs 0.06, atr_pct 0.010 vs 0.006), which
    # is invisible in normal use because buy_params shadows the declaration --
    # and misleading when reading the file to find out what a parameter's
    # default actually is. Aligned 2026-09-02. If you change one, change both.
    # guard_metric = 2*(RMI-50)/100 with RMI in [0,100], so it is MATHEMATICALLY bounded
    # to [-1, +1] (measured over 140k bars, 4 pairs, W1: min -1.000, max +1.000). The old
    # upper bound of 5.0 made 80% of the search space a degenerate plateau — every value
    # >= 1.0 is identically "fully open", so hyperopt spent most of its budget resampling
    # the same behaviour. 1.1 is the in-range "off" setting.
    entry_guard_threshold = DecimalParameter(
        -1.0, 1.1, default=1.1, decimals=2, space="buy", load=True, optimize=opt_base_params
    )

    # close_norm is bounded to [-1, +1] (measured: min -0.999, max +0.999). The old lower
    # bound of -0.5 made the entire bottom HALF of the range unreachable, so hyperopt could
    # never test "enter only when close is deeply below its normalised mean" — exactly the
    # selective-entry region worth exploring. Upper bound 5.0 was the same degenerate
    # plateau as guard_metric.
    entry_close_norm_threshold = DecimalParameter(
        -1.0, 1.1, default=1.1, decimals=2, space="buy", load=True, optimize=opt_base_params
    )

    # ADX is defined on [0, 100] and reaches 100.0 in the data (measured max 100.000);
    # the old upper bound of 90 made the strongest-trend decile unreachable.
    entry_adx_threshold = DecimalParameter(
        -1.0, 100.0, default=-1.0, decimals=0, space="buy", load=True, optimize=opt_base_params
    )

    # Measured over W1: median 0.021, p99 0.135, max 0.529. The old upper bound of 0.100
    # sat BELOW the 99th percentile, so no high-volatility entry filter was reachable.
    entry_bb_width_threshold = DecimalParameter(
        -1.000, 0.550, default=-1.0, decimals=3, space="buy", load=True, optimize=opt_base_params,
    )

    # decimals=2: at 1dp this parameter silently rounded its own configured
    # values, so NNMT's 0.25 executed as 0.2 and a sweep "of 0.25" never tested
    # it. Every value configured anywhere in the repo is 1dp, so widening the
    # resolution changes NO current behaviour -- it only stops the next sweep
    # in the 0.2-0.3 range from being quietly quantised away.
    # Measured over W1: median 0.232, p99 4.802, max 5.500 — the old 5.0 cap truncated the
    # top of the observed range.
    entry_rvol_threshold = DecimalParameter(
        -1.0, 5.5, default=-1.0, decimals=2, space="buy", load=True, optimize=opt_base_params
    )

    # Gates atr_pct_roll (NOT atr_pct). Measured over W1: median 0.0034, p99 0.0210,
    # max 0.1806 — the old 0.060 cap cut off the entire high-volatility tail.
    entry_atr_pct = DecimalParameter(
        -1.000,
        0.200,
        default=-1.0,
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

    # Exit-side relative-volume gate. Lower bound is -1.0 so the guard can be
    # NEUTRALISED (rvol is non-negative), matching how the entry-side guards are
    # opened. Default 2.0 = the value that was previously hardcoded.
    # decimals=2 for the same reason as entry_rvol_threshold. Every configured
    # value is 2.0, so this changes no current behaviour.
    exit_rvol_threshold = DecimalParameter(
        -1.0, 5.0, default=2.0, decimals=2, space="sell", load=True, optimize=opt_base_params
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

    def warn_if_gate_is_inert(self) -> None:
        """Warn when entry_enable_guards is True but every guard is neutralised.

        The class-level defaults in BaseStrategy (and NNNCStrategy) leave every quality guard
        OPEN; the tuned values live in <Strategy>.json, which is a FILE and is therefore NOT
        inherited by a subclass. So a strategy that correctly merges its parent's buy_params can
        still run completely ungated, with `entry_enable_guards = True` making it look gated.

        This is not hypothetical. NNNC_DDPM_MLX ran ungated for its entire history: neutralising
        all six guards produced a BYTE-IDENTICAL trade set, because there was nothing to
        neutralise. Every DDPM-vs-NNNC comparison made on that basis compared a gated strategy
        against an ungated one.

        Reports only; changes nothing.
        """
        try:
            enabled = bool(getattr(self.entry_enable_guards, "value",
                                   self.entry_enable_guards))
        except Exception:
            return
        if not enabled:
            return                      # deliberately off — the operator knows

        # (parameter, comparison, sentinel meaning "wide open")
        checks = [
            ("entry_adx_threshold", "le", 0.0),
            ("entry_atr_pct", "le", 0.0),
            ("entry_bb_width_threshold", "le", 0.0),
            ("entry_close_norm_threshold", "ge", 5.0),
            ("entry_guard_threshold", "ge", 5.0),
            ("entry_rvol_threshold", "le", 0.0),
        ]
        inert = []
        for name, op, sentinel in checks:
            attr = getattr(self, name, None)
            if attr is None:
                continue
            try:
                v = float(getattr(attr, "value", attr))
            except (TypeError, ValueError):
                continue
            if (op == "le" and v <= sentinel) or (op == "ge" and v >= sentinel):
                inert.append(f"{name}={v:g}")

        if not inert:
            return
        if len(inert) == len(checks):
            print("    " + "=" * 72)
            print("    *** WARNING: entry_enable_guards is True but EVERY guard is neutralised.")
            print("    *** This strategy is running COMPLETELY UNGATED.")
            print(f"    *** {', '.join(inert)}")
            print("    *** The tuned gate lives in <Strategy>.json, which is NOT inherited by a")
            print(f"    *** subclass. Give {self.__class__.__name__} its own .json, or its results")
            print("    *** are not comparable with any gated strategy.")
            print("    " + "=" * 72)
        else:
            print(f"    NOTE: {len(inert)} of {len(checks)} entry guards are neutralised "
                  f"({', '.join(inert)}) while entry_enable_guards is True.")

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

        # MANDATORY -- deliberately OUTSIDE the entry_enable_guards block
        # (2026-09-08). `volume > 0` is a data-sanity check and rvol is a
        # TRADEABILITY constraint: you cannot fill into a candle that did not
        # trade, however confident the model is. Neither is a quality filter, so
        # neither belongs behind a flag that exists to disable quality filters.
        #
        # This also fixes a long-standing trap. `--guards off` previously
        # removed the volume sanity check along with the quality guards, which
        # is why every guards-off arm in this repository had to be expressed by
        # NEUTRALISING six thresholds instead. That workaround is no longer
        # required for volume, though it remains the honest way to open the
        # QUALITY guards.
        #
        # rvol stays a hyperparameter: set entry_rvol_threshold to -1.0 to make
        # it permissive. It is no longer possible to remove it entirely by
        # accident.
        # Checked HERE, not in bot_start: this is where the gate values are actually read, so
        # whatever is live at this point is what filters the trades. bot_start ran too early --
        # it reported class defaults while the file values were still being applied, which made
        # the warning fire on a correctly-gated strategy.
        if not getattr(self, "_gate_inertia_checked", False):
            self._gate_inertia_checked = True
            self.warn_if_gate_is_inert()

        conditions.append(dataframe["volume"] > 0.0)
        conditions.append(dataframe["rvol"] > self.entry_rvol_threshold.value)

        # Common guard conditions -- QUALITY filters only
        if self.entry_enable_guards.value:
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

        # NOTE: this block is gated on entry_enable_guards, so the ENTRY flag
        # controls the EXIT guards. Left as-is deliberately -- changing it would
        # alter behaviour for every strategy; recorded as a separate open item in
        # regime/CANDIDATE_STATUS.md.
        if self.entry_enable_guards.value:
            conditions.append(dataframe["rvol"] > self.exit_rvol_threshold.value)

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

    def _resolve_normal_stop(self, pair: str) -> float:
        """Normal (post-grace) stop level relative to entry: ATR-adaptive if
        enabled, else the static ``self.stoploss``. Mirrors the after_fill ATR
        logic in ``custom_stoploss``."""
        if self.use_atr_adaptive_stoploss:
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
        return self.stoploss

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
        # First-hour stop-loss GRACE (opt-in via stoploss_grace_hours > 0).
        # Hold a wide stop for the initial window so entry noise doesn't stop
        # the trade out, then tighten to the normal stop. The stop still fires
        # if the wide level is hit (safe), unlike the old min-hold that rejected
        # the exit outright.
        _gh = getattr(self, "stoploss_grace_hours", 0.0)
        grace_hours = float(getattr(_gh, "value", _gh) or 0.0)  # Parameter or float
        if grace_hours > 0.0:
            _gl = getattr(self, "stoploss_grace_level", -0.15)
            grace_level = float(getattr(_gl, "value", _gl))
            age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            if age_h < grace_hours:
                # Wide stop through the window. freqtrade lets the INITIAL stop
                # (set at after_fill) be arbitrarily wide; leave it unchanged
                # for the rest of the window.
                return grace_level if after_fill else 1.0
            # Window expired -> tighten to the normal stop, anchored to the
            # entry price so the tightened stop is fixed (no trailing on
            # winners). freqtrade only tightens, so this applies once.
            normal = self._resolve_normal_stop(pair)
            if trade.open_rate and current_rate and current_rate > 0:
                return (trade.open_rate * (1.0 + normal) / current_rate) - 1.0
            return normal

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

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Reduce (rather than cancel) entries that would dominate a thin candle.

        ``confirm_trade_entry`` rejects any order whose quote size exceeds
        1 / QUOTE_VOLUME_HEADROOM_MULT of the candle's traded quote volume
        (the ≤10% market-impact target). Here we instead cap the stake at the
        largest size that stays within that headroom, so a marginal-liquidity
        entry still happens at a fillable size instead of being dropped.

        The absolute floors are left to the existing gate: if the reduced
        stake falls below ``min_stake`` (or the candle's quote volume is under
        MIN_QUOTE_VOLUME), ``confirm_trade_entry`` still cancels — so the
        phantom-fill protection is preserved by construction, since any order
        that does enter is ≤10% of the candle by definition.
        """
        if self.dp.runmode.value in ("plot", "other"):
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        last_candle = dataframe.iloc[-1].squeeze()
        quote_volume = last_candle["volume"] * last_candle["close"]
        fillable_stake = quote_volume / self.QUOTE_VOLUME_HEADROOM_MULT
        return min(proposed_stake, fillable_stake)

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

        # Reject exit if trade has been open for less than 1 hour. This paces
        # only DISCRETIONARY exits (exit_signal / roi). Risk-management and
        # forced exits must always be honoured — never hold a position past
        # its stop just because the trade is young.
        if exit_reason not in [
            "force_exit",
            "emergency_exit",
            "stop_loss",
            "trailing_stop_loss",
            "stoploss_on_exchange",
            "liquidation",
        ]:
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
