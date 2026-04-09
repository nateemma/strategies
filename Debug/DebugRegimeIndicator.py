"""
DebugRegimeIndicator - strategy used for comparing different methods of creating the regime indicator

    It turns out that this is difficult to predict, so tgis helps visualise and debug different approaches
"""

import sys
from pathlib import Path

# set path such that python can find other directories
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# Standard library imports
# pylint: disable=import-outside-toplevel
import pandas as pd
from pandas import DataFrame
import numpy as np
from functools import reduce
import talib.abstract as ta

# Local imports (must come after sys.path manipulation)
from NNMTStrategy import NNMTStrategy, TradingAction, MarketRegime  # noqa: E402
from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402

# Third-party imports
from freqtrade.strategy import (  # noqa: E402
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)


class DebugRegimeIndicator(NNMTStrategy):
    """
    Simple strategy that just uses the lookahead buy/sell signals
    """

    # re-declare class variables so that we can override them later
    MIN_BUY_GAIN_THRESHOLD = 0.008  # minimum gain for buy signals
    MIN_SELL_LOSS_THRESHOLD = 0.008  # minimum loss for sell signals
    PEAK_WINDOW = 36
    augment_training_data = False
    aggregate_pairs = False

    # --------------------------------

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
            # "smoothed_close": {"color": "lightsalmon"},
        },
        "subplots": {
            "Diff": {
                "predict_buy": {"color": "green"},
                "predict_sell": {"color": "red"},
                "regime_0": {"color": "green"},
                "regime_1": {"color": "purple"},
                "regime_2": {"color": "brown"},
                "regime_3": {"color": "orange"},
                "regime_4": {"color": "pink"},
            },
        },
    }

    # --------------------------------

    # override NNStrategy hyperopt params (mostly to disable optimization for now)
    opt_framework_params = False
    prediction_threshold = DecimalParameter(
        0.5,
        0.9,
        default=0.5,
        decimals=1,
        space="buy",
        load=True,
        optimize=opt_framework_params,
    )

    enable_exit_signal = CategoricalParameter(
        [True, False],
        default=False,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    entry_guard_threshold = DecimalParameter(
        -0.9,
        -0.0,
        default=-0.7,
        decimals=1,
        space="buy",
        load=True,
        optimize=opt_framework_params,
    )

    entry_close_norm_threshold = DecimalParameter(
        -0.5,
        0.0,
        default=0.0,
        decimals=1,
        space="buy",
        load=True,
        optimize=opt_framework_params,
    )

    entry_adx_threshold = DecimalParameter(
        50.0,
        80.0,
        default=50.0,
        decimals=0,
        space="buy",
        load=True,
        optimize=opt_framework_params,
    )

    entry_bb_width_threshold = DecimalParameter(
        0.01,
        0.08,
        default=0.04,
        decimals=2,
        space="buy",
        load=True,
        optimize=opt_framework_params,
    )

    exit_guard_threshold = DecimalParameter(
        0.0,
        0.9,
        default=0.7,
        decimals=1,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    exit_close_norm_threshold = DecimalParameter(
        0.0,
        1.0,
        default=0.0,
        decimals=1,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    cexit_enable_profit_checks = CategoricalParameter(
        [True, False],
        default=True,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    cexit_take_profit = DecimalParameter(
        0.005,
        0.025,
        default=0.008,
        decimals=3,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    cexit_max_days = IntParameter(
        1,
        30,
        default=21,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    # hyperparams to control buy/sell signals
    opt_train_signals = True
    min_buy_gain_threshold = DecimalParameter(
        0.008,
        0.02,
        default=0.01,
        decimals=3,
        space="buy",
        load=True,
        optimize=opt_train_signals,
    )

    min_sell_loss_threshold = DecimalParameter(
        0.008,
        0.02,
        default=0.01,
        decimals=3,
        space="sell",
        load=True,
        optimize=opt_train_signals,
    )

    training_type = IntParameter(
        0,
        6,
        default=8,
        space="buy",
        load=True,
        optimize=False,
    )

    profit_take_threshold = DecimalParameter(
        0.01,
        0.05,
        default=0.02,
        decimals=2,
        space="buy",
        optimize=True,
        load=True,
    )
    profit_stop_loss_threshold = DecimalParameter(
        0.005,
        0.05,
        default=0.015,
        decimals=3,
        space="buy",
        optimize=True,
        load=True,
    )

    # --------------------------------
    # Regime indicators

    ADX_THRESHOLD = -0.2

    def get_regime_0(self, dataframe: DataFrame) -> np.ndarray:

        # Check that required normalized columns are in include_list
        self.check_columns_included(
            ["close_norm", "adx_scaled", "rsi_scaled", "macd_norm"], "get_market_regime"
        )

        # Get normalized indicators (all pair-agnostic)
        close_norm = dataframe.get("close_norm", pd.Series(np.zeros(len(dataframe))))
        close_norm = pd.Series(close_norm).fillna(0)

        # Get ADX_scaled for trend strength (normalized [-1, 1], pair-agnostic)
        # adx_scaled is normalized: ADX=0 → -1.0, ADX=27.5 → 0.0, ADX=55 → 1.0
        adx_scaled = dataframe.get("adx_scaled", pd.Series(np.zeros(len(dataframe))))
        adx_scaled = pd.Series(adx_scaled).fillna(0)

        # Get RSI_scaled for momentum (normalized [-1, 1], pair-agnostic)
        rsi_scaled = dataframe.get("rsi_scaled", pd.Series(np.zeros(len(dataframe))))
        rsi_scaled = pd.Series(rsi_scaled).fillna(0)

        # Get MACD_norm for momentum confirmation (normalized, pair-agnostic)
        macd_norm = dataframe.get("macd_norm", pd.Series(np.zeros(len(dataframe))))
        macd_norm = pd.Series(macd_norm).fillna(0)

        # Calculate momentum using normalized close_norm (pair-agnostic)
        # close_norm is rolling MinMax normalized to [-1, 1], so differences reflect relative price changes
        momentum_short = close_norm.diff(5)  # 5-period momentum
        momentum_medium = close_norm.diff(10)  # 10-period momentum

        # Initialize regime as Sideways
        regime = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        # Create validity mask
        valid_data = ~(pd.isna(close_norm) | pd.isna(adx_scaled))

        # Use normalized ADX threshold (pair-agnostic)
        # adx_scaled >= -0.2 is equivalent to ADX >= 22 (moderate trend strength for short-term)
        trend_strength = adx_scaled >= self.ADX_THRESHOLD

        # Bull conditions (multiple confirmations using normalized features):
        # 1. Price above recent mean (close_norm > 0) AND rising (momentum positive)
        # 2. Positive momentum (both short and medium term)
        # 3. Strong trend (adx_scaled >= -0.2)
        # 4. RSI not oversold (rsi_scaled > -0.4, equivalent to RSI > 30)
        # 5. MACD positive (macd_norm > 0) for confirmation
        bull_price_position = close_norm > 0  # Price above recent mean
        bull_momentum = (momentum_short > 0) & (momentum_medium > 0)
        bull_rsi = rsi_scaled > -0.4  # Not oversold (RSI > 30 in normalized scale)
        bull_macd = macd_norm > 0  # MACD positive

        bull_condition = (
            bull_price_position
            & bull_momentum
            & trend_strength
            & bull_rsi
            & bull_macd
            & valid_data
        )

        # Bear conditions (inverse of bull):
        # 1. Price below recent mean (close_norm < 0) AND falling (momentum negative)
        # 2. Negative momentum (both short and medium term)
        # 3. Strong trend (adx_scaled >= -0.2)
        # 4. RSI not overbought (rsi_scaled < 0.4, equivalent to RSI < 70)
        # 5. MACD negative (macd_norm < 0) for confirmation
        bear_price_position = close_norm < 0  # Price below recent mean
        bear_momentum = (momentum_short < 0) & (momentum_medium < 0)
        bear_rsi = rsi_scaled < 0.4  # Not overbought (RSI < 70 in normalized scale)
        bear_macd = macd_norm < 0  # MACD negative

        bear_condition = (
            bear_price_position
            & bear_momentum
            & trend_strength
            & bear_rsi
            & bear_macd
            & valid_data
        )

        # Assign regimes
        regime[bull_condition] = MarketRegime.BULL
        regime[bear_condition] = MarketRegime.BEAR
        # Everything else defaults to Sideways

        return regime

    def get_regime_1(self, dataframe: DataFrame) -> np.ndarray:

        regime_1 = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        # 1. Calculate Indicators
        close = dataframe["close"]

        # Short-term trend (20 periods)
        sma_20 = ta.SMA(dataframe['close'], timeperiod=20)

        # Trend strength (14 periods)
        adx = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        # Momentum (14 periods)
        rsi = ta.RSI(dataframe["close"], timeperiod=14)

        regime_1 = np.where((close > sma_20) & (rsi > 50), MarketRegime.BULL, regime_1)
        regime_1 = np.where((close < sma_20) & (rsi < 50), MarketRegime.BEAR, regime_1)
        regime_1 = np.where((adx < 20), MarketRegime.SIDEWAYS, regime_1)
        return regime_1

    def get_regime_2(self, dataframe: DataFrame) -> np.ndarray:

        regime_2 = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        close = dataframe["close"]

        # 1. Calculate the Hilbert Transform Trend Mode
        # This identifies if we are in a 'Trend' (1) or 'Cycle' (0)
        trend_mode = ta.HT_TRENDMODE(dataframe['close'])

        # 2. Calculate a fast EMA for direction
        ema_fast = ta.EMA(dataframe['close'], timeperiod=20)

        # If in Trend mode, check direction relative to EMA
        regime_2 = np.where((trend_mode == 1) & (close > ema_fast), MarketRegime.BULL, regime_2)
        regime_2 = np.where((trend_mode == 1) & (close < ema_fast), MarketRegime.BEAR, regime_2)

        # Trend strength (14 periods)
        adx = ta.ADX(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )

        adxscaled = dataframe["adx_scaled"]
        # regime_2 = np.where((adx < 20), MarketRegime.SIDEWAYS, regime_2)
        regime_2 = np.where((adxscaled < 0.0), MarketRegime.SIDEWAYS, regime_2)

        return regime_2

    def get_regime_3(self, dataframe: DataFrame) -> np.ndarray:

        regime_3 = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        close_norm = dataframe["close_norm"]

        # 1. Calculate the Hilbert Transform Trend Mode
        # This identifies if we are in a 'Trend' (1) or 'Cycle' (0)
        trend_mode = dataframe["trend_mode"]

        # 2. Calculate a fast EMA for direction
        ema_fast_norm = dataframe["ema_fast_norm"]

        # If in Trend mode, check direction relative to EMA
        regime_3 = np.where(
            # (trend_mode == 1) & (close_norm > 0) & (ema_fast_norm > 0),
            (trend_mode == 1) & (close_norm > 0),
            MarketRegime.BULL,
            regime_3,
        )
        regime_3 = np.where(
            # (trend_mode == 1) & (close_norm < 0) & (ema_fast_norm < 0),
            (trend_mode == 1) & (close_norm < 0),
            MarketRegime.BEAR,
            regime_3,
        )

        # Trend strength (14 periods)
        adxscaled = dataframe["adx_scaled"]

        regime_3 = np.where(
            (adxscaled < self.ADX_THRESHOLD), MarketRegime.SIDEWAYS, regime_3
        )

        return regime_3

    def get_regime_4(self, dataframe: DataFrame) -> np.ndarray:

        regime_4 = np.ones(len(dataframe), dtype=int) * MarketRegime.SIDEWAYS

        close_norm = dataframe["close_norm"]
        adxscaled = dataframe["adx_scaled"]
        di_diff_scaled = dataframe["di_diff_scaled"]

        # trend directio

        # 1. Calculate the Hilbert Transform Trend Mode
        # This identifies if we are in a 'Trend' (1) or 'Cycle' (0)
        trend_mode = dataframe["trend_mode"]

        # If in Trend mode, check direction relative to EMA
        regime_4 = np.where(
            (trend_mode == 1) & (di_diff_scaled > 0),
            MarketRegime.BULL,
            regime_4,
        )
        regime_4 = np.where(
            (trend_mode == 1) & (di_diff_scaled < 0),
            MarketRegime.BEAR,
            regime_4,
        )

        # regime_4 = np.where((adxscaled < self.ADX_THRESHOLD), MarketRegime.SIDEWAYS, regime_4)

        return regime_4

    # --------------------------------

    buy_predictions = None
    sell_predictions = None

    def update_predictions(self, dataframe: DataFrame):
        """Update the predictions based on the training signals"""

        self.MIN_BUY_GAIN_THRESHOLD = self.min_buy_gain_threshold.value
        self.TRAINING_TYPE = self.training_type.value
        self.PROFIT_TAKE_THRESHOLD = self.profit_take_threshold.value
        self.PROFIT_STOP_LOSS_THRESHOLD = self.profit_stop_loss_threshold.value
        self.MIN_SELL_LOSS_THRESHOLD = self.min_sell_loss_threshold.value

        # these are just to be able to visualise data
        dataframe["regime_0"] = self.get_regime_0(dataframe)
        dataframe["regime_1"] = self.get_regime_1(dataframe)
        dataframe["regime_2"] = self.get_regime_2(dataframe)
        dataframe["regime_3"] = self.get_regime_3(dataframe)
        dataframe["regime_4"] = self.get_regime_4(dataframe)
        self.print_distribution_compact("  Market Regime 0", dataframe["regime_0"])
        self.print_distribution_compact("  Market Regime 1", dataframe["regime_1"])
        self.print_distribution_compact("  Market Regime 2", dataframe["regime_2"])
        self.print_distribution_compact("  Market Regime 3", dataframe["regime_3"])
        self.print_distribution_compact("  Market Regime 4", dataframe["regime_4"])

        # everything else is done in populate_entry_trend and populate_exit_trend
        # this alows us to use hyperopt to find parameters for the training signals
        # (which are normlly run within populate_indicators)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # set parameters within NNMTStrategy based on hyperopt params here
        # cannot hyperopt in main strategy
        self.MIN_BUY_GAIN_THRESHOLD = self.min_buy_gain_threshold.value
        self.TRAIN_BUY_GUARD_THRESHOLD = self.min_buy_gain_threshold.value
        self.TRAINING_TYPE = self.training_type.value
        self.PROFIT_TAKE_THRESHOLD = self.profit_take_threshold.value
        self.PROFIT_STOP_LOSS_THRESHOLD = self.profit_stop_loss_threshold.value
        self.MIN_SELL_LOSS_THRESHOLD = self.min_sell_loss_threshold.value
        self.TRAIN_SELL_GUARD_THRESHOLD = self.min_sell_loss_threshold.value

        # add the indicators to the dataframe

        self.iteration_init()

        dataframe = self.dataframePopulator.add_indicators(
            dataframe, dataset_type=DatasetType.MINIMAL
        )
        self.add_additional_indicators(dataframe)

        dataframe = self.update_predictions(dataframe)

        # everything else is done in populate_entry_trend and populate_exit_trend
        # this alows us to use hyperopt to find parameters for the training signals
        # (which are normlly run within populate_indicators)

        return dataframe

    # --------------------------------
    def emulate_buy_signals(self, dataframe: DataFrame):
        """Emulate the buy signals based on the training signals"""

        dataframe = self.update_predictions(dataframe)
        return self.buy_predictions

    def emulate_sell_signals(self, dataframe: DataFrame):
        """Emulate the sell signals based on the training signals"""

        dataframe = self.update_predictions(dataframe)
        return self.sell_predictions

    # --------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Common entry trend population - calls strategy-specific method for custom conditions"""
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""

        # DEBUG: disable additional conditions for now
        # Add common conditions
        quote_volume = dataframe["volume"] * dataframe["close"]
        conditions.append(quote_volume > self.MIN_QUOTE_VOLUME)
        # conditions.append(dataframe["volume"] > 100)

        # common guard conditions
        conditions.append(
            dataframe["close_norm"] < self.entry_close_norm_threshold.value
        )
        conditions.append(dataframe["guard_metric"] < self.entry_guard_threshold.value)
        conditions.append(dataframe["adx"] > self.entry_adx_threshold.value)
        conditions.append(dataframe["bb_width"] > self.entry_bb_width_threshold.value)

        # get the lookahead buy/sell signals
        # self.PEAK_WINDOW = self.peak_window.value

        buys = self.emulate_buy_signals(dataframe)
        dataframe["predict_buy"] = np.where(buys == 1, 1, 0)

        # ad buy/sell signals to the dataframe
        conditions.append(dataframe["predict_buy"] == 1)

        # Apply conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1
        else:
            dataframe["enter_long"] = 0

        return dataframe

    # --------------------------------

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Common exit trend population - calls strategy-specific method for custom conditions"""
        conditions = []
        dataframe.loc[:, "exit_tag"] = ""
        dataframe["exit_long"] = 0

        if not self.enable_exit_signal.value:
            return dataframe

        # Add common conditions
        conditions.append(dataframe["volume"] > 1)

        # common guard conditions
        conditions.append(
            dataframe["close_norm"] > self.exit_close_norm_threshold.value
        )
        conditions.append(dataframe["guard_metric"] > self.exit_guard_threshold.value)

        # get the lookahead buy/sell signals
        sells = self.emulate_sell_signals(dataframe)
        dataframe["predict_sell"] = np.where(sells == 1, 1, 0)

        conditions.append(dataframe["predict_sell"] == 1)

        # Apply conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
        else:
            dataframe["exit_long"] = 0

        return dataframe
