"""
DebugTradingType - strategy used for comparing different methods of creating the trading indicator

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
from pandas import DataFrame
import numpy as np
from functools import reduce
import talib.abstract as ta

# Local imports (must come after sys.path manipulation)
from NNMT.NNMTStrategy import NNMTStrategy, TradingAction, MarketRegime  # noqa: E402
from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402
from Framework.TrainingSignals import available_methods  # noqa: E402

# Third-party imports
from freqtrade.strategy import (  # noqa: E402
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)


class DebugTradingType(NNMTStrategy):
    """
    Simple strategy that just uses the lookahead buy/sell signals
    """

    # Overrides for triple-barrier label investigation. The threshold values
    # match the Framework.TrainingConfig defaults but are re-declared
    # explicitly because this debug strategy treats them as the experiment's
    # independent variables. TRAINING_TYPE=1 is the intentional override
    # (default in TrainingConfig is the indicator-combo type).
    MIN_BUY_GAIN_THRESHOLD = 0.008
    MIN_SELL_LOSS_THRESHOLD = 0.008
    TRAINING_TYPE = 1
    PEAK_WINDOW = 6
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
                # "predict_buy": {"color": "green"},
                # "predict_sell": {"color": "red"},
                # "trading_0": {"color": "green"},
                "trading_1": {"color": "purple"},
                # "trading_2": {"color": "brown"},
                # "trading_3": {"color": "blue"},
                # "trading_4": {"color": "orange"},
                # "trading_5": {"color": "red"},
                # "trading_6": {"color": "cyan"},
                # "trading_7": {"color": "magenta"},
                # "trading_8": {"color": "yellow"},
                # "trading_9": {"color": "pink"},
                # "trading_10": {"color": "teal"},
                # "trading_11": {"color": "olive"},
                # "trading_12": {"color": "navy"},
                # "trading_13": {"color": "maroon"},
                # "trading_14": {"color": "lime"},
                # "trading_15": {"color": "gold"},
                "trading_16": {"color": "orange"},
                # "trading_17": {"color": "black"},
                # "trading_18": {"color": "lightseagreen"},
                "trading_19": {"color": "purple"},
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
        18,
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

    buy_predictions = None
    sell_predictions = None
    predictions_updated = False

    def update_predictions(self, dataframe: DataFrame):
        """Update the predictions based on the training signals"""

        if not self.predictions_updated:
            self.predictions_updated = True

            self.MIN_BUY_GAIN_THRESHOLD = self.min_buy_gain_threshold.value
            self.TRAINING_TYPE = self.training_type.value
            self.PROFIT_TAKE_THRESHOLD = self.profit_take_threshold.value
            self.PROFIT_STOP_LOSS_THRESHOLD = self.profit_stop_loss_threshold.value
            self.MIN_SELL_LOSS_THRESHOLD = self.min_sell_loss_threshold.value

            # these are just to be able to visualise data
            profit_targets = self.get_profit_target(dataframe)
            regime_targets = self.get_market_target(dataframe)
            momentum_targets = self.get_momentum_target(dataframe)
            risk_targets = self.get_risk_target(dataframe)
            flow_targets = self.get_flow_target(dataframe)

            trading_methods = available_methods()
            for i in range(len(trading_methods)):
                self.TRAINING_TYPE = i
                method = trading_methods[i]
                print(f"     trading_method[{i}]: {method}")
                dataframe[f"trading_{i}"] = self.get_trading_classes(
                    dataframe,
                    profit_targets,
                    regime_targets,
                    momentum_targets,
                    risk_targets,
                    flow_targets,
                )

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
