"""
DebugNNMTStrategy - simple strategy that just uses the lookahead buy/sell signals, and
leverages all of the other features of NNStrategy (hyperparams, guards, custom_exit etc)

Yes, this is using lookahead, but the intent is provide a reference for how a strategy
with perfect signal selection would perform. Very useful for debugging the training
signal logic

I mostly use this to hyperopt the framework parameters (guards, custom_exit, etc)
"""

import sys
from pathlib import Path


# set path such that python can find other directories
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy
from pandas import DataFrame
import numpy as np


from utils.DataframePopulator import DataframePopulator, DatasetType
from functools import reduce
from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType


class DebugNNMTStrategy(NNMTStrategy):
    """
    Simple strategy that just uses the lookahead buy/sell signals
    """

    # re-declare class variables so that we can override them later
    MIN_BUY_GAIN_THRESHOLD = 0.009
    MIN_SELL_LOSS_THRESHOLD = 0.01
    TRAINING_TYPE = 16
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
                "profit": {"color": "blue"},
                "trading": {"color": "orange"},
                "regime": {"color": "purple"},
                "risk": {"color": "brown"},
                "momentum": {"color": "gray"},
                "flow": {"color": "pink"},
            },
        },
    }

    # --------------------------------
    
    # Buy hyperspace params:
    buy_params = {
        "apply_task_filters": False,
        "entry_adx_threshold": 60.0,
        "entry_atr_pct": 0.003,
        "entry_bb_width_threshold": 0.07,
        "entry_close_norm_threshold": -0.4,
        "entry_enable_guards": True,
        "entry_guard_threshold": -0.0,
        "min_consecutive_buys": 2,
        "prediction_threshold": 0.7,
        "bias_profit_high": 0.08,  # value loaded from strategy
        "bias_profit_low": 0.09,  # value loaded from strategy
        "bias_trading_buy": 0.05,  # value loaded from strategy
        "bias_trading_sell": 0.03,  # value loaded from strategy
        "min_buy_gain_threshold": 0.01,  # value loaded from strategy
        "training_type": 16,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 9,
        "cexit_take_profit": 0.023,
        "enable_exit_signal": True,
        "exit_close_norm_threshold": 0.5,
        "exit_guard_threshold": 0.9,
        "min_sell_loss_threshold": 0.01,  # value loaded from strategy
    }

    # override NNStrategy hyperopt params (mostly to disable optimization for now)
    opt_framework_params = True
    opt_train_signals = False
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
        16,
        default=9,
        space="buy",
        load=True,
        optimize=opt_train_signals,
    )

    # --------------------------------

    buy_predictions = None
    sell_predictions = None

    def update_predictions(self, dataframe: DataFrame):
        """Update the predictions based on the training signals"""

        self.MIN_BUY_GAIN_THRESHOLD = self.min_buy_gain_threshold.value
        self.TRAINING_TYPE = self.training_type.value
        self.MIN_SELL_LOSS_THRESHOLD = self.min_sell_loss_threshold.value

        # these are just to be able to visualise data
        profit_targets = self.get_profit_target(dataframe)
        regime_targets = self.get_market_target(dataframe)
        momentum_targets = self.get_momentum_target(dataframe)
        risk_targets = self.get_risk_target(dataframe)
        flow_targets = self.get_flow_target(dataframe)
        trading_targets = self.get_trading_classes(
            dataframe,
            profit_targets,
            regime_targets,
            momentum_targets,
            risk_targets,
            flow_targets,
        )
        dataframe["profit"] = profit_targets
        dataframe["trading"] = trading_targets
        dataframe["regime"] = regime_targets
        dataframe["risk"] = risk_targets
        dataframe["momentum"] = momentum_targets
        dataframe["flow"] = flow_targets

        # save buy/sell predictions
        predictions = {}
        predictions["trading"] = trading_targets
        predictions["profit"] = profit_targets
        predictions["regime"] = regime_targets
        predictions["risk"] = risk_targets
        predictions["momentum"] = momentum_targets
        predictions["flow"] = flow_targets
        dataframe = self.process_predictions(dataframe, predictions)

        self.buy_predictions = dataframe["predict_buy"]
        self.sell_predictions = dataframe["predict_sell"]

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
        if (
            self.buy_predictions is None
            or len(self.buy_predictions) != len(dataframe)
        ):
            dataframe = self.update_predictions(dataframe)
        return self.buy_predictions

    def emulate_sell_signals(self, dataframe: DataFrame):
        """Emulate the sell signals based on the training signals"""
        if (
            self.sell_predictions is None
            or len(self.sell_predictions) != len(dataframe)
        ):
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
        # quote_volume_avg = quote_volume.rolling(window=self.PEAK_WINDOW, min_periods=1).mean()
        # conditions.append(quote_volume_avg > self.MIN_QUOTE_VOLUME)

        conditions.append(quote_volume > self.MIN_QUOTE_VOLUME)
        # conditions.append(dataframe["volume"] > 100)

        # DEBUG: disable additional conditions for now
        enable_checks = False  # only disable for debugging
        if enable_checks:

            # common guard conditions

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
        # conditions.append(dataframe["volume"] > 1)
        quote_volume = dataframe["volume"] * dataframe["close"]
        # quote_volume_avg = quote_volume.rolling(
        #     window=self.PEAK_WINDOW, min_periods=1
        # ).mean()
        # conditions.append(quote_volume_avg > self.MIN_QUOTE_VOLUME)
        conditions.append(quote_volume > self.MIN_QUOTE_VOLUME)

        # # common guard conditions
        # conditions.append(dataframe["close_norm"] > self.exit_close_norm_threshold.value)
        # conditions.append(dataframe["guard_metric"] > self.exit_guard_threshold.value)

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
