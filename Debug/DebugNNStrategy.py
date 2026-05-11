"""
DebugNNStrategy - simple strategy that just uses the lookahead buy/sell signals, and
leverages all of the other features of NNStrategy (hyperparams, guards, custom_exit etc)

Yes, this is using lookahead, but the intent is provide a reference for how a strategy
with perfect signal selection would perform. Very useful for debugging the training
signal logic

I mostly use this to hyperopt and/or visualise the framework parameters (guards,
custom_exit, etc) and training signals
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

# Local imports (must come after sys.path manipulation)
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.DataframePopulator import DataframePopulator, DatasetType


# Third-party imports
from freqtrade.strategy import (  # noqa: E402
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)


class DebugNNStrategy(BaseNNStrategy):
    """
    Simple strategy that just uses the lookahead buy/sell signals
    """

    # re-declare class variables so that we can override them later
    MIN_BUY_GAIN_THRESHOLD = 0.009  # minimum gain for buy signals
    MIN_SELL_LOSS_THRESHOLD = 0.011  # minimum loss for sell signals
    TRAINING_TYPE = 16
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
                "predict_buy": {"color": "green"},
                "predict_sell": {"color": "red"},
                "%train_buy": {"color": "purple"},
                "%train_sell": {"color": "orange"},
                # "trading": {"color": "orange"},
                # # "bb_width": {"color": "purple"},
                # "close_norm": {"color": "blue"},
                # "atr_norm": {"color": "yellow"},
                # "gain_norm": {"color": "blue"},
                "rsi_scaled": {"color": "orange"},
                "mfi_scaled": {"color": "purple"},
                "ema_fast_norm": {"color": "green"},
                "fastk_scaled": {"color": "yellow"},
                "di_diff_scaled": {"color": "brown"},
                # "bb_width": {"color": "brown"},
                # "momentum": {"color": "orange"},
                # "regime": {"color": "brown"},
                # "risk": {"color": "green"},
                # "flow": {"color": "purple"},
            },
        },
    }

    # --------------------------------
    # Buy hyperspace params:
    buy_params = {
        "entry_adx_threshold": 20.0,
        "entry_atr_pct": 0.001,
        "entry_bb_width_threshold": 0.0,
        "entry_close_norm_threshold": 0.0,
        "entry_enable_guards": True,
        "entry_guard_threshold": -0.0,
        "entry_rvol_threshold": 1.0,
        "prediction_threshold": 0.3,
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 3,
        "cexit_take_profit": 0.013,
        "enable_exit_signal": True,
        "exit_close_norm_threshold": 0.0,
        "exit_guard_threshold": 0.0,
    }

    # override NNStrategy hyperopt params (mostly to disable optimization for now)
    opt_framework_params = False
    opt_train_signals = True

    # don't optimise this here because predictions are perfect
    prediction_threshold = DecimalParameter(
        0.3,
        0.7,
        default=0.4,
        decimals=1,
        space="buy",
        load=True,
        optimize=False,
    )

    enable_exit_signal = CategoricalParameter(
        [True, False],
        default=False,
        space="sell",
        load=True,
        optimize=opt_framework_params,
    )

    entry_enable_guards = CategoricalParameter(
        [True, False],
        default=True,
        space="buy",
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

    entry_atr_pct = DecimalParameter(
        0.000,
        0.010,
        default=0.001,
        decimals=3,
        space="buy",
        load=True,
        optimize=opt_framework_params,
    )

    entry_rvol_threshold = DecimalParameter(
        1.0,
        4.0,
        default=1.0,
        decimals=1,
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
        0.002,
        0.01,
        default=0.03,
        decimals=3,
        space="buy",
        load=True,
        optimize=opt_train_signals,
    )

    min_sell_loss_threshold = DecimalParameter(
        0.002,
        0.01,
        default=0.03,
        decimals=3,
        space="sell",
        load=True,
        optimize=opt_train_signals,
    )

    # NOTE: only a few options appear to be trainable
    training_type = IntParameter(
        0,
        19,
        default=16,
        space="buy",
        load=True,
        optimize=False,
    )

    # --------------------------------

    buy_predictions = None
    sell_predictions = None

    def update_predictions(self, dataframe: DataFrame):
        """Update the predictions based on the training signals"""
        # Note: These variables are already set in populate_indicators,
        # but we set them here again to ensure they're current before get_training_labels
        self.MIN_BUY_GAIN_THRESHOLD = self.min_buy_gain_threshold.value
        self.TRAINING_TYPE = self.training_type.value
        self.MIN_SELL_LOSS_THRESHOLD = self.min_sell_loss_threshold.value

        # # Debug: verify values are set correctly
        # if hasattr(self, 'dbg_verbose') and self.dbg_verbose:
        # print(f"DEBUG update_predictions: TRAINING_TYPE={self.TRAINING_TYPE}, "
        #         f"MIN_BUY_GAIN_THRESHOLD={self.MIN_BUY_GAIN_THRESHOLD}, "
        #         f"MIN_SELL_LOSS_THRESHOLD={self.MIN_SELL_LOSS_THRESHOLD}")

        trading_targets = self.get_training_labels(
            dataframe,
        )
        dataframe["trading"] = trading_targets

        # save buy/sell predictions
        self.buy_predictions = np.where(trading_targets == TradingAction.BUY, 1, 0)
        self.sell_predictions = np.where(trading_targets == TradingAction.SELL, 1, 0)

        # Verify predictions match trading_targets
        df_len = len(dataframe)
        if len(self.buy_predictions) != df_len or len(self.sell_predictions) != df_len:
            print(f"WARNING: Length mismatch in update_predictions!")
            print(f"  dataframe length: {df_len}")
            print(f"  buy_predictions length: {len(self.buy_predictions)}")
            print(f"  sell_predictions length: {len(self.sell_predictions)}")
            print(f"  trading_targets length: {len(trading_targets)}")

        # Verify no conflicts (both buy and sell set)
        conflicts = np.sum((self.buy_predictions == 1) & (self.sell_predictions == 1))
        if conflicts > 0:
            print(
                f"WARNING: {conflicts} positions have both buy and sell predictions set!"
            )

        # Verify counts match trading_targets
        num_buys = np.sum(self.buy_predictions == 1)
        num_sells = np.sum(self.sell_predictions == 1)
        num_holds = df_len - num_buys - num_sells
        expected_buys = np.sum(trading_targets == TradingAction.BUY)
        expected_sells = np.sum(trading_targets == TradingAction.SELL)
        expected_holds = np.sum(trading_targets == TradingAction.HOLD)

        if num_buys != expected_buys or num_sells != expected_sells:
            print(f"WARNING: Prediction counts don't match trading_targets!")
            print(
                f"  Expected: buys={expected_buys}, sells={expected_sells}, holds={expected_holds}"
            )
            print(f"  Got: buys={num_buys}, sells={num_sells}, holds={num_holds}")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # add the indicators to the dataframe

        # set parameters within NNMTStrategy based on hyperopt params here
        # cannot hyperopt in main strategy
        self.MIN_BUY_GAIN_THRESHOLD = self.min_buy_gain_threshold.value
        self.TRAINING_TYPE = self.training_type.value
        self.MIN_SELL_LOSS_THRESHOLD = self.min_sell_loss_threshold.value

        self.iteration_init()

        dataframe = self.dataframePopulator.add_indicators(
            dataframe,
            dataset_type=DatasetType.MINIMAL,
            # dataframe, dataset_type=DatasetType.CUSTOM3
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

        if self.dp.runmode.value in ("hyperopt"):
            dataframe = self.update_predictions(dataframe)
        else:
            if "%train_buy" in dataframe.columns:
                buys = dataframe["%train_buy"].to_numpy()
                if len(buys) == len(dataframe):
                    return buys
            if self.buy_predictions is None or len(self.buy_predictions) != len(
                dataframe
            ):
                dataframe = self.update_predictions(dataframe)

        # Ensure predictions match dataframe length
        df_len = len(dataframe)
        if len(self.buy_predictions) != df_len:
            print(f"WARNING: emulate_buy_signals length mismatch!")
            print(f"  dataframe length: {df_len}")
            print(f"  buy_predictions length: {len(self.buy_predictions)}")
            # Truncate or pad to match
            if len(self.buy_predictions) > df_len:
                return self.buy_predictions[:df_len]
            else:
                padded = np.zeros(df_len, dtype=int)
                padded[: len(self.buy_predictions)] = self.buy_predictions
                return padded

        return self.buy_predictions

    def emulate_sell_signals(self, dataframe: DataFrame):
        """Emulate the sell signals based on the training signals"""
        if "%train_sell" in dataframe.columns:
            sells = dataframe["%train_sell"].to_numpy()
            if len(sells) == len(dataframe):
                return sells
        if self.sell_predictions is None or len(self.sell_predictions) != len(
            dataframe
        ):
            dataframe = self.update_predictions(dataframe)

        # Ensure predictions match dataframe length
        df_len = len(dataframe)
        if len(self.sell_predictions) != df_len:
            print(f"WARNING: emulate_sell_signals length mismatch!")
            print(f"  dataframe length: {df_len}")
            print(f"  sell_predictions length: {len(self.sell_predictions)}")
            # Truncate or pad to match
            if len(self.sell_predictions) > df_len:
                return self.sell_predictions[:df_len]
            else:
                padded = np.zeros(df_len, dtype=int)
                padded[: len(self.sell_predictions)] = self.sell_predictions
                return padded

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
        if self.entry_enable_guards.value:

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

        conditions.append(dataframe["atr_pct_roll"] > self.entry_atr_pct.value)

        # get the lookahead buy/sell signals
        # self.PEAK_WINDOW = self.peak_window.value

        buys = self.emulate_buy_signals(dataframe)

        # Ensure buys array matches dataframe length
        df_len = len(dataframe)
        if len(buys) != df_len:
            print(f"WARNING: populate_entry_trend length mismatch!")
            print(f"  dataframe length: {df_len}")
            print(f"  buys length: {len(buys)}")
            if len(buys) > df_len:
                buys = buys[:df_len]
            else:
                padded = np.zeros(df_len, dtype=int)
                padded[: len(buys)] = buys
                buys = padded

        dataframe["predict_buy"] = np.where(buys == 1, 1, 0)

        # Verify counts match training labels
        num_predict_buy = np.sum(dataframe["predict_buy"] > 0.5)
        num_buys_array = np.sum(buys == 1)
        if "%train_buy" in dataframe.columns:
            num_train_buy = np.sum(dataframe["%train_buy"] > 0.5)
            if num_predict_buy != num_train_buy:
                print(
                    f"WARNING: predict_buy count ({num_predict_buy}) doesn't match %train_buy count ({num_train_buy})"
                )

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

        # Ensure sells array matches dataframe length
        df_len = len(dataframe)
        if len(sells) != df_len:
            print(f"WARNING: populate_exit_trend length mismatch!")
            print(f"  dataframe length: {df_len}")
            print(f"  sells length: {len(sells)}")
            if len(sells) > df_len:
                sells = sells[:df_len]
            else:
                padded = np.zeros(df_len, dtype=int)
                padded[: len(sells)] = sells
                sells = padded

        dataframe["predict_sell"] = np.where(sells == 1, 1, 0)

        # Verify counts match training labels
        num_predict_sell = np.sum(dataframe["predict_sell"] > 0.5)
        num_sells_array = np.sum(sells == 1)
        if "%train_sell" in dataframe.columns:
            num_train_sell = np.sum(dataframe["%train_sell"] > 0.5)
            if num_predict_sell != num_train_sell:
                print(
                    f"WARNING: predict_sell count ({num_predict_sell}) doesn't match %train_sell count ({num_train_sell})"
                )

        # Calculate holds from predict_buy and predict_sell (if both exist)
        if "predict_buy" in dataframe.columns and "predict_sell" in dataframe.columns:
            num_predict_buy = np.sum(dataframe["predict_buy"] > 0.5)
            num_predict_hold = np.sum(
                (dataframe["predict_buy"] < 0.5) & (dataframe["predict_sell"] < 0.5)
            )
            total = len(dataframe)
            # print(f"    predict_buy/predict_sell distribution: buys={num_predict_buy}, sells={num_predict_sell}, holds={num_predict_hold} (total={total})")

        conditions.append(dataframe["predict_sell"] == 1)

        # Apply conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
        else:
            dataframe["exit_long"] = 0

        return dataframe
