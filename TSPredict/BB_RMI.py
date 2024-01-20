# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413,  W1203, W291

"""
####################################################################################
BB_RMI - very simple strategy that just monitors Billinger Band and RSI Momentum Indicator (RMI)

        Note: this is mostly intended as a away to get optimal values for the guard indicators, using hyperopt
        These guard values are used in many strategies, especially the TSPredict family
####################################################################################
"""


import copy
import cProfile
import os
import pstats

import sys
import traceback
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Optional

import logging
import warnings

import joblib
import numpy as np


import pandas as pd
import pywt
from pandas import DataFrame, Series

import talib.abstract as ta
import finta

import technical.indicators as ftt

# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from freqtrade import leverage

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

# import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IStrategy


group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)


import utils.custom_indicators as cta

from utils.DataframeUtils import DataframeUtils, ScalerType  # pylint: disable=E0401


class BB_RMI(IStrategy):
    # Do *not* hyperopt for the roi and stoploss spaces

    plot_config = {
        "main_plot": {
            "close": {"color": "cornflowerblue"},
            },
        "subplots": {
            "Diff": {
                "guard_metric": {"color": "orange"},
                "buy_region": {"color": "darkseagreen"},
                "sell_region": {"color": "darksalmon"},
            },
        },
    }

    # ROI table:
    minimal_roi = {"0": 0.04, "100": 0.02}

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = "5m"
    inf_timeframe = "15m"

    use_custom_stoploss = True

    leverage = 1.0
    can_short = False
    # if setting can-short to True, remember to update the config file:
    #   "trading_mode": "futures",
    #   "margin_mode": "isolated",

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 128  # must be power of 2

    process_only_new_candles = True

    custom_trade_info = {}  # pair-specific data
    curr_pair = ""

    ###################################

    # Strategy Specific Variable Storage
    dataframeUtils = None
    scaler = RobustScaler()
    training_mode = False

    # hyperparams


    # Buy hyperspace params:
    buy_params = {
        "cexit_min_profit_th": 0.1,
        "cexit_profit_nstd": 0.6,
        "entry_bb_factor": 1.11,
        "entry_bb_width": 0.02,
        "entry_guard_metric": -0.2,
        "enable_bb_check": True,  # value loaded from strategy
        "enable_guard_metric": True,  # value loaded from strategy
        "enable_squeeze": True,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_loss_nstd": 1.1,
        "cexit_metric_overbought": 0.74,
        "cexit_metric_take_profit": 0.76,
        "cexit_min_loss_th": -1.3,
        "exit_bb_factor": 0.75,
        "exit_guard_metric": 0.1,
        "enable_exit_signal": True,  # value loaded from strategy
    }

    # Entry

    # the following flags apply to both entry and exit
    enable_guard_metric = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
        )

    enable_bb_check = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
        )

    enable_squeeze = CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=False
        )

    entry_guard_metric = DecimalParameter(
        -0.8, 0.0, default=-0.6, decimals=1, space="buy", load=True, optimize=True
        )

    entry_bb_width = DecimalParameter(
        0.020, 0.100, default=0.04, decimals=3, space="buy", load=True, optimize=True
        )

    entry_bb_factor = DecimalParameter(
        0.70, 1.20, default=1.1, decimals=2, space="buy", load=True, optimize=True
        )


    # Exit
    # use exit signal? If disabled, just rely on the custom exit checks (or stoploss) to get out
    enable_exit_signal = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=True
        )

    exit_guard_metric = DecimalParameter(
        0.0, 0.8, default=0.2, decimals=1, space="sell", load=True, optimize=True
        )

    exit_bb_factor = DecimalParameter(
        0.70, 1.20, default=0.8, decimals=2, space="sell", load=True, optimize=True
        )



    # Custom Exit

    # No. Standard Deviations of profit/loss for target, and lower limit
    cexit_min_profit_th = DecimalParameter(0.0, 1.5, default=0.7, decimals=1, space="buy", load=True, optimize=True)
    cexit_profit_nstd = DecimalParameter(0.0, 3.0, default=0.9, decimals=1, space="buy", load=True, optimize=True)

    cexit_min_loss_th = DecimalParameter(-1.5, -0.0, default=-0.4, decimals=1, space="sell", load=True, optimize=True)
    cexit_loss_nstd = DecimalParameter(0.0, 3.0, default=0.7, decimals=1, space="sell", load=True, optimize=True)

    # Guard metric sell limits - used to bail out when in profit
    cexit_metric_overbought = DecimalParameter(
        0.55, 0.99, default=0.96, decimals=2, space="sell", load=True, optimize=True
        )

    cexit_metric_take_profit = DecimalParameter(
        0.55, 0.99, default=0.76, decimals=2, space="sell", load=True, optimize=True
        )


    ###################################

    def bot_start(self, **kwargs) -> None:
        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()
            self.dataframeUtils.set_scaler_type(ScalerType.Robust)

        return

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        return []

    ###################################
    
    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # NOTE: if you change the indicators, you need to regenerate the model

        # Base pair dataframe timeframe indicators
        curr_pair = metadata["pair"]

        win_size = 32

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=win_size, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'])
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
        dataframe["bb_loss"] = ((dataframe["bb_lowerband"] - dataframe["close"]) / dataframe["close"])

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=win_size, mom=5)

        # scaled version for use as guard metric
        dataframe['srmi'] = 2.0 * (dataframe['rmi'] - 50.0) / 100.0

        # guard metric must be in range [-1,+1], with -ve values indicating oversold and +ve values overbought
        dataframe['guard_metric'] = dataframe['srmi']


        dataframe.fillna(0.0, inplace=True)

        return dataframe

    ###################################

    # -------------

    def convert_dataframe(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()

        # convert date column so that it can be scaled.
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], utc=True)
            df["date"] = dates.astype("int64")

        df.fillna(0.0, inplace=True)

        df.set_index("date")
        df.reindex()

        # print(f'    norm_data:{self.norm_data}')
        if self.norm_data:
            # scale the dataframe
            self.scaler.fit(df)
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        return df


    ###################################

    """
    entry Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""
        dataframe["enter_long"] = 0
        dataframe['buy_region'] = 0

        if self.training_mode:
            return dataframe

        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 1.0)

        guard_conditions = []

        if self.enable_guard_metric.value:

            # Guard metric in oversold region
            guard_conditions.append(dataframe["guard_metric"] < self.entry_guard_metric.value)

            # in lower portion of previous window
            # conditions.append(dataframe["close"] < dataframe["local_mean"])

        if self.enable_bb_check.value:
            # Bollinger band-based bull/bear indicators:
            # Done here so that we can use hyperopt to find values

            lower_limit = dataframe['bb_middleband'] - \
                self.exit_bb_factor.value * (dataframe['bb_middleband'] - dataframe['bb_lowerband'])

            dataframe['bullish'] = np.where(
                (dataframe['close'] <= lower_limit)
                , 1, 0)

            # bullish region
            guard_conditions.append(dataframe["bullish"] > 0)

            # # not bearish (looser than bullish)
            # conditions.append(dataframe["bearish"] >= 0)

        if self.enable_squeeze.value:
            dataframe['squeeze'] = np.where(
                (dataframe['bb_width'] >= self.entry_bb_width.value)
                , 1, 0)

            guard_conditions.append(dataframe['squeeze'] > 0)


        # add coulmn that combines guard conditions (for plotting)
        if guard_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, guard_conditions), "buy_region"] = 1

        # model triggers
        model_cond = (
            # buy region
            (dataframe["buy_region"] > 0)
        )

        # conditions.append(metric_cond)
        conditions.append(model_cond)

        # set entry tags
        dataframe.loc[model_cond, "enter_tag"] += "model_entry "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    ###################################

    """
    exit Signal
    """

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "exit_tag"] = ""
        dataframe["exit_long"] = 0
        dataframe['sell_region'] = 0

        if self.training_mode or (not self.enable_exit_signal.value):
            return dataframe


        dataframe['sell_region'] = 0
        guard_conditions = []


        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 0)

        if self.enable_guard_metric.value:

            # Guard metric in overbought region
            guard_conditions.append(dataframe["guard_metric"] > self.exit_guard_metric.value)

            # in upper portion of previous window
            # guard_conditions.append(dataframe["close"] > dataframe["local_mean"])

        if self.enable_bb_check.value:

            # Bollinger band-based bull/bear indicators:
            # Done here so that we can use hyperopt to find values

            upper_limit = dataframe['bb_middleband'] + \
            self.entry_bb_factor.value * (dataframe['bb_upperband'] - dataframe['bb_middleband'])

            dataframe['bearish'] = np.where(
                (dataframe['close'] >= upper_limit)
                , -1, 0)

            # bearish region
            guard_conditions.append(dataframe["bearish"] < 0)

            # # not bullish (looser than bearish)
            # conditions.append(dataframe["bullish"] <= 0)

        if self.enable_squeeze.value:
            if not ('squeeze' in dataframe.columns):
                dataframe['squeeze'] = np.where(
                    (dataframe['bb_width'] >= self.entry_bb_width.value)
                , 1, 0)

            guard_conditions.append(dataframe['squeeze'] > 0)


        if guard_conditions:
            # add column that combines guard conditions (for plotting)
            dataframe.loc[reduce(lambda x, y: x & y, guard_conditions), "sell_region"] = -1

        # model triggers
        model_cond = (

            # sell region
            (dataframe["sell_region"] < 0)
        )

        conditions.append(model_cond)

        # set exit tags
        dataframe.loc[model_cond, "exit_tag"] += "model_exit "

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe

    ###################################


    """
    Custom Stoploss
    """

    # simplified version of custom trailing stoploss
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
        # this is just here so that we can use custom_exit

        # return min(-0.001, max(stoploss_from_open(0.05, current_profit), -0.99))
        return self.stoploss

    ###################################

    """
    Custom Exit
    (Note that this runs even if use_custom_stoploss is False)
    """

    # simplified version of custom exit

    def custom_exit(self, pair: str, trade: Trade, current_time: "datetime", current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if not self.use_custom_stoploss:
            return None

        # check volume?!
        if last_candle['volume'] <= 1.0:
            return None

        if trade.is_short:
            print("    short trades not yet supported in custom_exit()")

        else:

            # print("    checking long trade")

            # strong sell signal, in profit
            if (current_profit > 0.0) and (last_candle["guard_metric"] >= self.cexit_metric_overbought.value):
                return "metric_overbought"

            # Above 0.5%, sell if Fisher/Williams in sell range
            if current_profit > 0.005:
                if last_candle["guard_metric"] >= self.cexit_metric_take_profit.value:
                    return "take_profit"

            # if in profit and exit signal is set, sell (even if exit signals are disabled)
            if (current_profit > 0) and (last_candle["exit_long"] > 0):
                return "exit_signal"


        # The following apply to both long & short trades:

        # Sell any positions if open for >= 1 day with any level of profit
        if ((current_time - trade.open_date_utc).days >= 1) & (current_profit > 0):
            return "unclog_1"

        # Sell any positions at a loss if they are held for more than 7 days.
        if (current_time - trade.open_date_utc).days >= 7:
            return "unclog_7"

        return None
