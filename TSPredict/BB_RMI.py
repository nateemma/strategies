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
                "bullish": {"color": "darkseagreen"},
                "bearish": {"color": "darksalmon"},
                "squeeze": {"color": "cornflowerblue"},
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

    # hyperparams
    # Buy hyperspace params:
    buy_params = {
        "entry_bb_factor": 1.14,
        "entry_bb_width": 0.051,
        "entry_enable_squeeze": True,
        "entry_guard_metric": -0.6,
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_metric_overbought": 0.73,
        "cexit_metric_take_profit": 0.86,
        "exit_bb_factor": 0.91,
        "exit_guard_metric": 0.3,
        "enable_exit_signal": True,  # value loaded from strategy
    }

    # entry params

    entry_enable_squeeze= CategoricalParameter(
        [True, False], default=True, space="buy", load=True, optimize=True
        )

    entry_bb_width = DecimalParameter(
        0.01, 0.100, default=0.02, decimals=3, space="buy", load=True, optimize=True
        )

    entry_bb_factor = DecimalParameter(
        0.70, 1.20, default=0.8, decimals=2, space="buy", load=True, optimize=True
        )

    entry_guard_metric = DecimalParameter(
        -0.8, 0.0, default=-0.5, decimals=1, space="buy", load=True, optimize=True
        )

    # exit params

    # use exit signal? If disabled, just rely on the custom exit checks (or stoploss) to get out
    enable_exit_signal = CategoricalParameter(
        [True, False], default=True, space="sell", load=True, optimize=False
        )

    exit_bb_factor = DecimalParameter(
        0.70, 1.20, default=0.8, decimals=2, space="sell", load=True, optimize=True
        )

    exit_guard_metric = DecimalParameter(
        0.0, 0.8, default=0.5, decimals=1, space="sell", load=True, optimize=True
        )

    # Custom Exit params

    # Metric-based sell limits - used to bail out when in profit
    cexit_metric_overbought = DecimalParameter(
        0.55, 0.99, default=0.99, decimals=2, space="sell", load=True, optimize=True
        )
    cexit_metric_take_profit = DecimalParameter(
        0.55, 0.99, default=0.99, decimals=2, space="sell", load=True, optimize=True
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

        # Bollinger band-based bull/bear indicators:

        # calculate limits slightly within upper/lower bands

        lower_limit = dataframe['bb_middleband'] - \
             self.exit_bb_factor.value * (dataframe['bb_middleband'] - dataframe['bb_lowerband'])

        dataframe['bullish'] = np.where(
            (dataframe['close'] <= lower_limit)
            , 1, 0)

        dataframe['squeeze'] = np.where(
            (dataframe['bb_width'] >= self.entry_bb_width.value)
            , 1, 0)

        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 1.0)

        # Guard metric in oversold region
        conditions.append(dataframe["guard_metric"] < self.entry_guard_metric.value)

        # bullish region
        conditions.append(dataframe["bullish"] > 0)

        # wide Bollinger Bands
        if self.entry_enable_squeeze.value:
            conditions.append(dataframe['squeeze'] > 0)

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

        upper_limit = dataframe['bb_middleband'] + \
            self.entry_bb_factor.value * (dataframe['bb_upperband'] - dataframe['bb_middleband'])

        dataframe['bearish'] = np.where(
            (dataframe['close'] >= upper_limit)
            , -1, 0)

        # some trading volume (otherwise expect spread problems)
        conditions.append(dataframe["volume"] > 0)

        # Guard metric in overbought region
        conditions.append(dataframe["guard_metric"] > self.exit_guard_metric.value)

        # bearish region
        conditions.append(dataframe["bearish"] < 0)

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
