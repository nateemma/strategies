
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from FBB_ import FBB_


class FBB_WtdProfit(FBB_):
    """
    Simple strategy based on Inverse Fisher Transform and Bollinger Bands
    This is a self-contained version tuned for profit, weighted with expectancy, duration etc.
    Note that this inherits from FBB_ - the only difference is the hyperparameter tuning

    How to use it?
    > python3 ./freqtrade/main.py -s FBB_WtdProfit
    """

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_gain": 0.171,
        "buy_fisher": -0.507,
        "buy_num_candles": 0
    }

    # ROI table:
    minimal_roi = {
        "0": 0.204,
        "20": 0.048,
        "78": 0.01,
        "166": 0
    }

    # Stoploss:
    stoploss = -0.349

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.271
    trailing_stop_positive_offset = 0.337
    trailing_only_offset_is_reached = True

    timeframe = '1m'


    buy_bb_gain = DecimalParameter(0.01, 0.20, decimals=3, default=0.070, space="buy")
    buy_fisher = DecimalParameter(-1.0, 1.0, decimals=3, default=-0.280, space="buy")
    buy_num_candles = IntParameter(0, 12, default=2, space="buy")