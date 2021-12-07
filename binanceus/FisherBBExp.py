
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


from FisherBB import FisherBB


class FisherBBExp(FisherBB):
    """
    Simple strategy based on Inverse Fisher Transform and Bollinger Bands
    This is a self-contained version tuned for Expectancy (see freqtrade edge page)

    How to use it?
    > python3 ./freqtrade/main.py -s FisherBBExp
    """

    # Buy hyperspace params:
    buy_params = {
        "buy_bb_gain": 0.1,
        "buy_fisher": 0.03,
        "buy_num_candles": 5
    }

    # ROI table:
    minimal_roi = {
        "0": 0.133,
        "16": 0.08,
        "30": 0.01,
        "97": 0
    }

    # Stoploss:
    stoploss = -0.071

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.036
    trailing_only_offset_is_reached = True


    buy_bb_gain = DecimalParameter(0.01, 0.20, decimals=3, default=0.070, space="buy")
    buy_fisher = DecimalParameter(-1.0, 1.0, decimals=3, default=-0.280, space="buy")
    buy_num_candles = IntParameter(0, 12, default=2, space="buy")
