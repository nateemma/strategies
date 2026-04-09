from freqtrade.strategy import IStrategy
import pandas as pd
import pandas_ta as pta
import numpy as np
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_open,
)

# set paths so that we can find imports in parallel directories
import os
import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)


import warnings

warnings.filterwarnings(
    "ignore", message="The objective has been evaluated at this point before."
)

from Ehlers_Indicator import Ehlers_Indicator

"""
Base class for reversion using Fisher Transform to find momentum changes
"""


class Ehlers_ROC(Ehlers_Indicator):

    indicator: str = "roc"

    def add_main_indicator(self, dataframe):
        # 10-period ROC
        dataframe["roc"] = (
            (dataframe["close"] - dataframe["close"].shift(10))
            / dataframe["close"].shift(10).replace(0, np.nan).fillna(1e-12)
        ).fillna(0.0)
        return dataframe
