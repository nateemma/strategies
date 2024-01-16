"""
####################################################################################
TS_Simple - base class for 'simple' time series prediction
             Handles most of the logic for time series prediction. Subclasses should
             override the model-related functions

             This strategy uses only the 'base' indicators (open, close etc.) to estimate future gain.
             Note that I use gain rather than price because it is a normalised value, and works better with prediction algorithms.
             I use the actual (future) gain to train a base model, which is then further refined for each individual pair.
             The model is created if it does not exist, and is trained on all available data before being saved.
             Models are saved in user_data/strategies/TSPredict/models/<class>/<class>.sav, where <class> is the name of the current class
             (TS_Simple if running this directly, or the name of the subclass). 
             If the model already exits, then it is just loaded and used.
             So, it makes sense to do initial training over a long period of time to create the base model. 
             If training, then no backtesting or tuning for individual pairs is performed (way faster).
             If you want to retrain (e.g. you changed indicators), then delete the model and run the strategy over a long time period

####################################################################################
"""


#pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, W1203


import sys
from pathlib import Path

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series


import freqtrade.vendor.qtpylib.indicators as qtpylib

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy


group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

import talib.abstract as ta

from TSPredict import TSPredict



class TS_Simple(TSPredict):

    use_rolling = True
    merge_indicators = True

    def add_strategy_indicators(self, dataframe):

        # add some extra indicators

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']


        # moving averages
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.win_size)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.win_size)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.win_size)

        # Donchian Channels
        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.win_size)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.win_size)
        dataframe['dc_mid'] = ta.TEMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2), timeperiod=self.win_size)

        return dataframe

