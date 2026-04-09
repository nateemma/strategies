# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413,  W1203, W291



"""
####################################################################################
NNQGain - uses a Long-Short Term Memory neural network to try and predict the future stock price

      This works by creating a LSTM model that we train on the quantised gain, then use that model to predict 
      future values

      This is a derivative of NNGain that converts the actual gain to a quantised/classified value for prediction.
      The theory is that classification problems are better suited to neural network architectures - time series
      prediction is quite a hard problem.
      The predicted quantised/classified values are converted back to a gain estimate in order fit in with the rest of 
      the architecture (and for visualisation)

####################################################################################
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

import os
import sys
from pathlib import Path
import logging
import warnings


from utils.Detrenders import make_detrender, DetrenderType

import utils.Wavelets as Wavelets

# set paths so that we can find imports in parallel directories
group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# logging setup
pd.options.mode.chained_assignment = None  # default='warn'

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


from NNGain import NNGain
import utils.NNPredictors as NNPredictors

class NNQGain(NNGain):


    # Buy hyperspace params:
    buy_params = {
        "cexit_min_profit_th": 0.7,
        "cexit_profit_nstd": 0.0,
        "enable_bb_check": False,
        "enable_guard_metric": True, # set to True once model verified
        "enable_squeeze": False,
        "entry_bb_factor": 0.82,
        "entry_bb_width": 0.025,
        "entry_guard_metric": -0.7,
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_loss_nstd": 0.0,
        "cexit_metric_overbought": 0.76,
        "cexit_metric_take_profit": 0.94,
        "cexit_min_loss_th": -0.7,
        "enable_exit_signal": False, # set to False once model verified
        "exit_bb_factor": 1.01,
        "exit_guard_metric": 0.8,
    }


    predictor_type = NNPredictors.PredictorType.LSTM

    #-----------------------

    def reduce_dataframe(self, dataframe: DataFrame) -> DataFrame:

        # convert to quantised form (steps of 0.25)
        gain = dataframe['gain']


        # # get the underlying trend of the gain signal
        # detrender = make_detrender(DetrenderType.SMOOTH)
        # detrend = detrender.detrend(gain) # need this call to set up trend
        # detrend = detrender.get_trend()


        # get multiple of 0.25 and convert to int
        # qgain = (detrend / 0.25).round().astype(int)
        qgain = (gain / 0.25).round().astype(int)

        # copy the gain column into a new dataframe (don't want any other columns)
        df = dataframe[['gain']].copy()

        # replace with quanitised/classified gain
        # df['gain'] = detrend
        df['gain'] = qgain

        # print(f'detrend: {detrend}')
        # print(f'qgain: {qgain}')
        df.reset_index()

        return df

    
    #-------------


    def backtest_data(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super().backtest_data(dataframe)
        qgain = dataframe['predicted_gain']

        # gain = (qgain).astype(float) * 0.25
        gain = qgain.round().astype(int) * 0.25
        dataframe['predicted_gain'] = gain


        # print(f'qgain: {qgain}')
        # print(f'gain: {gain}')

        return dataframe
