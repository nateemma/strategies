# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413,  W1203, W291


"""
####################################################################################
NNCoeffs - uses a Long-Short Term Memory neural network to try and predict the future stock price

      This works by creating a LSTM model that we train on the historical gain, then use that model to predict 
      future values

      This is a derivative of NNGain that adds in wavelet coefficients in addition to the gain. 
      The theory is that this should make it easier for a neural network to model the trendsof the signal
      Note: this needs a lot more data to train than NNGain (because there are way more parameters)

####################################################################################
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy.strategy_helper import merge_informative_pair

import os
import sys
from pathlib import Path
import logging
import warnings

import tensorflow as tf

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
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer.*")

tf.get_logger().setLevel('ERROR')

from NNGain import NNGain
import utils.NNPredictors as NNPredictors

class NNCoeffs(NNGain):

    # Buy hyperspace params:
    # buy_params = {
    #     "cexit_min_profit_th": 0.5,
    #     "cexit_profit_nstd": 1.0,
    #     "enable_bb_check": False,
    #     "enable_guard_metric": True, # set to True once model verified
    #     "enable_squeeze": False,
    #     "entry_bb_factor": 0.82,
    #     "entry_bb_width": 0.025,
    #     "entry_guard_metric": -0.7,
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "cexit_loss_nstd": 1.0,
    #     "cexit_metric_overbought": 0.76,
    #     "cexit_metric_take_profit": 0.94,
    #     "cexit_min_loss_th": -0.9,
    #     "enable_exit_signal": False, # set to False once model verified
    #     "exit_bb_factor": 1.01,
    #     "exit_guard_metric": 0.8,
    # }

    # values from running hyperopt:
    # Buy hyperspace params:
    buy_params = {
        "cexit_min_profit_th": 0.5,
        "cexit_profit_nstd": 0.4,
        "entry_guard_metric": -0.2,
        "enable_bb_check": False,  # value loaded from strategy
        "enable_guard_metric": True,  # value loaded from strategy
        "enable_squeeze": False,  # value loaded from strategy
        "entry_bb_factor": 0.77,  # value loaded from strategy
        "entry_bb_width": 0.097,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_loss_nstd": 1.8,
        "cexit_metric_overbought": 0.7,
        "cexit_metric_take_profit": 0.83,
        "cexit_min_loss_th": -1.0,
        "exit_guard_metric": 0.8,
        "enable_exit_signal": False,  # value loaded from strategy
        "exit_bb_factor": 1.12,  # value loaded from strategy
    }

    # ROI from hyperopt:
    minimal_roi = {
        "0": 0.12,
        "11": 0.051,
        "65": 0.01,
        "180": 0
    }

    force_normalise = True # overrides predictor 

    model_window = NNGain.startup_candle_count
    coeff_table = None
    # wavelet_type: Wavelets.WaveletType = Wavelets.WaveletType.DWT
    # wavelet_type: Wavelets.WaveletType = Wavelets.WaveletType.DWTA
    wavelet_type: Wavelets.WaveletType = Wavelets.WaveletType.SWTA
    wavelet = None

    predictor_type = NNPredictors.PredictorType.LSTM

    # -----------------------
    # add in any indicators to be used for training
    def add_training_indicators(self, dataframe: DataFrame) -> DataFrame:

        # placeholders, just need the columns to be there for later
        dataframe['predicted_gain'] = 0.0
        dataframe['temp'] = 0.0

        '''
        # NOTE: we are applying this to the BTC/USD dataframe, not the normal dataframe (or in addition to anyway)
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        # get BTC dataframe
        btc_dataframe = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.timeframe)

        # merge into main dataframe. This will create columns with a "_5m" suffix for the BTC data
        dataframe = merge_informative_pair(dataframe, btc_dataframe, self.timeframe, self.timeframe, ffill=True)

        dataframe['btc_volume'] = dataframe[f'volume_{self.timeframe}']
        dataframe["btc_gain"] = (
            100.0
            * (dataframe[f"close_{self.timeframe}"] - dataframe[f"close_{self.timeframe}"].shift(self.lookahead))
            / dataframe[f"close_{self.timeframe}"].shift(self.lookahead)
        )

        '''
        return dataframe

    # -----------------------

    def reduce_dataframe(self, dataframe: DataFrame) -> DataFrame:

        # # only use 'gain' (and date index)
        # df = dataframe[['date', 'gain']].copy()
        # df.set_index('date')
        # df.reindex()

        if 'btc_gain' in dataframe.keys():
            df = dataframe[['gain', 'volume', 'btc_gain', 'btc_volume']].copy()
        else:
            df = dataframe[['gain', 'volume']].copy()

        # get the underlying trend of the gain signal
        detrender = make_detrender(DetrenderType.SMOOTH)
        # detrender = make_detrender(DetrenderType.NULL)
        gain = df['gain']
        detrend = detrender.detrend(gain) # need this call to set up trend
        detrend = detrender.get_trend()

        # Hack: constrain to 3 decimal places to make it a little easier for the model
        detrend = np.round(detrend, decimals=2)
        df['gain'] = detrend

        df.reset_index()

        # add the wavelet coefficients of the gain/trend
        if self.wavelet is None:
            self.wavelet = Wavelets.make_wavelet(self.wavelet_type)
        self.build_coefficient_table(detrend)
        df2 = self.merge_coeff_table(df)

        # print(f'df2 cols: {df2.columns}')

        return df2

    # -------------

    # builds a numpy array of coefficients
    # Note that this must be done ina rolling fashion, otherwise it is effectively a form of lookahead
    def build_coefficient_table(self, data):

        # roll through the  data and create coefficients for each step
        nrows = np.shape(data)[0]

        # print(f'build_coefficient_table() data:{np.shape(data)}')

        start = 0
        if nrows > self.model_window:
            end = start + self.model_window - 1
        else:
            end = start + 32
        dest = end

        # print(f"nrows:{nrows} start:{start} end:{end} dest:{dest} nbuffs:{nbuffs}")

        self.coeff_table = None
        num_coeffs = 0
        init_done = False

        while end < nrows:
            dslice = data[start:end]

            # print(f"start:{start} end:{end} dest:{dest} len:{len(dslice)}")

            coeffs = self.wavelet.get_coeffs(dslice) # type: ignore
            features = self.wavelet.coeff_to_array(coeffs) # type: ignore
            # print(f'build_coefficient_table() features: {np.shape(features)}')

            # initialise the np.array (need features first to know size)
            if not init_done:
                init_done = True
                num_coeffs = len(features)
                self.coeff_table = np.zeros((nrows, num_coeffs), dtype=float)
                # print(f"coeff_table:{np.shape(self.coeff_table)}")

            # copy the features to the appropriate row of the coefficient array (offset due to startup window)
            self.coeff_table[dest] = features # type: ignore

            start = start + 1
            dest = dest + 1
            end = end + 1

        # print(f'build_coefficient_table() self.coeff_table: {np.shape(self.coeff_table)}')

        return

    # -------------

    # merge the supplied dataframe with the coefficient table. Number of rows must match
    def merge_coeff_table(self, dataframe: DataFrame) -> DataFrame:

        # print(f'merge_coeff_table() self.coeff_table: {np.shape(self.coeff_table)}')

        num_coeffs = np.shape(self.coeff_table)[1] # type: ignore

        # build column names
        cnames = []
        for i in range(num_coeffs):
            cnames.append(f'coeff_{i}')

        dataframe[cnames] = self.coeff_table
        # merged_table = np.concatenate([np.array(dataframe), self.coeff_table], axis=1)

        return dataframe

    # -------------
