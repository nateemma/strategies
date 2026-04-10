"""
####################################################################################
TS_Coeff - base class for 'simple' time series prediction
             Handles most of the logic for time series prediction. Subclasses should
             override the model-related functions

             This strategy uses only the 'base' indicators (open, close etc.) to estimate future gain.
             Note that I use gain rather than price because it is a normalised value, and works better with prediction algorithms.
             I use the actual (future) gain to train a base model, which is then further refined for each individual pair.
             The model is created if it does not exist, and is trained on all available data before being saved.
             Models are saved in user_data/strategies/TSPredict/models/<class>/<class>.sav, where <class> is the name of the current class
             (TS_Coeff if running this directly, or the name of the subclass).
             If the model already exits, then it is just loaded and used.
             So, it makes sense to do initial training over a long period of time to create the base model.
             If training, then no backtesting or tuning for individual pairs is performed (way faster).
             If you want to retrain (e.g. you changed indicators), then delete the model and run the strategy over a long time period

####################################################################################
"""

# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, W1203


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
import utils.custom_indicators as cta

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

import pywt


from sklearn.linear_model import SGDRegressor

from TSPredict import TSPredict


class TS_Coeff(TSPredict):

    coeff_table = None

    # wavelet_type = Wavelets.WaveletType.DWT

    use_rolling = True

    # -------------

    # override func to get data for this strategy
    def get_data(self, dataframe):

        df_norm = self.convert_dataframe(dataframe)
        gain_data = df_norm["gain"].to_numpy()
        # gain_data = self.smooth(gain_data, 2)
        self.build_coefficient_table(gain_data)
        
        # Get the feature columns from the normalized dataframe
        features = [col for col in df_norm.columns if col != "gain"]
        
        # Merge with coefficients
        data = self.merge_coeff_table(df_norm[features])
        
        # Generate names for coefficients
        num_coeffs = np.shape(self.coeff_table)[1]
        coeff_names = [f"coeff_{i}" for i in range(num_coeffs)]
        
        # Combined feature list
        all_features = features + coeff_names
        
        return data, all_features

    # -------------

    # builds a numpy array of coefficients
    def build_coefficient_table(self, win_data: np.array):
        # Optimized vectorized version
        from numpy.lib.stride_tricks import sliding_window_view

        # lazy initialisation of vars (so thatthey can be changed in subclasses)
        if self.wavelet is None:
            self.wavelet = Wavelets.make_wavelet(self.wavelet_type)
        
        self.wavelet.set_lookahead(self.lookahead)

        # Prepare windows
        if len(win_data) < self.wavelet_size:
            # Not enough data yet
            self.coeff_table = np.zeros((len(win_data), 10))  # dummy size
            return

        windows = sliding_window_view(win_data, self.wavelet_size)

        # Batch transform if using SWT, otherwise loop (DWT/others are less batch-friendly in PyWT)
        # Note: TS_Coeff usually uses DWT which isn't natively batchable in PyWT without reshapes

        # Check if we can use the vectorized SWT path (if applicable)
        if self.wavelet_type == Wavelets.WaveletType.SWT:
            levels = pywt.swt_max_level(self.wavelet_size)
            wname = getattr(self.wavelet, "wavelet", None) or "db1"
            coeffs = pywt.swt(
                windows, wname, level=levels, axis=-1, trim_approx=True
            )
            # coeffs is a list of levels, each is an array of (N, WaveletSize)
            # we need to flat-concatenate them along the last axis
            c_array = np.concatenate(coeffs, axis=-1)
            result = c_array
        else:
            # Fallback for DWT/others - still faster than the original because we only copy once
            c_table = []
            for i in range(len(windows)):
                # .copy() fixes the "read-only buffer" error
                coeffs = self.wavelet.get_coeffs(windows[i].copy())
                features = self.wavelet.coeff_to_array(coeffs)
                c_table.append(features)
            result = np.array(c_table)

        # Initialize and populate
        nrows = len(win_data)
        num_coeffs = result.shape[1]
        self.coeff_table = np.zeros((nrows, num_coeffs), dtype=float)

        # Offset matches the end of the window
        offset = self.wavelet_size - 1
        self.coeff_table[offset : offset + result.shape[0]] = result
        return

    # -------------

    # merge the supplied dataframe with the coefficient table. Number of rows must match
    def merge_coeff_table(self, dataframe: DataFrame):

        # print(f'merge_coeff_table() self.coeff_table: {np.shape(self.coeff_table)}')

        # # apply smoothing to each column, otherwise prediction alogorithms will struggle
        # num_cols = np.shape(self.coeff_table)[1]
        # for j in range (num_cols):
        #     feature = self.coeff_table[:,j]
        #     feature = self.smooth(feature, 2)
        #     self.coeff_table[:,j] = feature

        num_coeffs = np.shape(self.coeff_table)[1]

        merged_table = np.concatenate([np.array(dataframe), self.coeff_table], axis=1)

        return merged_table

    # -------------
