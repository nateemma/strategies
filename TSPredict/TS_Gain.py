# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0411, C0413,  W1203

"""
####################################################################################
TS_Gain - predict future values of 'gain' column (and nothing else)


####################################################################################
"""


import sys
from pathlib import Path

import numpy as np

# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series


pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy


group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)


import utils.Forecasters as Forecasters

from TSPredict import TSPredict
from Framework.BaseStrategy import (
    StrategyConfig,
    NormalizationType,
    ModelType,
)


class TS_Gain(TSPredict):
    # Strategy configuration
    strategy_config = StrategyConfig(
        normalization=NormalizationType.ROLLING_ROBUST,
        norm_data=True,
        scale_results=True,
        aggregate_pairs=True,
        model_type=ModelType.CUSTOM,
        needs_training=True,
    )

    use_rolling = True
    merge_indicators = True
    single_col_prediction = True
    detrend_data = True

    forecaster_type = Forecasters.ForecasterType.PA

    # exclude indicators from base class (we only want gain history)
    include_list = []

    def add_strategy_indicators(self, dataframe):
        return dataframe

    def get_data(self, dataframe):
        # Create a matrix of lagged gain values to give the model 'memory'
        # This prevents the 'lagging' prediction problem where the model just matches the current value
        gain = dataframe["gain"].to_numpy()
        
        # Smooth slightly to remove noise
        gain = self.smooth(gain, 1)
        
        # Create lags (8-16 is usually good for simple autoregression)
        lags = 16
        features = []
        feature_names = []
        for i in range(lags):
            # Shift data causally
            feat = np.nan_to_num(np.roll(gain, i))
            # Zero out the 'look-backward' artifacts at the start
            feat[:i] = 0.0
            features.append(feat)
            feature_names.append(f"gain_lag_{i}")
            
        # Combine into (N, lags) matrix
        data = np.column_stack(features)
        
        return data, feature_names
