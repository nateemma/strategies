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

    buy_params = { **TSPredict.buy_params,
        "entry_enable_guards": False
        }
    
    forecaster_type = Forecasters.ForecasterType.PA

    # Run add_rolling_predictions through the declared forecaster (PA) rather than
    # the hardcoded MLX-MLP / LightGBM paths. The MLX path (a from-scratch 256-128
    # MLP, full-batch, 100 epochs per chunk) is what hung on the ~750k-row
    # aggregate; LightGBM ran but overfit the smoothed-gain target into
    # near-constant forecasts (1 trade in 2yr). PA generalises (11 trades, PF 3.02,
    # Calmar 5.51) and matches the live path, which already forecasts via
    # self.forecaster.
    use_forecaster = True

    # Shorten the walk-forward startup ramp. The base 2000-candle floor leaves
    # predicted_gain=0 (no entries) for ~20d at 15m / ~333d at 4h; PA needs far
    # less history to start, so 500 (~5d at 15m) cuts the dead zone ~4x.
    # (use_forecaster / initial_train_min only affect the walk-forward path; they
    # are superseded by static_model below but kept for the non-static fallback.)
    initial_train_min = 500

    # Live-safe: train the PA forecaster once on full history + persist, then
    # load + predict per bar in backtest AND live. The walk-forward path can't
    # run live (needs tens of thousands of rows; the live buffer is ~950 candles).
    static_model = True

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
