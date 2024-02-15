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



class TS_Gain(TSPredict):



    # Buy hyperspace params:
    buy_params = {
        "cexit_min_profit_th": 0.3,
        "cexit_profit_nstd": 2.2,
        "enable_bb_check": True,
        "entry_bb_factor": 0.79,
        "entry_bb_width": 0.023,
        "entry_guard_metric": -0.5,
        "enable_guard_metric": True,  # value loaded from strategy
        "enable_squeeze": True,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_loss_nstd": 0.6,
        "cexit_metric_overbought": 0.62,
        "cexit_metric_take_profit": 0.62,
        "cexit_min_loss_th": -0.1,
        "enable_exit_signal": False,
        "exit_bb_factor": 0.72,
        "exit_guard_metric": 0.0,
    }


    use_rolling = True
    merge_indicators = True
    single_col_prediction = True
    detrend_data = True

    forecaster_type = Forecasters.ForecasterType.PA

    def add_strategy_indicators(self, dataframe):
        return dataframe

    def get_data(self, dataframe):
        # supply *only* the gain column
        gain = dataframe['gain'].to_numpy()
        gain = self.smooth(gain, 1)
        return gain

