# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0411, C0413,  W1203

"""
####################################################################################
TS_Gain - predict future values of 'gain' column
             

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

    use_rolling = True
    merge_indicators = True
    single_col_prediction = False

    if single_col_prediction:
        forecaster_type = Forecasters.ForecasterType.SGD
    else:
        forecaster_type = Forecasters.ForecasterType.PA

    def add_strategy_indicators(self, dataframe):
        return dataframe

    def get_data(self, dataframe):
        # supply *only* the gain column (and standard indicators)
        if self.single_col_prediction:
            col_list = ['date', 'gain']
        else:
            col_list = ['date', 'open', 'close', 'high', 'low', 'volume', 'gain']

        df = dataframe[col_list].copy()
        gain = dataframe['gain'].to_numpy()
        gain = self.smooth(gain, 2)
        dataframe['gain'] = gain
        return np.array(self.convert_dataframe(df))

