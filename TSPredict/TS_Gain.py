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



from TSPredict import TSPredict



class TS_Gain(TSPredict):

    use_rolling = True

    def add_strategy_indicators(self, dataframe):
        return dataframe

    def get_data(self, dataframe):
        # supply *only* the gain column (and standard indicators)
        col_list = ['date', 'open', 'close', 'high', 'low', 'volume', 'gain']
        # col_list = ['date', 'gain']
        df = dataframe[col_list].copy()
        return np.array(self.convert_dataframe(df))
        
