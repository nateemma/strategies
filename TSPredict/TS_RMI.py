# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0411, C0413,  W1203

"""
####################################################################################
TS_RMI - predict future values of 'rmi' column


This is currently a complete hack. I'm using the 'gain' infrastructure to predict RMI
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


from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor

import utils.custom_indicators as cta

from TSPredict import TSPredict



class TS_RMI(TSPredict):


    def add_strategy_indicators(self, dataframe):

        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        dataframe['srmi'] = 2.0 * (dataframe['rmi'] - 50.0) / 100.0

        # target profit/loss thresholds
        dataframe["profit"] = dataframe["srmi"].clip(upper=0.0)
        dataframe["loss"] = dataframe["srmi"].clip(lower=0.0)

        return dataframe

    def get_data(self, dataframe):
        # supply *only* the gain column (and standard indicators)
        col_list = ['date', 'open', 'close', 'high', 'low', 'volume', 'srmi']
        # col_list = ['date', 'gain']
        df = dataframe[col_list].copy()
        return np.array(self.convert_dataframe(df))

    def create_model(self, df_shape):
        # print("    creating new model using: XGBRegressor")
        # params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
        # self.model = XGBRegressor(**params)

        self.model = PassiveAggressiveRegressor(warm_start=True)
        # self.model = SGDRegressor(loss='huber')

        print(f"    creating new model using: {type(self.model)}")

        if self.model is None:
            print("***    ERR: create_model() - model was not created ***")
        return


    ###################################

