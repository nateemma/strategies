# pragma pylint: disable=W0105, C0103, C0115, C0116, C0301, C0411, C0413,  W1203



from xgboost import XGBRegressor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pywt


from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

from TS_Wavelet_DWTA import TS_Wavelet_DWTA


# this class is intended to experiment with global settings (without needing to change the base classes)

class TS_Wavelet_roll(TS_Wavelet_DWTA):

    lookahead = 6
    model_window = 128
    train_len = 256
    use_rolling = True
    single_col_prediction = True
    scale_len = 16
    norm_data = True

    # Buy hyperspace params:
    buy_params = {
        "cexit_min_profit_th": 0.5,
        "cexit_profit_nstd": 1.0,
        "entry_guard_fwr": 0.1,
        "enable_entry_guards": True
    }

    # Sell hyperspace params:
    sell_params = {
        "cexit_fwr_overbought": 0.98,
        "cexit_fwr_take_profit": 0.98,
        "cexit_loss_nstd": 1.4,
        "cexit_min_loss_th": -0.5,
        "exit_guard_fwr": 0.0,
        "cexit_enable_large_drop": False,  # value loaded from strategy
        "cexit_large_drop": -1.9,  # value loaded from strategy
        "enable_exit_guards": True,
        "enable_exit_signal": True
    }
