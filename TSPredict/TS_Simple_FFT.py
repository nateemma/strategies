#pragma pylint: disable=W0105, C0103, C0301, W1203

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


import utils.Forecasters as Forecasters

from TS_Simple import TS_Simple

"""
####################################################################################
TS_Simple_FFT - subclass of TS_Simple that uses an FFT forecaster

####################################################################################
"""


class TS_Simple_FFT(TS_Simple):

    combine_models = True
    training_mode = False # set to True to train initial model (over long period)
    supports_incremental_training = True
    forecaster_type = Forecasters.ForecasterType.FFT_EXTRAPOLATION
    use_rolling = True
