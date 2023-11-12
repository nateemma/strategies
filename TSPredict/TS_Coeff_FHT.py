#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_FHT - use a Fast Hankel Transform model

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scipy.fft import fht

from TS_Coeff import TS_Coeff

class TS_Coeff_FHT(TS_Coeff):

    ###################################

 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        n = len(data)
                
        # # de-trend the data
        # w_mean = data.mean()
        # w_std = data.std()
        # x = (data - w_mean) / w_std

        x = np.nan_to_num(data)

        # compute the fft
        dln = np.log(x[1]/x[0]) 
        features = fht(x, mu=0.0, dln=dln)

        # print(f'features: {features}')

        return features

    #-------------

