#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Coeff_DCT - use a Discreet Cosine Transform model

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scipy.fft import dct

from TS_Coeff import TS_Coeff

class TS_Coeff_DCT(TS_Coeff):

    ###################################

 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        
        # compute the dct
        features = dct(data)

        # print(f'features: {features}')

        return features

    #-------------

