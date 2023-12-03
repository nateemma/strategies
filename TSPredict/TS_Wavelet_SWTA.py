#pragma pylint: disable=W0105, C0103, C0301, W1203


"""
####################################################################################
TS_Wavelet_SWTA - use a Standing Wave Transform model, Approxinate coefficients only

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pywt


from sklearn.preprocessing import RobustScaler

from TS_Wavelet import TS_Wavelet

class TS_Wavelet_SWTA(TS_Wavelet):

    ###################################

    save_coeffs = None
 
    # function to get dwt coefficients
    def get_coeffs(self, data: np.array) -> np.array:

        length = len(data)

        # # de-trend the data
        # w_mean = data.mean()
        # w_std = data.std()
        # x = (data - w_mean) / w_std

        x = data

        # data must be of even length, so trim if necessary
        if (len(x) % 2) != 0:
            x = x[1:]

        # self.wavelet = 'db4'

        self.wavelet = 'bior3.9'

        self.coeff_format = "wavedec"

        # (cA2, cD2), (cA1, cD1) = pywt.swt(data, wavelet, level=2)
        


        # if we set trim_approx=True, we only get the detail coeffs. 
        # So, since we are trying to reduce the number of coerffs do that and set level=1

        coeffs = pywt.swt(x, self.wavelet, level=1, trim_approx=True)

        return self.coeff_to_array(coeffs)


    # -------------

    # override the get_data func so that only the gain data (and default data) are included
    # This should ensure that there is no lookahead bias
    def get_data(self, dataframe):

        col_list = ['date', 'open', 'close', 'high', 'low', 'volume', 'gain']
        # col_list = ['date', 'gain']
        df = dataframe[col_list].copy()
        df_norm = self.convert_dataframe(df)
        # df_norm = self.convert_dataframe(dataframe)
        gain_data = df_norm['gain'].to_numpy()
        self.build_coefficient_table(gain_data)
        data = self.merge_coeff_table(df_norm)
        return data
    
    def get_value(self, coeffs):
        # series = pywt.waverec(coeffs, self.wavelet)

        # print(f'    coeffs: {coeffs}')

        series = pywt.iswt(coeffs, wavelet=self.wavelet)
        # print(f'    coeff_slices:{self.coeff_slices}, coeff_shapes:{self.coeff_shapes} series:{np.shape(series)}')

        # de-norm
        scaler = RobustScaler()
        scaler.fit(self.gain_data.reshape(-1, 1))
        denorm_series = scaler.inverse_transform(series.reshape(-1, 1)).squeeze()
        return denorm_series
    
    # -------------
