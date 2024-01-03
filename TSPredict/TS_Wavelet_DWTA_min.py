# pragma pylint: disable=W0105, C0103, C0115, C0116, C0301, C0411, C0413,  W1203


"""
####################################################################################
TS_Wavelet_DWTA_min - use a Digital Wave Transform model (Approximate Coefficients)

                This is a 'stripped down' version of TS_Wavelet_DWT where we use
                only the 'approximate' coefficients of the transform.
                This greatly reduces the number of coefficients to fit, so it runs
                way faster than the normal version

                This is an even more minimalist version where all (added) indicators except 
                gain are removed. This is mostly to serve as a benchmark for the other TS_Wavelet
                variants, since there should not be any lookahead possible)

####################################################################################
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))

import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters

from TS_Wavelet_DWTA import TS_Wavelet_DWTA

class TS_Wavelet_DWTA_min(TS_Wavelet_DWTA):

    ###################################

    merge_indicators = True
    use_rolling = True
    single_col_prediction = True

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
