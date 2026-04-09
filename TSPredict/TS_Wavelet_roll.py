# pragma pylint: disable=W0105, C0103, C0115, C0116, C0301, C0411, C0413,  W1203



import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__)))


import utils.Wavelets as Wavelets
import utils.Forecasters as Forecasters
from TS_Wavelet_DWT import TS_Wavelet_DWT


# this class is intended to experiment with global settings (without needing to change the base classes)

class TS_Wavelet_roll(TS_Wavelet_DWT):


    use_rolling = True
    single_col_prediction = True

