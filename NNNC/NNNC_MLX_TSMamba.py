import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from NNNClassifierMLX import ClassifierTypeMLX

class NNNC_MLX_TSMamba(NNNC_MLX):

    # default is LSTM type. Override get_classifier_type() in subclass
    def get_classifier_type(self):
        return ClassifierTypeMLX.TSMamba
