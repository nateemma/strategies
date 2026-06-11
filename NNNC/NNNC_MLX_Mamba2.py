import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from NNNClassifierMLX import ClassifierTypeMLX

class NNNC_MLX_Mamba2(NNNC_MLX):

    classifier_type = ClassifierTypeMLX.Mamba2
