import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_CGP_MLX import NNNC_CGP_MLX
from NNNClassifierMLX import ClassifierTypeMLX

class NNNC_CGP_MLX_MLP(NNNC_CGP_MLX):

    classifier_type = ClassifierTypeMLX.MLP
