# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_CGP - Subclass of NNNCStrategy using CTAB-GAN+ for high-fidelity augmentation
"""

import sys
import traceback
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Tuple
import os

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_CGP import NNNC_CGP
import NNNClassifier


class NNNC_CGP_Transformer(NNNC_CGP):


    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return NNNClassifier.ClassifierType.Transformer
