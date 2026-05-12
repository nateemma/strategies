# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621
# type: ignore
# pylint: disable=import-error

"""
NNMT_DDPM_MLX_MultiLSTM - Subclass of NNMT_DDPM using MLX MultiLSTM classifier.

Case A applies: NNMT_WGAN hardcodes GANType.MT_WGAN in both the class attribute
and the GANInterface constructor call in preprocess_training_data.  A simple
attribute override is therefore insufficient; NNMT_DDPM.py is the sibling base
that replaces both references with GANType.MT_DDPM.

This class only overrides the classifier factory methods.
"""

import sys
from pathlib import Path
import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_DDPM import NNMT_DDPM
from ClassifierKeras import ClassifierKeras
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, create_classifier_mlx


class NNMT_DDPM_MLX_MultiLSTM(NNMT_DDPM):

    def get_classifier_type(self):
        return ClassifierTypeMLX.Multi_LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features) -> ClassifierKeras:
        if hasattr(mx, "metal") and mx.metal.is_available():
            clf, _ = create_classifier_mlx(classifier_type, pair, num_features, seq_len)
        else:
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            clf = None
        return clf
