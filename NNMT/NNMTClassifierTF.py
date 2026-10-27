# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""TensorFlow twins of NNMT's MLX multi-task classifiers (NNMTClassifierMLX), for hosts without MLX.

Only the default model is ported: LSTM_RAW (NNMTClassifierMLX_LSTM_Raw), the network NNMT_MLX = NNMT_LR_RI
uses. Same architecture, initialisers and training loop (Predictors/TFPortClassifierMultiTask.py):

    input (B, seq_len, F) -> LSTM(F -> nf) -> last step -> LayerNorm -> Dropout(0.25)
      -> per task (trading, regime, risk, momentum, flow, profit): Dense(nf, elu) -> Dropout -> Dense(3, softmax)

``load_mlx_weights`` copies a trained MLX model's weights into the Keras model (NNMT/tests/test_tf_port_nnmt.py;
scripts/convert_mlx_to_tf.py). A strategy runs on TensorFlow by putting ``TFMultiTaskClassifierMixin``
first in its bases (e.g. NNMT_LR_RI_TF).
"""

import sys
from enum import Enum
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import keras  # noqa: E402

from NNNC.NNNClassifierTF import decode_head, lstm_raw_trunk, num_filters_for  # noqa: E402
from Predictors.TFPortBasePredictor import read_safetensors, set_mlx_layernorm, set_mlx_linear, set_mlx_lstm  # noqa: E402
from Predictors.TFPortClassifierMultiTask import TASK_NAMES, TFPortClassifierMultiTask  # noqa: E402


class NNMTClassifierTF_LSTM_Raw(TFPortClassifierMultiTask):
    """TensorFlow twin of NNMTClassifierMLX_LSTM_Raw (its RAW_DROPOUT is 0.25, as here)."""

    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> keras.Model:
        nf = num_filters_for(num_features)
        inp = keras.Input((seq_len, num_features), name="x")
        h = lstm_raw_trunk(inp, num_features, nf, "ln")
        outs = {t: decode_head(h, nf, 3, prefix=f"{t}_") for t in TASK_NAMES}
        return keras.Model(inp, outs, name=self.__class__.__name__)

    @staticmethod
    def load_mlx_weights(model: keras.Model, path) -> None:
        """Copy NNMTClassifierMLX_LSTM_Raw weights (.safetensors) into `model`; MLX's task_heads.<i>
        follow TASK_NAMES order."""
        w = read_safetensors(path)
        set_mlx_lstm(model.get_layer("lstm"), w, "lstm")
        set_mlx_layernorm(model.get_layer("ln"), w, "ln")
        for i, t in enumerate(TASK_NAMES):
            set_mlx_linear(model.get_layer(f"{t}_bottleneck"), w, f"task_heads.{i}.bottleneck")
            set_mlx_linear(model.get_layer(f"{t}_out"), w, f"task_heads.{i}.out")


class ClassifierTypeTF(Enum):
    LSTM_RAW = NNMTClassifierTF_LSTM_Raw


def create_classifier_tf(clf_type: ClassifierTypeTF, pair: str, nfeatures: int, seq_len: int, *, tag: str = ""):
    """NNMTClassifierMLX.create_classifier_mlx's interface: returns (classifier, name)."""
    return clf_type.value(pair, seq_len, nfeatures, tag=tag), clf_type.name


class TFMultiTaskClassifierMixin:
    """Runs an NNMT strategy's model on TensorFlow: put it FIRST in the bases of a twin class
    (``class X_TF(TFMultiTaskClassifierMixin, X)``). Applies the strategy's classifier overrides (task
    weights, epochs, entropy penalty, learning rate) as the MLX mixin does."""

    classifier_type = ClassifierTypeTF.LSTM_RAW
    # BaseNNStrategy's default is 3 with MLX, 0 without, and it is in the model key; method 0 also drops
    # the first seq_len-1 rows. Method 1 builds exactly method 3's tensor, in numpy, on every host.
    tensor_method = 1

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf, _ = create_classifier_tf(classifier_type, pair, num_features, seq_len)
        self._apply_classifier_overrides(clf)
        return clf
