# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""TensorFlow twins of NNNC's MLX classifiers (NNNClassifierMLX), for hosts without Apple's MLX.

Only the default model is ported: LSTM_RAW (NNNClassifierMLX_LSTM_Raw), the network ARO / cs_aro use.
Same architecture, initialisers and training loop (Predictors/TFPortClassifierNary.py):

    input (B, seq_len, F) -> LSTM(F -> nf) -> last step -> LayerNorm -> Dropout(0.25)
      -> Dense(nf, elu) -> Dropout(0.25) -> Dense(3, softmax),   nf = max(32, 2 ** (F - 1).bit_length())

``load_mlx_weights`` copies a trained MLX model's weights into the Keras model, so a converted model
predicts what the MLX one does (NNNC/tests/test_tf_port_nnnc.py; scripts/convert_mlx_to_tf.py).
A strategy runs on TensorFlow by putting ``TFClassifierMixin`` first in its bases (e.g. NNNC_CS_ARO_TF).
"""

import sys
from enum import Enum
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import keras  # noqa: E402

from Predictors.TFPortBasePredictor import (  # noqa: E402
    read_safetensors, set_mlx_layernorm, set_mlx_linear, set_mlx_lstm, uniform_init)
from Predictors.TFPortClassifierNary import TFPortClassifierNary  # noqa: E402

DROPOUT = 0.25  # NNNClassifierMLX.DROPOUT
MIN_FILTER = 32


def num_filters_for(num_features: int) -> int:
    """NNNClassifierMLX: max(MIN_FILTER, nearest_power_of_2(num_features - 1))."""
    return max(MIN_FILTER, 2 ** (num_features - 1).bit_length())


def lstm_raw_trunk(x, num_features: int, nf: int, ln_name: str):
    """LSTM(F -> nf) on the raw inputs -> last step -> LayerNorm (eps 1e-5, as MLX) -> Dropout."""
    s = 1.0 / nf ** 0.5
    h = keras.layers.LSTM(nf, kernel_initializer=uniform_init(s), recurrent_initializer=uniform_init(s),
                          bias_initializer=uniform_init(s), unit_forget_bias=False, name="lstm")(x)
    h = keras.layers.LayerNormalization(epsilon=1e-5, name=ln_name)(h)
    return keras.layers.Dropout(DROPOUT)(h)


def decode_head(h, nf: int, num_classes: int, prefix: str = ""):
    """ARO's decode: Dense(nf, elu) -> Dropout -> Dense(num_classes, softmax); MLX Linear init U(+-1/sqrt(in))."""
    s = 1.0 / nf ** 0.5
    h = keras.layers.Dense(nf, activation="elu", kernel_initializer=uniform_init(s), bias_initializer=uniform_init(s),
                           name=f"{prefix}bottleneck")(h)
    h = keras.layers.Dropout(DROPOUT)(h)
    return keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=uniform_init(s),
                              bias_initializer=uniform_init(s), name=f"{prefix}out")(h)


class NNNClassifierTF_LSTM_Raw(TFPortClassifierNary):
    """TensorFlow twin of NNNClassifierMLX_LSTM_Raw."""

    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> keras.Model:
        nf = num_filters_for(num_features)
        inp = keras.Input((seq_len, num_features), name="x")
        out = decode_head(lstm_raw_trunk(inp, num_features, nf, "ln_lstm"), nf, self.num_classes)
        return keras.Model(inp, out, name=self.__class__.__name__)

    @staticmethod
    def load_mlx_weights(model: keras.Model, path) -> None:
        """Copy NNNClassifierMLX_LSTM_Raw weights (.safetensors) into `model`. The MLX model also stores
        the unused resize / conv1 / ln_conv of its parent class; they do not take part in the forward pass."""
        w = read_safetensors(path)
        set_mlx_lstm(model.get_layer("lstm"), w, "lstm")
        set_mlx_layernorm(model.get_layer("ln_lstm"), w, "ln_lstm")
        set_mlx_linear(model.get_layer("bottleneck"), w, "bottleneck")
        set_mlx_linear(model.get_layer("out"), w, "out")


class ClassifierTypeTF(Enum):
    LSTM_RAW = NNNClassifierTF_LSTM_Raw


def create_classifier_tf(clf_type: ClassifierTypeTF, pair: str, nfeatures: int, seq_len: int, nclasses: int,
                         tag: str = ""):
    """create_classifier_mlx's interface: returns (classifier, name)."""
    clf = clf_type.value(pair, seq_len, nfeatures, tag=tag)
    clf.num_classes = nclasses
    return clf, clf_type.name


class TFClassifierMixin:
    """Runs an NNNC strategy's model on TensorFlow: put it FIRST in the bases of a twin class
    (``class X_TF(TFClassifierMixin, X)``). The twin keeps every other setting of X; its model key differs
    from X's (classifier_type and get_classifier differ), so it never loads X's MLX weights."""

    classifier_type = ClassifierTypeTF.LSTM_RAW
    # BaseNNStrategy's default is 3 with MLX, 0 without, and it is in the model key; method 0 also drops
    # the first seq_len-1 rows. Method 1 builds exactly method 3's tensor, in numpy, on every host.
    tensor_method = 1

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf, _ = create_classifier_tf(classifier_type, pair, num_features, seq_len, 3)
        return clf
