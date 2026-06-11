# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_MLX - Subclass of NNMTStrategy using MLX models
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple
import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402
from ClassifierKeras import ClassifierKeras
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, create_classifier_mlx

# -----------


class NNMT_MLX(NNMTStrategy):

    buy_params = { **NNMTStrategy.buy_params,
        "prediction_threshold": 0.5,
        }

    # ------------------------------------------------------------------ #
    # Classifier-side overrides — set on subclasses to push values onto   #
    # the MLX classifier instance after construction. ``None`` leaves the #
    # classifier's own default in place.                                  #
    # ------------------------------------------------------------------ #
    # Per-task loss-weight override (passed to clf.task_weights_override).
    # _CLASSIFIER_TASK_WEIGHTS = None
    _CLASSIFIER_TASK_WEIGHTS = {'trading': 4.0, 'regime': 2.0, 'risk': 1.0, 'momentum': 1.0, 'flow': 1.0, 'profit': 3.0}

    # Training-epoch ceiling override (passed to clf.max_epochs).
    _CLASSIFIER_MAX_EPOCHS = None
    # Entropy-penalty applied to softmax outputs during training (passed
    # to clf.entropy_penalty_weight). Pushes the model toward sharper,
    # more confident per-bar predictions. Useful for backbones that
    # naturally produce diffuse softmax (Attention/Transformer) — the
    # strategy's filter benefits from sharp outputs even at equal MCC.
    #
    # Accepts:
    #   * float                — same penalty across all six task heads.
    #   * Dict[str, float]     — per-task penalty keyed by task name.
    #     Missing tasks default to 0.0 (no pressure on that head). Use
    #     when the strategy depends on specific heads (e.g. trading +
    #     profit) and aux heads should be left to find their own
    #     calibration. Typical: {"trading": 0.1, "profit": 0.1}.
    # Default None disables. Useful values 0.01-0.5 per task.
    # _CLASSIFIER_ENTROPY_PENALTY = None
    _CLASSIFIER_ENTROPY_PENALTY = {"trading": 0.5, "profit": 0.5, "regime": 0.5}

    def _apply_classifier_overrides(self, clf) -> None:
        """Push strategy-level classifier overrides onto a freshly-built
        MLX classifier. Subclasses that swap classifier types or tune
        training-side knobs override ``_CLASSIFIER_*`` class attributes
        rather than re-implementing this helper."""
        if clf is None:
            return
        lr = getattr(self, "learning_rate", None)
        if lr is not None:
            clf.learning_rate = lr
        weights = getattr(self, "_CLASSIFIER_TASK_WEIGHTS", None)
        if weights:
            clf.task_weights_override = weights
        max_epochs = getattr(self, "_CLASSIFIER_MAX_EPOCHS", None)
        if max_epochs is not None:
            clf.max_epochs = int(max_epochs)
        entropy_penalty = getattr(self, "_CLASSIFIER_ENTROPY_PENALTY", None)
        if entropy_penalty is not None:
            # Pass dict through unchanged so the classifier can route
            # per-task; coerce scalars to float for the uniform form.
            if isinstance(entropy_penalty, dict):
                clf.entropy_penalty_weight = dict(entropy_penalty)
            else:
                clf.entropy_penalty_weight = float(entropy_penalty)

    # default is LSTM type. Override get_classifier_type() in subclass
    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return ClassifierTypeMLX.LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> ClassifierKeras:
        """Return the classifier used for training/predicting.

        The multi-task MLX factory does not take an ``nclasses`` argument —
        each of the six task heads emits a fixed 3-way softmax — so the call
        site here matches the Keras MT factory (NNMTClassifier.create_classifier)
        rather than the single-task NNNClassifierMLX one.
        """
        if hasattr(mx, "metal") and mx.metal.is_available():
            clf, _ = create_classifier_mlx(
                classifier_type, pair, num_features, seq_len
            )
            self._apply_classifier_overrides(clf)
        else:
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            clf = None
        return clf
