"""Stand-ins for the MLX classifier modules on a host without Apple's MLX (Linux, NVIDIA, Intel Mac).

The MLX strategy classes name their architecture at class level (``classifier_type =
ClassifierTypeMLX.LSTM_RAW``) and inherit ``MLXClassifierMixin.get_classifier``, so importing them
needs the MLX classifier module, which imports ``mlx``. Their TensorFlow twins (``<Class>_TF``)
subclass them, so the MLX classes must still IMPORT where MLX is absent. These stand-ins make
that possible; they never run a model:

* ``ClassifierTypeMLX.<NAME>`` is a placeholder string naming the MLX type;
* ``MLXClassifierMixin.get_classifier`` raises, pointing at the ``_TF`` twin.

A ``_TF`` class sets its own ``classifier_type`` and ``get_classifier``, so its model key never
reads a stand-in: it is the same on every host.
"""


class _AbsentTypes(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return f"{cls.__name__}.{name} (MLX not installed)"


class ClassifierTypeMLX(metaclass=_AbsentTypes):
    """Placeholder for NNNClassifierMLX / NNMTClassifierMLX ``ClassifierTypeMLX``."""


class MLXClassifierMixin:
    """Placeholder for the MLX ``get_classifier`` mixins (single- and multi-task)."""

    classifier_type = ClassifierTypeMLX.LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        raise RuntimeError(
            f"{type(self).__name__} needs Apple's MLX, which is not installed here; "
            f"run its TensorFlow twin ({type(self).__name__}_TF) instead"
        )


MLXMultiTaskClassifierMixin = MLXClassifierMixin
