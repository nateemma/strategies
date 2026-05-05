"""
MLXBaseClassifier - MLX-backed marker for classifier predictors.

Pure marker class. Concrete MLX classifier predictors (e.g.
MLXClassifierNary, MLXClassifierMultiTask) inherit from this PLUS the
corresponding utils.Classifier* class via multiple inheritance -- that
is where the actual MLX behavior comes from.

See KerasBaseClassifier for the design rationale (avoiding a broken
diamond MRO caused by sibling-style imports in utils/).
"""

from Predictors.BaseClassifier import BaseClassifier


class MLXBaseClassifier(BaseClassifier):
    """Marker for MLX-backed classifier predictors. Carries no behavior on its own."""
    pass
