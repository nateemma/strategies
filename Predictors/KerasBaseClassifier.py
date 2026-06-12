"""
KerasBaseClassifier - Keras-backed marker for classifier predictors.

Pure marker class. Concrete Keras classifier predictors inherit from this
PLUS the corresponding utils.Classifier* class via multiple inheritance --
that is where the actual Keras behavior comes from.

This class deliberately does NOT inherit from utils.ClassifierKeras.
Doing so would put utils.ClassifierKeras in the MRO twice (once via
this base, once via the concrete class's other parent), and because
utils/ClassifierKerasMultiTask.py uses a sibling-style import that
loads ClassifierKeras under a *second* module name, the two copies
resolve to different class objects. The C3 linearization then
interleaves them in a way that breaks the cooperative super().__init__()
chain in utils.ClassifierKeras at line 70.

Keeping this class as a pure marker sidesteps the diamond entirely:
the concrete class's MRO contains exactly one ClassifierKeras (via the
utils.Classifier* parent), super() chains work as designed, and
isinstance(x, KerasBaseClassifier) still succeeds.
"""

from Predictors.BaseClassifier import BaseClassifier


class KerasBaseClassifier(BaseClassifier):
    """Marker for Keras-backed classifier predictors. Carries no behavior on its own."""
    pass
