# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""RETIRED in feature 030 -- weights are found by model key, not by a class-name redirect.

This mixin used to point a per-fold class `<Name>_fNN` at `<MODEL_SOURCE>_fNN`'s saved model
(023 R12; extracted in 029), because `get_model_path` keyed off the class name. Since 030,
`Framework/ModelKey.py` resolves weights from everything that shapes them, so a class that differs
from its source only after the model (exits, gates, triggers) resolves to the source's key with no
redirect. A class whose run differs in a training input the weights did not see (e.g. a 4096-bar
warm-up on weights trained with 64) declares `MODEL_TRAINED_WITH`.

The module stays so an old import fails with this explanation instead of an ImportError.
"""


class BorrowedModelMixin:
    def __init_subclass__(cls, **kwargs):
        raise TypeError(
            f"{cls.__name__}: BorrowedModelMixin is retired (030). Drop it -- weights resolve by "
            f"model key; declare MODEL_TRAINED_WITH for a training input your run changes.")
