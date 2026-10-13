# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Borrowed per-fold weights, never trained (feature 023 R12; extracted in 029).

A per-fold class `<Name>_fNN` loads `<MODEL_SOURCE>_fNN`'s saved model instead of its own.
`get_model_path` keys off the class name in the base, so without this redirect a fold class would
find no model and TRAIN one inside its own backtest. `train_model` raises, so a missing model is a
loud failure, never a silent retrain. Used by the equivalence twin, the 029 stop-invariant probe
and the 029 liveness arms.
"""

import re
from pathlib import Path


_FOLD = re.compile(r"_(f\d{2})$")


class BorrowedModelMixin:
    MODEL_SOURCE = "NNNC_ARO"

    # Feature 030: the model key (Framework/ModelKey.py). This class declares the names it
    # owns; the lists merge along the MRO. ATTRIBUTES/METHODS shape trained weights;
    # IGNORED act only after the model. The audit test keeps every name classified.
    MODEL_KEY_IGNORED = (
        "MODEL_SOURCE",
    )

    def get_model_path(self) -> str:
        m = _FOLD.search(self.__class__.__name__)
        if not m:
            raise ValueError(
                f"{self.__class__.__name__}: runs only as a per-fold class (`*_fNN`) that "
                f"borrows {self.MODEL_SOURCE}_fNN's weights"
            )
        src = f"{self.MODEL_SOURCE}_{m.group(1)}"
        d = Path(self.get_storage_location()) / src
        if not d.is_dir():
            raise FileNotFoundError(f"borrowed model dir {d} is missing -- refusing to train")
        return str(d / f"{src}.keras")

    def train_model(self, *args, **kwargs):
        raise RuntimeError(f"{self.__class__.__name__} must never train")
