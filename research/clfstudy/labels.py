"""Labels: class-attribute shim over `BaseNNStrategy`'s own methods (research R2).

`_LabelShim` reads the label-relevant class attributes off `NNNC.NNNC_MLX.NNNC_MLX`
(the class object -- never instantiated, so no Parameter singletons / config coupling)
and binds the five label-producing methods from whichever class in `NNNC_MLX.__mro__`
actually defines them (`getattr(NNNC_MLX, name)` walks the MRO, so a mixin override
would be honoured). This makes drift between the study's labels and production labels
structurally impossible: change the strategy class attribute or override a method, and
the shim picks it up automatically.

Do NOT reimplement the label logic here (forbidden by FR-007) -- only bind to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from NNNC.NNNC_MLX import NNNC_MLX

#: attributes the shim reads off NNNC_MLX (research R2); test_labels_shim.py asserts
#: this list matches what NNNC_MLX's own label methods actually touch.
_SHIM_ATTRS = (
    "HORIZON",
    "MIN_BUY_GAIN_THRESHOLD",
    "MIN_SELL_LOSS_THRESHOLD",
    "TRAINING_TYPE",
    "LABEL_RVOL_THRESHOLD",
    "LABEL_FILTER_WINDOW",
    "apply_label_conflict_resolution",
)

#: methods bound from NNNC_MLX's MRO (unbound function, called with the shim as self)
_SHIM_METHODS = (
    "get_train_buy_signals",
    "get_train_sell_signals",
    "filter_peaks_by_future_performance",
    "augment_training_signals",
    "get_training_labels",
    # Not itself a label method, but `get_training_labels` calls
    # `self.print_distribution_compact(...)` UNCONDITIONALLY (StrategyDiagnostics,
    # inherited via BaseStrategy) -- without binding it too, the shim would raise
    # AttributeError on every call.
    "print_distribution_compact",
)


class _LabelShim:
    """A minimal stand-in for `NNNC_MLX` that exposes only what its label methods read.

    No `IStrategy` machinery (no `self.dp`, no config, no Parameter singletons) --
    `dbg_verbose=False` keeps every `debug_print` call a no-op so none of it is ever
    needed.
    """

    dbg_verbose: bool = False

    def __init__(self) -> None:
        for name in _SHIM_ATTRS:
            setattr(self, name, getattr(NNNC_MLX, name))
        self.dbg_curr_df = None
        for name in _SHIM_METHODS:
            setattr(self, name, getattr(NNNC_MLX, name).__get__(self))

    def debug_print(self, msg: str) -> None:  # pragma: no cover - never called
        """No-op: dbg_verbose=False already short-circuits every debug_print call
        in BaseNNStrategy/BaseStrategy, but methods called directly (bypassing that
        guard) would otherwise touch the never-set `self.dp`."""
        return None


def label_params() -> dict:
    """The shim's attribute values, for the report (research R2 fidelity check)."""
    shim = _LabelShim()
    return {name: getattr(shim, name) for name in _SHIM_ATTRS}


def ternary_labels(df: pd.DataFrame) -> np.ndarray:
    """SELL=0/HOLD=1/BUY=2 labels for `df`, produced by `NNNC_MLX`'s own label methods.

    `get_training_labels` mutates its dataframe argument (adds %train_buy/%train_sell),
    so this always runs on a copy -- `df` itself is never modified.
    """
    shim = _LabelShim()
    working = df.copy()
    trading_classes = shim.get_training_labels(working)
    return np.asarray(trading_classes, dtype=np.int8)
