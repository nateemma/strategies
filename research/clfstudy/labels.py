"""Labels: class-attribute shim over `BaseNNStrategy`'s own methods (research R2).

`_LabelShim` reads the label-relevant class attributes off `NNNC.NNNC_MLX.NNNC_MLX`
(the class object -- never instantiated, so no Parameter singletons / config coupling)
and binds the five label-producing methods from whichever class in `NNNC_MLX.__mro__`
actually defines them (`getattr(NNNC_MLX, name)` walks the MRO, so a mixin override
would be honoured). This makes drift between the study's labels and production labels
structurally impossible: change the strategy class attribute or override a method, and
the shim picks it up automatically. This is the `{"name": "gbb"}` (default) label path
and is completely unchanged by everything below it in this module.

Do NOT reimplement the label logic here (forbidden by FR-007) -- only bind to it.

`ternary_labels(df, label=...)` additionally supports `{"name": "triple_barrier", ...}`
(feature: let an experiment SELECT the label). See `_TripleBarrierShim` for why that
path binds its own `get_train_buy_signals`/`get_train_sell_signals` instead of NNNC_MLX's
(a pre-existing `Framework/TrainingSignals.py` gap, documented on the class).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from Framework import TrainingSignals
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

#: methods shared by BOTH label paths (bound unchanged from NNNC_MLX's MRO); the
#: triple_barrier path additionally overrides get_train_buy_signals/get_train_sell_signals
#: (and, when peak_filter=False, filter_peaks_by_future_performance) on the instance.
_SHARED_METHODS = (
    "filter_peaks_by_future_performance",
    "augment_training_signals",
    "get_training_labels",
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


class _TripleBarrierShim:
    """Triple-barrier ternary labels, sharing `get_training_labels`/
    `augment_training_signals`/`print_distribution_compact` with `_LabelShim` (bound
    unchanged from the same `NNNC_MLX` MRO -- `_SHARED_METHODS`) -- only signal
    generation differs.

    `BaseNNStrategy.get_train_buy_signals` unconditionally adds a `rvol_threshold` kwarg
    to the params dict it passes to `TrainingSignals.get_train_buy_signals` (see
    `Framework/BaseNNStrategy.py`); `TrainingSignals.labels_triple_barrier` does not
    accept that kwarg and raises `TypeError` (confirmed empirically -- this is a
    pre-existing gap in `Framework/TrainingSignals.py`'s buy-side dispatch, which,
    unlike its sell-side counterpart, does not whitelist params per method; out of
    scope to fix here, `Framework/` is untouched). So this shim's own
    `get_train_buy_signals`/`get_train_sell_signals` call
    `TrainingSignals.get_train_buy_signals`/`get_train_sell_signals` (the production
    label DISPATCHER -- never reimplemented) with only the kwargs
    `labels_triple_barrier`/`labels_triple_barrier_sell` actually accept. Structurally
    otherwise identical to `BaseNNStrategy`'s own methods: same peak-filter window,
    same zero/one `Series` shape.

    `peak_filter=False` (the label default) bypasses `filter_peaks_by_future_performance`
    entirely (peaks pass through unchanged) -- the gbb path above keeps that filter,
    bound unchanged from NNNC_MLX, exactly as production runs it today.
    """

    dbg_verbose: bool = False

    def __init__(self, horizon: int, up: float, down: float,
                 conflict_resolution: bool, peak_filter: bool) -> None:
        self.TRAINING_TYPE = TrainingSignals.LabelMethod.triple_barrier
        self.HORIZON = horizon
        self.MIN_BUY_GAIN_THRESHOLD = up
        self.MIN_SELL_LOSS_THRESHOLD = down
        self.LABEL_FILTER_WINDOW = horizon
        self.apply_label_conflict_resolution = conflict_resolution
        self.dbg_curr_df = None
        for name in _SHARED_METHODS:
            setattr(self, name, getattr(NNNC_MLX, name).__get__(self))
        if not peak_filter:
            self.filter_peaks_by_future_performance = self._passthrough_filter

    @staticmethod
    def _passthrough_filter(future_df, peaks, signal_type="buy", window=64):
        return list(np.asarray(peaks, dtype=int))

    def debug_print(self, msg: str) -> None:  # pragma: no cover - never called
        return None

    def get_train_buy_signals(self, future_df: pd.DataFrame) -> pd.Series:
        labels = TrainingSignals.get_train_buy_signals(
            future_df, method=self.TRAINING_TYPE,
            params={
                "horizon": self.HORIZON,
                "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
                "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
            },
        )
        peaks = np.where(labels.values > 0.5)[0]
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "buy", window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))
        signals[filtered_peaks] = 1
        return signals

    def get_train_sell_signals(self, future_df: pd.DataFrame) -> pd.Series:
        labels = TrainingSignals.get_train_sell_signals(
            future_df, method=self.TRAINING_TYPE,
            params={
                "horizon": self.HORIZON,
                "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
                "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
            },
        )
        peaks = np.where(labels.values > 0.5)[0]
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "sell", window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float), dtype=float)
        if len(filtered_peaks) > 0:
            signals.iloc[filtered_peaks] = 1.0
        return signals


#: `label` dict default -- means "exactly the production gbb label, unchanged"; kept in
#: sync with `registry.DEFAULT_LABEL` but defined locally so this module has no
#: dependency on registry.py (data.py imports labels.py, not the other way round).
_DEFAULT_LABEL: Dict[str, Any] = {"name": "gbb"}


def label_params() -> dict:
    """The shim's attribute values, for the report (research R2 fidelity check)."""
    shim = _LabelShim()
    return {name: getattr(shim, name) for name in _SHIM_ATTRS}


def label_key(label: Optional[Dict[str, Any]] = None) -> str:
    """Short, stable string for `label`, for cache/artifact naming.

    `None` or `{"name": "gbb"}` -> `"gbb"`. `{"name": "triple_barrier", "horizon": 48,
    "up": 0.02, "down": 0.02}` -> `"tb_h48_u0.02_d0.02_cr1_pf0"` (conflict_resolution and
    peak_filter default True/False per `registry._LABEL_TRIPLE_BARRIER_DEFAULTS`).
    """
    label = label or _DEFAULT_LABEL
    name = label.get("name", "gbb")
    if name == "gbb":
        return "gbb"
    horizon = label["horizon"]
    up = label["up"]
    down = label["down"]
    cr = int(bool(label.get("conflict_resolution", True)))
    pf = int(bool(label.get("peak_filter", False)))
    return f"tb_h{horizon}_u{up:g}_d{down:g}_cr{cr}_pf{pf}"


def ternary_labels(df: pd.DataFrame, label: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """SELL=0/HOLD=1/BUY=2 labels for `df`.

    `label=None` or `{"name": "gbb"}` (default): produced by `NNNC_MLX`'s own label
    methods, byte-identical to before this parameter existed.

    `{"name": "triple_barrier", "horizon": int, "up": float, "down": float,
    "conflict_resolution": bool (default True), "peak_filter": bool (default False)}`:
    produced by `TrainingSignals.LabelMethod.triple_barrier` via `_TripleBarrierShim`.

    Either shim's `get_training_labels` mutates its dataframe argument (adds
    %train_buy/%train_sell), so this always runs on a copy -- `df` itself is never
    modified.
    """
    label = label or _DEFAULT_LABEL
    name = label.get("name", "gbb")

    if name == "gbb":
        shim: Any = _LabelShim()
    elif name == "triple_barrier":
        shim = _TripleBarrierShim(
            horizon=label["horizon"],
            up=label["up"],
            down=label["down"],
            conflict_resolution=label.get("conflict_resolution", True),
            peak_filter=label.get("peak_filter", False),
        )
    else:
        raise ValueError(f"unknown label {name!r}; valid: gbb, triple_barrier")

    working = df.copy()
    trading_classes = shim.get_training_labels(working)
    return np.asarray(trading_classes, dtype=np.int8)
