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

`ternary_labels` also supports any other PRICE-ONLY `Framework.TrainingSignals.LabelMethod`
(`GENERIC_LABEL_NAMES`) via `_GenericLabelShim`/`ternary_labels_generic`, generalising
`_TripleBarrierShim` to methods with heterogeneous signatures: rather than hand-written
param names per method, it inspects each labeler's own signature (`inspect.signature`) and
passes only the argument names it accepts. This class + function are also the canonical
home imported by `research.clfstudy.labeltriage` (a training-free triage CLI over the same
methods) -- moved here (rather than kept local to that CLI) since experiment configs are
the more natural, permanent caller.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from Framework import TrainingSignals
from NNNC.NNNC_MLX import NNNC_MLX

LabelMethod = TrainingSignals.LabelMethod

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

    `atr_scale` (optional) is passed straight through in both params dicts when set: the
    sell-side whitelist above (`TrainingSignals.get_train_sell_signals`'s
    `LabelMethod.triple_barrier` branch) already includes `"atr_scale"`, and the buy-side
    call goes directly to `TrainingSignals.get_train_buy_signals` (not
    `BaseNNStrategy.get_train_buy_signals`), which forwards every param -- so, unlike
    `rvol_threshold` above, there is no landmine to route around for `atr_scale`.

    `pt`/`sl` are passed EXPLICITLY (not only via `min_gain`/`min_loss`): both
    `labels_triple_barrier` and `labels_triple_barrier_sell` default `pt=0.03, sl=0.02`
    and only ever raise a barrier via `pt = max(pt, min_gain)` / `sl = max(sl, min_loss)`
    -- so relying on `min_gain`/`min_loss` alone made those defaults a FLOOR a narrower
    `up`/`down` could never go below (and `labels_triple_barrier` ignores `min_loss`
    entirely, `labels_triple_barrier_sell` ignores `min_gain` entirely, per their own
    docstrings). Passing `pt=up, sl=down` directly sets the actual barrier; `min_gain`/
    `min_loss` are still passed too (so the `max()` each function applies is a no-op),
    which is also what makes `atr_scale`'s `max(pt, atr_scale_width)` /
    `max(sl, atr_scale_width)` compare against the CONFIGURED width rather than the
    0.03/0.02 defaults.
    """

    dbg_verbose: bool = False

    def __init__(self, horizon: int, up: float, down: float,
                 conflict_resolution: bool, peak_filter: bool,
                 atr_scale: Optional[float] = None) -> None:
        self.TRAINING_TYPE = TrainingSignals.LabelMethod.triple_barrier
        self.HORIZON = horizon
        self.MIN_BUY_GAIN_THRESHOLD = up
        self.MIN_SELL_LOSS_THRESHOLD = down
        self.LABEL_FILTER_WINDOW = horizon
        self.apply_label_conflict_resolution = conflict_resolution
        self.ATR_SCALE = atr_scale
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
        params = {
            "horizon": self.HORIZON,
            "pt": self.MIN_BUY_GAIN_THRESHOLD,
            "sl": self.MIN_SELL_LOSS_THRESHOLD,
            "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
            "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
        }
        if self.ATR_SCALE is not None:
            params["atr_scale"] = self.ATR_SCALE
        labels = TrainingSignals.get_train_buy_signals(
            future_df, method=self.TRAINING_TYPE, params=params,
        )
        peaks = np.where(labels.values > 0.5)[0]
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "buy", window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))
        signals[filtered_peaks] = 1
        return signals

    def get_train_sell_signals(self, future_df: pd.DataFrame) -> pd.Series:
        params = {
            "horizon": self.HORIZON,
            "pt": self.MIN_BUY_GAIN_THRESHOLD,
            "sl": self.MIN_SELL_LOSS_THRESHOLD,
            "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
            "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
        }
        if self.ATR_SCALE is not None:
            params["atr_scale"] = self.ATR_SCALE
        labels = TrainingSignals.get_train_sell_signals(
            future_df, method=self.TRAINING_TYPE, params=params,
        )
        peaks = np.where(labels.values > 0.5)[0]
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "sell", window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float), dtype=float)
        if len(filtered_peaks) > 0:
            signals.iloc[filtered_peaks] = 1.0
        return signals


#: PRICE-ONLY `LabelMethod` names routed through `_GenericLabelShim` (every name in this
#: module's supported set except the two with dedicated paths above, `gbb` and
#: `triple_barrier`). Indicator-based methods (technical_indicators, trends, indicators2/3/4,
#: bands, breakout*, gbb_path) read columns (guard_metric, adx, rsi, ...) this study's price-
#: only label selection does not support.
GENERIC_LABEL_NAMES: Tuple[str, ...] = (
    "forward_mae",
    "quantile_future",
    "mfe_mae_ratio",
    "local_min_followthrough",
    "risk_adj_future",
    "multi_horizon_vote",
    "optimal_signals",
    "future_sharpe",
    "future_sortino",
    "future_expectancy",
    "local_extrema",
    "geometry",
)

#: defaults for `min_gain`/`min_loss` when a generic-name label omits them (matches every
#: `TrainingSignals.labels_*` buy function's own default of 0.02).
_GENERIC_DEFAULT_MIN_GAIN = 0.02
_GENERIC_DEFAULT_MIN_LOSS = 0.02


def _generic_passthrough_filter(future_df: pd.DataFrame, peaks, signal_type: str = "buy", window: int = 64):
    """Peak filter bypass (peak_filter=False): peaks pass through unchanged, matching
    `_TripleBarrierShim._passthrough_filter` above (duplicated, not shared, since that one
    is a private staticmethod of an unrelated class)."""
    return list(np.asarray(peaks, dtype=int))


def _filter_kwargs(fn, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keys of `params` that `fn`'s signature actually accepts (excluding `df`)."""
    accepted = set(inspect.signature(fn).parameters) - {"df"}
    return {k: v for k, v in params.items() if k in accepted}


#: per-method extra kwargs, applied on top of {horizon, min_gain, min_loss} before signature
#: filtering. Only `triple_barrier` needs an override: `labels_triple_barrier`/`_sell` default
#: pt=0.03/sl=0.02 and only ever RAISE a barrier via max(pt, min_gain)/max(sl, min_loss) (see
#: `_TripleBarrierShim`'s docstring above), so min_gain/min_loss alone would silently floor the
#: up-barrier at 0.03 regardless of the requested threshold. Passing pt/sl directly sets the
#: actual barrier; every other labeler here takes its threshold via min_gain/min_loss (or an
#: unrelated method-specific default we deliberately leave alone, e.g. min_sharpe).
def _generic_extra_params(method: "LabelMethod", min_gain: float, min_loss: float) -> Dict[str, Any]:
    if method == LabelMethod.triple_barrier:
        return {"pt": min_gain, "sl": min_loss}
    return {}


class _GenericLabelShim:
    """Ternary-label shim for any PRICE-ONLY `LabelMethod`. See module docstring.

    Unlike `_TripleBarrierShim` (one method, hand-written param names), this shim is generic
    over methods with heterogeneous signatures. Rather than a per-method branch, it inspects
    each labeler's actual signature (`inspect.signature`) and passes only the argument names
    it accepts (`_filter_kwargs`). It binds the same production composition methods
    (`_SHARED_METHODS`) from `NNNC_MLX`'s MRO, and supplies its own
    `get_train_buy_signals`/`get_train_sell_signals` that call the module-level
    `TrainingSignals.get_train_buy_signals`/`get_train_sell_signals` dispatcher directly (the
    same landmine documented on `_TripleBarrierShim`: `BaseNNStrategy`'s own wrapper injects
    `rvol_threshold`, which most of these labelers don't accept).
    """

    dbg_verbose: bool = False

    def __init__(
        self,
        method: "LabelMethod",
        horizon: int,
        min_gain: float,
        min_loss: float,
        extra_params: Optional[Dict[str, Any]] = None,
        conflict_resolution: bool = True,
        peak_filter: bool = False,
    ) -> None:
        self.TRAINING_TYPE = method
        self.HORIZON = horizon
        self.MIN_BUY_GAIN_THRESHOLD = min_gain
        self.MIN_SELL_LOSS_THRESHOLD = min_loss
        self.LABEL_FILTER_WINDOW = horizon
        self.apply_label_conflict_resolution = conflict_resolution
        self.extra_params = dict(extra_params or {})
        self.dbg_curr_df = None
        for name in _SHARED_METHODS:
            setattr(self, name, getattr(NNNC_MLX, name).__get__(self))
        if not peak_filter:
            self.filter_peaks_by_future_performance = _generic_passthrough_filter

    def debug_print(self, msg: str) -> None:  # pragma: no cover - never called
        return None

    def _base_params(self) -> Dict[str, Any]:
        params = {
            "horizon": self.HORIZON,
            "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
            "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
        }
        params.update(self.extra_params)
        return params

    def _filtered_params(self) -> Dict[str, Any]:
        buy_fn = TrainingSignals.METHODS[self.TRAINING_TYPE]
        return _filter_kwargs(buy_fn, self._base_params())

    def get_train_buy_signals(self, future_df: pd.DataFrame) -> pd.Series:
        params = self._filtered_params()
        labels = TrainingSignals.get_train_buy_signals(
            future_df, method=self.TRAINING_TYPE, params=params,
        )
        peaks = np.where(labels.values > 0.5)[0]
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "buy", window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))
        signals[filtered_peaks] = 1
        return signals

    def get_train_sell_signals(self, future_df: pd.DataFrame) -> pd.Series:
        params = self._filtered_params()
        labels = TrainingSignals.get_train_sell_signals(
            future_df, method=self.TRAINING_TYPE, params=params,
        )
        peaks = np.where(labels.values > 0.5)[0]
        filtered_peaks = self.filter_peaks_by_future_performance(
            future_df, peaks, "sell", window=(self.LABEL_FILTER_WINDOW or self.HORIZON)
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float), dtype=float)
        if len(filtered_peaks) > 0:
            signals.iloc[filtered_peaks] = 1.0
        return signals


def ternary_labels_generic(
    df: pd.DataFrame,
    method: "LabelMethod | int",
    horizon: int,
    min_gain: float = _GENERIC_DEFAULT_MIN_GAIN,
    min_loss: float = _GENERIC_DEFAULT_MIN_LOSS,
    extra_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """SELL=0/HOLD=1/BUY=2 labels for `df` under `method`, conflict resolution ON, peak
    filter OFF (matching this study's triple-barrier setting). Raises whatever the
    underlying labeler/dispatcher raises (e.g. `TypeError` on an unsupported kwarg) --
    callers that want a per-method error row must catch it themselves."""
    method = LabelMethod(int(method))
    if extra_params is None:
        extra_params = _generic_extra_params(method, min_gain, min_loss)
    shim = _GenericLabelShim(
        method=method, horizon=horizon, min_gain=min_gain, min_loss=min_loss,
        extra_params=extra_params, conflict_resolution=True, peak_filter=False,
    )
    working = df.copy()
    trading_classes = shim.get_training_labels(working)
    return np.asarray(trading_classes, dtype=np.int8)


#: `label` dict default -- means "exactly the production gbb label, unchanged"; kept in
#: sync with `registry.DEFAULT_LABEL` but defined locally so this module has no
#: dependency on registry.py (data.py imports labels.py, not the other way round).
_DEFAULT_LABEL: Dict[str, Any] = {"name": "gbb"}


def label_params() -> dict:
    """The shim's attribute values, for the report (research R2 fidelity check)."""
    shim = _LabelShim()
    return {name: getattr(shim, name) for name in _SHIM_ATTRS}


#: short prefix per `GENERIC_LABEL_NAMES` entry, for `label_key`.
_GENERIC_METHOD_ABBR: Dict[str, str] = {
    "forward_mae": "fmae",
    "quantile_future": "qfut",
    "mfe_mae_ratio": "mmr",
    "local_min_followthrough": "lmft",
    "risk_adj_future": "raf",
    "multi_horizon_vote": "mhv",
    "optimal_signals": "opts",
    "future_sharpe": "fsharpe",
    "future_sortino": "fsortino",
    "future_expectancy": "fexp",
    "local_extrema": "lext",
    "geometry": "geom",
}

#: short abbreviation per extra-key name shared across the generic labelers (research: the
#: additional keys an experiment may supply, per the task list); an unrecognised key falls
#: back to its own name unabbreviated (`label_key` still stays a total function).
_GENERIC_KEY_ABBR: Dict[str, str] = {
    "horizon": "h",
    "min_gain": "g",
    "min_loss": "l",
    "max_drawdown": "dd",
    "atr_scale": "atr",
    "top_quantile": "q",
    "min_ratio": "r",
    "min_sharpe": "s",
}


def _format_label_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def label_key(label: Optional[Dict[str, Any]] = None) -> str:
    """Short, stable string for `label`, for cache/artifact naming.

    `None` or `{"name": "gbb"}` -> `"gbb"`. `{"name": "triple_barrier", "horizon": 48,
    "up": 0.02, "down": 0.02}` -> `"tb_h48_u0.02_d0.02_cr1_pf0"` (conflict_resolution and
    peak_filter default True/False per `registry._LABEL_TRIPLE_BARRIER_DEFAULTS`). When
    `atr_scale` is present, `"_atr{atr_scale:g}"` is appended, e.g.
    `"tb_h48_u0.005_d0.005_cr1_pf0_atr2"`.

    Any `GENERIC_LABEL_NAMES` entry: `<abbrev>` followed by `_<key_abbrev><value>` for every
    key other than `name`, in the order the keys appear in `label` (dict insertion order),
    e.g. `{"name": "forward_mae", "horizon": 12, "min_gain": 0.015, "max_drawdown": 0.05}`
    -> `"fmae_h12_g0.015_dd0.05"`, `{"name": "quantile_future", "horizon": 12,
    "top_quantile": 0.8}` -> `"qfut_h12_q0.8"`.
    """
    label = label or _DEFAULT_LABEL
    name = label.get("name", "gbb")
    if name == "gbb":
        return "gbb"
    if name in GENERIC_LABEL_NAMES:
        prefix = _GENERIC_METHOD_ABBR[name]
        parts = [prefix]
        for key, value in label.items():
            if key == "name":
                continue
            abbr = _GENERIC_KEY_ABBR.get(key, key)
            parts.append(f"{abbr}{_format_label_value(value)}")
        return "_".join(parts)
    horizon = label["horizon"]
    up = label["up"]
    down = label["down"]
    cr = int(bool(label.get("conflict_resolution", True)))
    pf = int(bool(label.get("peak_filter", False)))
    key = f"tb_h{horizon}_u{up:g}_d{down:g}_cr{cr}_pf{pf}"
    atr_scale = label.get("atr_scale")
    if atr_scale is not None:
        key += f"_atr{atr_scale:g}"
    return key


def ternary_labels(df: pd.DataFrame, label: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """SELL=0/HOLD=1/BUY=2 labels for `df`.

    `label=None` or `{"name": "gbb"}` (default): produced by `NNNC_MLX`'s own label
    methods, byte-identical to before this parameter existed.

    `{"name": "triple_barrier", "horizon": int, "up": float, "down": float,
    "conflict_resolution": bool (default True), "peak_filter": bool (default False)}`:
    produced by `TrainingSignals.LabelMethod.triple_barrier` via `_TripleBarrierShim`.

    Any `GENERIC_LABEL_NAMES` entry (any other PRICE-ONLY `LabelMethod`):
    `{"name": <method name>, "horizon": int, ...}` -- `min_gain`/`min_loss` default to 0.02
    (matching every `TrainingSignals.labels_*` buy function's own default) when omitted;
    every other key (e.g. `max_drawdown`, `top_quantile`, `min_ratio`, `min_sharpe`,
    `atr_scale`) is forwarded as `extra_params` to `ternary_labels_generic`, which
    signature-filters per labeler.

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
            atr_scale=label.get("atr_scale"),
        )
    elif name in GENERIC_LABEL_NAMES:
        extra_params = {
            k: v for k, v in label.items()
            if k not in ("name", "horizon", "min_gain", "min_loss")
        }
        return ternary_labels_generic(
            df,
            LabelMethod[name],
            label["horizon"],
            min_gain=label.get("min_gain", _GENERIC_DEFAULT_MIN_GAIN),
            min_loss=label.get("min_loss", _GENERIC_DEFAULT_MIN_LOSS),
            extra_params=extra_params,
        )
    else:
        raise ValueError(
            f"unknown label {name!r}; valid: gbb, triple_barrier, "
            + ", ".join(GENERIC_LABEL_NAMES)
        )

    working = df.copy()
    trading_classes = shim.get_training_labels(working)
    return np.asarray(trading_classes, dtype=np.int8)
