# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Training-free triage of every PRICE-ONLY labeler in `Framework/TrainingSignals.py`
(feature 018 follow-up). Nothing here trains a model or reports P&L (spec FR-036,
Constitution III) -- every number is either a label-composition statistic (class
balance, signal rate) or a metric computed with NO fitted parameters at all
(persistence MCC, an AUROC of a fixed trailing feature against the label).

`_GenericLabelShim` generalises `research.clfstudy.labels._TripleBarrierShim` to any
`TrainingSignals.LabelMethod`: it binds the same production composition methods
(`get_training_labels`/`augment_training_signals`/`print_distribution_compact`, and
`filter_peaks_by_future_performance` when `peak_filter=True`) from `NNNC_MLX`'s MRO
(`_SHARED_METHODS`, imported from `labels.py` -- never redefined here), and supplies its
own `get_train_buy_signals`/`get_train_sell_signals` that call the module-level
`TrainingSignals.get_train_buy_signals`/`get_train_sell_signals` dispatcher directly (the
same landmine documented on `_TripleBarrierShim`: `BaseNNStrategy`'s own wrapper injects
`rvol_threshold`, which most of these labelers don't accept).

Unlike `_TripleBarrierShim` (one method, hand-written param names), this shim is generic
over methods with thirteen different signatures. Rather than a per-method branch, it
inspects each labeler's actual signature (`inspect.signature`) and passes only the
argument names it accepts (`_filter_kwargs`). This is safe for BOTH buy and sell calls:
`TrainingSignals.get_train_buy_signals` never filters its `params` dict itself (`fn(df,
**params)`, unfiltered -- the landmine), and `get_train_sell_signals`'s per-method
branches either (a) filter to their OWN allowed set internally (a superset intersection
is harmless) or (b) (methods with no explicit sell branch: mfe_mae_ratio,
risk_adj_future, multi_horizon_vote) fall back to calling the buy accessor with the same
params -- which is exactly what our buy-signature filtering already prepared. So filtering
once, against the BUY function's signature, is sufficient for every one of the 13 methods.
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from Framework import TrainingSignals
from NNNC.NNNC_MLX import NNNC_MLX
from research.clfstudy import REPORT_DIR, SPLITS_DIR
from research.clfstudy import data, splits
from research.clfstudy.labels import _SHARED_METHODS

LabelMethod = TrainingSignals.LabelMethod

#: (id, name) for every PRICE-ONLY labeler, in the order the task lists them. Indicator-based
#: methods (technical_indicators=7, trends=8, indicators2/3/4=15/16/19, gbb=17, bands=18,
#: breakout*=20-26, gbb_path=27) are intentionally excluded -- they read columns (guard_metric,
#: adx, rsi, ...) this triage does not evaluate.
PRICE_ONLY_METHOD_IDS: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14,
)

DEFAULT_PAIRS: Tuple[str, ...] = ("LINK/USDT", "SOL/USDT", "ZEC/USDT")
DEFAULT_HORIZONS: Tuple[int, ...] = (24, 48)
DEFAULT_MIN_GAIN = 0.02
DEFAULT_MIN_LOSS = 0.02

#: trailing feature windows (research choice, stated in the report): rolling std of log
#: close-to-close returns over this many bars for `vol_auroc`, log(close[t]/close[t-lag]) for
#: `dir_auroc`. Both are causal (use only bars up to and including t).
VOL_WINDOW = 48
DIR_LAG = 16

_SPLITS_PATH = SPLITS_DIR / "v1.json"


def _passthrough_filter(future_df: pd.DataFrame, peaks, signal_type: str = "buy", window: int = 64):
    """Peak filter bypass (peak_filter=False): peaks pass through unchanged, matching
    `_TripleBarrierShim._passthrough_filter` in labels.py (duplicated, not imported, since
    that one is a private staticmethod of an unrelated class not meant to be a shared util)."""
    return list(np.asarray(peaks, dtype=int))


def _filter_kwargs(fn, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keys of `params` that `fn`'s signature actually accepts (excluding `df`)."""
    accepted = set(inspect.signature(fn).parameters) - {"df"}
    return {k: v for k, v in params.items() if k in accepted}


#: per-method extra kwargs, applied on top of {horizon, min_gain, min_loss} before signature
#: filtering. Only `triple_barrier` needs an override: `labels_triple_barrier`/`_sell` default
#: pt=0.03/sl=0.02 and only ever RAISE a barrier via max(pt, min_gain)/max(sl, min_loss) (see
#: `_TripleBarrierShim`'s docstring in labels.py), so min_gain/min_loss alone would silently
#: floor the up-barrier at 0.03 regardless of the requested threshold. Passing pt/sl directly
#: sets the actual barrier; every other labeler here takes its threshold via min_gain/min_loss
#: (or an unrelated method-specific default we deliberately leave alone, e.g. min_sharpe).
def _extra_params(method: "LabelMethod", min_gain: float, min_loss: float) -> Dict[str, Any]:
    if method == LabelMethod.triple_barrier:
        return {"pt": min_gain, "sl": min_loss}
    return {}


class _GenericLabelShim:
    """Ternary-label shim for any PRICE-ONLY `LabelMethod`. See module docstring."""

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
            self.filter_peaks_by_future_performance = _passthrough_filter

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
    min_gain: float = DEFAULT_MIN_GAIN,
    min_loss: float = DEFAULT_MIN_LOSS,
    extra_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """SELL=0/HOLD=1/BUY=2 labels for `df` under `method`, conflict resolution ON, peak
    filter OFF (matching this study's triple-barrier setting). Raises whatever the
    underlying labeler/dispatcher raises (e.g. `TypeError` on an unsupported kwarg) --
    callers that want a per-method error row must catch it themselves (see `triage()`)."""
    method = LabelMethod(int(method))
    if extra_params is None:
        extra_params = _extra_params(method, min_gain, min_loss)
    shim = _GenericLabelShim(
        method=method, horizon=horizon, min_gain=min_gain, min_loss=min_loss,
        extra_params=extra_params, conflict_resolution=True, peak_filter=False,
    )
    working = df.copy()
    trading_classes = shim.get_training_labels(working)
    return np.asarray(trading_classes, dtype=np.int8)


def _trailing_features(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Causal (t) trailing vol/momentum features on a pair's CONTINUOUS frame, computed
    BEFORE any role/window selection (same convention as `windows.py`): a test row's
    lookback legitimately reaches back over real preceding bars.

    vol: rolling std of log close-to-close returns over `VOL_WINDOW` bars ending at t
    (`min_periods=VOL_WINDOW`, so the first `VOL_WINDOW` rows are NaN -- not shifted
    further, since a trailing-window std ending at t only ever uses bars <= t).
    dir: log(close[t] / close[t - DIR_LAG]) (NaN for the first `DIR_LAG` rows).
    """
    close = frame["close"].to_numpy(dtype=float)
    log_close = np.log(close)
    logret = np.empty_like(log_close)
    logret[0] = np.nan
    logret[1:] = np.diff(log_close)
    vol = pd.Series(logret).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy()

    mom = np.full(len(close), np.nan)
    if len(close) > DIR_LAG:
        mom[DIR_LAG:] = log_close[DIR_LAG:] - log_close[:-DIR_LAG]
    return vol, mom


def _safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(matthews_corrcoef(y_true, y_pred))


def _safe_auroc(y_bin: np.ndarray, score: np.ndarray) -> float:
    if len(y_bin) == 0 or np.unique(y_bin).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_bin, score))


def _one_row(
    method: "LabelMethod",
    horizon: int,
    frames: Dict[str, pd.DataFrame],
    vol_by_pair: Dict[str, np.ndarray],
    mom_by_pair: Dict[str, np.ndarray],
    split_set: "splits.SplitSet",
    min_gain: float,
    min_loss: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"method_id": int(method), "method": method.name, "horizon": horizon}

    labels_by_pair: Dict[str, np.ndarray] = {}
    try:
        for pair, frame in frames.items():
            labels_by_pair[pair] = ternary_labels_generic(
                frame, method, horizon, min_gain=min_gain, min_loss=min_loss,
            )
    except Exception as exc:  # noqa: BLE001 - triage must survive any labeler's own bugs
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["error"] = None

    per_window_balance: Dict[str, List[float]] = {}
    persistence_by_pair: Dict[str, Dict[str, List[np.ndarray]]] = {}
    pooled_vol_feat: List[np.ndarray] = []
    pooled_vol_lbl: List[np.ndarray] = []
    pooled_mom_feat: List[np.ndarray] = []
    pooled_mom_lbl: List[np.ndarray] = []
    pooled_signal: List[np.ndarray] = []

    for w in split_set.windows:
        window_labels: List[np.ndarray] = []
        for pair, frame in frames.items():
            roles = splits.assign_roles(frame["date"], w)
            test_mask = roles == "test"
            if not test_mask.any():
                continue
            y = labels_by_pair[pair][test_mask]
            window_labels.append(y)
            pooled_signal.append(y)

            # Persistence: lag-1 comparisons computed WITHIN this (pair, window) segment
            # only, so a segment boundary (window gap, or a different pair) never forms a
            # (y[t], y[t+1]) pair across two unrelated bars.
            if len(y) >= 2:
                d = persistence_by_pair.setdefault(pair, {"y0": [], "y1": []})
                d["y0"].append(y[:-1])
                d["y1"].append(y[1:])

            vol_feat = vol_by_pair[pair][test_mask]
            pooled_vol_feat.append(vol_feat)
            pooled_vol_lbl.append((y != 1).astype(int))

            mom_feat = mom_by_pair[pair][test_mask]
            nonhold = y != 1
            pooled_mom_feat.append(mom_feat[nonhold])
            pooled_mom_lbl.append((y[nonhold] == 2).astype(int))

        if window_labels:
            all_y = np.concatenate(window_labels)
            counts = np.bincount(all_y, minlength=3).astype(float)
            shares = (counts / counts.sum()).tolist()
        else:
            shares = [float("nan")] * 3
        per_window_balance[w.window_id] = shares

    balance_arr = np.array(list(per_window_balance.values()), dtype=float)
    balance_mean = (
        np.nanmean(balance_arr, axis=0).tolist() if balance_arr.size else [float("nan")] * 3
    )

    per_pair_mcc: Dict[str, float] = {}
    all_y0, all_y1 = [], []
    for pair, d in persistence_by_pair.items():
        y0 = np.concatenate(d["y0"])
        y1 = np.concatenate(d["y1"])
        per_pair_mcc[pair] = _safe_mcc(y0, y1)
        all_y0.append(y0)
        all_y1.append(y1)
    pooled_persistence_mcc = (
        _safe_mcc(np.concatenate(all_y0), np.concatenate(all_y1)) if all_y0 else float("nan")
    )

    vol_feat_all = np.concatenate(pooled_vol_feat) if pooled_vol_feat else np.array([])
    vol_lbl_all = np.concatenate(pooled_vol_lbl) if pooled_vol_lbl else np.array([])
    vol_valid = ~np.isnan(vol_feat_all)
    vol_auroc = _safe_auroc(vol_lbl_all[vol_valid], vol_feat_all[vol_valid])

    mom_feat_all = np.concatenate(pooled_mom_feat) if pooled_mom_feat else np.array([])
    mom_lbl_all = np.concatenate(pooled_mom_lbl) if pooled_mom_lbl else np.array([])
    mom_valid = ~np.isnan(mom_feat_all)
    dir_auroc = _safe_auroc(mom_lbl_all[mom_valid], mom_feat_all[mom_valid])

    pooled_signal_all = np.concatenate(pooled_signal) if pooled_signal else np.array([])
    signal_rate = (
        float(np.mean(pooled_signal_all != 1)) if len(pooled_signal_all) else float("nan")
    )

    row.update(
        {
            "balance_per_window": per_window_balance,
            "balance_mean": balance_mean,
            "signal_rate": signal_rate,
            "persistence_mcc_per_pair": per_pair_mcc,
            "persistence_mcc_pooled": pooled_persistence_mcc,
            "vol_auroc": vol_auroc,
            "dir_auroc": dir_auroc,
        }
    )
    return row


def triage(
    pairs: Sequence[str] = DEFAULT_PAIRS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_gain: float = DEFAULT_MIN_GAIN,
    min_loss: float = DEFAULT_MIN_LOSS,
    method_ids: Sequence[int] = PRICE_ONLY_METHOD_IDS,
) -> List[Dict[str, Any]]:
    """Training-free triage of `method_ids` x `horizons` on `pairs`' pooled TEST rows.

    Loads `pairs` ONCE via `data.load_all` (the default gbb-labeled cache; only OHLCV +
    date are used here, the cached `label` column is ignored) and reuses the resulting
    frames/trailing-features across every (method, horizon) combination.
    """
    frames = data.load_all(tuple(pairs))
    split_set = splits.load(str(_SPLITS_PATH))

    vol_by_pair: Dict[str, np.ndarray] = {}
    mom_by_pair: Dict[str, np.ndarray] = {}
    for pair, frame in frames.items():
        vol_by_pair[pair], mom_by_pair[pair] = _trailing_features(frame)

    rows: List[Dict[str, Any]] = []
    for method_id in method_ids:
        method = LabelMethod(int(method_id))
        for horizon in horizons:
            rows.append(
                _one_row(
                    method, horizon, frames, vol_by_pair, mom_by_pair, split_set,
                    min_gain=min_gain, min_loss=min_loss,
                )
            )
    return rows


_LEGEND = """
Legend:
- persistence MCC: matthews_corrcoef(y[1:], y[:-1]) on pooled test rows (per pair, boundaries
  never crossed between pairs or windows) -- what a PERSISTENCE oracle (predict the previous
  bar's label) would score, computed with NO training. High values (>= 0.7) mean the label has
  strong overlapping-horizon autocorrelation, which inflates any model's apparent MCC on the
  same label (adjacent rows share most of their label-horizon window).
- vol AUROC: AUROC of a trailing volatility feature (rolling std of log close-to-close returns,
  48 bars) against `label != HOLD`. Near 0.5 is good (the label is not just a volatility
  detector); >= 0.60 flags a labeler whose signal rate is substantially explained by recent
  realised volatility alone.
- dir AUROC: AUROC of a trailing momentum feature (log(close[t]/close[t-16])) against
  `label == BUY`, restricted to non-HOLD rows. Far from 0.5 (either direction) means a trivial
  linear momentum feature already separates BUY from SELL -- the labeler may not need a model.
"""


def _fmt_pct(x: float) -> str:
    return "nan" if x != x else f"{100 * x:.1f}%"


def _fmt(x: float, nd: int = 3) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    rows_sorted = sorted(rows, key=lambda r: (r["method_id"], r["horizon"]))
    header = (
        "| method | horizon | SELL | HOLD | BUY | signal rate | persistence MCC | "
        "vol AUROC | dir AUROC | error |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows_sorted:
        if r.get("error"):
            lines.append(
                f"| {r['method']} ({r['method_id']}) | {r['horizon']} | - | - | - | - | - | - "
                f"| - | {r['error']} |"
            )
            continue
        sell, hold, buy = r["balance_mean"]
        lines.append(
            f"| {r['method']} ({r['method_id']}) | {r['horizon']} | {_fmt_pct(sell)} | "
            f"{_fmt_pct(hold)} | {_fmt_pct(buy)} | {_fmt_pct(r['signal_rate'])} | "
            f"{_fmt(r['persistence_mcc_pooled'])} | {_fmt(r['vol_auroc'])} | "
            f"{_fmt(r['dir_auroc'])} | |"
        )
    return "\n".join(lines) + "\n" + _LEGEND


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.labeltriage")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--out", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    pairs = tuple(p.strip() for p in args.pairs.split(",") if p.strip())
    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())

    tracemalloc.start()
    t0 = time.perf_counter()
    rows = triage(pairs=pairs, horizons=horizons)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    table = _to_markdown(rows)
    print(table)
    print(f"\nwall time: {elapsed:.1f}s, peak traced memory: {peak / 1e6:.1f} MB")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_triage.md").write_text(table + "\n")
    (out_dir / "label_triage.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
