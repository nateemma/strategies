"""Classification metrics for the NNC classifier study (T012, spec FR-016..FR-025).

`score()` turns one test-set `(y_true, proba)` pair into every field of data-model.md
§RunResult's headline/per-class/probabilistic/matrix/secondary groups. `timing()` measures
inference latency of a fitted classifier's `predict`. `summarise_seeds()` aggregates a list of
`score()`-shaped dicts over seeds (mean/std, used later by the leaderboard).

Self-contained: numpy + sklearn only, no dependency on any other module in this package (so it
is safe to run/test while the rest of the study harness is still being written).

Labels are fixed SELL=0, HOLD=1, BUY=2 throughout (matches `research.clfstudy.CLASS_NAMES`,
duplicated here as local constants to avoid importing anything else from the package).

Per-class NaN convention (research R8): a class's precision/recall is NaN when its support in
that role (predicted count for precision, true count for recall) is zero, via
`precision_recall_fscore_support(..., zero_division=np.nan)` (sklearn 1.9). A class's one-vs-rest
AUROC is NaN when the class is absent or universal in `y_true` (only one unique value in the
binary indicator) -- computed manually rather than relying on sklearn's own NaN-with-warning
behaviour, so no UndefinedMetricWarning escapes `score()`. `auroc_macro` is the nanmean of the
three (NaN only if all three are NaN). Reliability bins with zero rows report `conf`/`acc` as NaN
and contribute 0 to the ECE sum.

Balanced accuracy uses `sklearn.metrics.balanced_accuracy_score`, which is the unweighted mean of
per-class recall over the classes PRESENT in `y_true` (absent classes are silently excluded, not
scored as 0) -- this is FR-016's "per-class recall" mean restricted to classes that can have a
recall at all.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

_N_CLASSES = 3
_CLASS_LABELS = (0, 1, 2)  # SELL, HOLD, BUY
_CLASS_KEYS = ("sell", "hold", "buy")


def _auroc_one_vs_rest(y_true: np.ndarray, class_score: np.ndarray, class_idx: int) -> float:
    """One-vs-rest AUROC for `class_idx`; NaN if that class is absent or universal in y_true."""
    y_bin = (y_true == class_idx).astype(int)
    if np.unique(y_bin).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_bin, class_score))


def score(
    y_true: np.ndarray,
    proba: np.ndarray,
    buy_threshold: float = 0.8,
    n_bins: int = 15,
    collapse_frac: float = 0.995,
) -> dict:
    """Score one test-set prediction against the ternary labels (spec FR-016..FR-025).

    Args:
        y_true: (N,) int array of true labels in {0, 1, 2}.
        proba: (N, 3) array of predicted class probabilities (rows need not be checked to sum
            to 1; callers are responsible for passing a valid softmax output).
        buy_threshold: decision threshold on `proba[:, 2]` for the secondary BUY view (FR-025).
        n_bins: number of equal-width confidence bins for ECE (FR-022).
        collapse_frac: fraction of predicted rows in a single class above which a run is
            flagged `collapsed`.

    Returns:
        dict with keys: bal_acc, macro_f1, mcc, prec_sell, prec_hold, prec_buy, rec_sell,
        rec_hold, rec_buy, logloss, auroc_sell, auroc_hold, auroc_buy, auroc_macro, ece,
        reliability_bins, confusion, buy_prec_at_08, buy_cov_at_08, pred_counts, class_counts,
        collapsed, class_absent.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = y_true.shape[0]
    if proba.shape != (n, _N_CLASSES):
        raise ValueError(f"proba must have shape ({n}, {_N_CLASSES}), got {proba.shape}")

    y_pred = np.argmax(proba, axis=1)

    with warnings.catch_warnings():
        # balanced_accuracy_score warns "y_pred contains classes not in y_true" on collapsed /
        # class_absent rows; that condition is exactly what `collapsed`/`class_absent` flag below.
        warnings.simplefilter("ignore")
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(
            f1_score(y_true, y_pred, labels=_CLASS_LABELS, average="macro", zero_division=0)
        )
        mcc = float(matthews_corrcoef(y_true, y_pred))
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=_CLASS_LABELS, zero_division=np.nan
        )

    out: dict = {"bal_acc": bal_acc, "macro_f1": macro_f1, "mcc": mcc}
    for key, p, r in zip(_CLASS_KEYS, precision, recall):
        out[f"prec_{key}"] = float(p)
        out[f"rec_{key}"] = float(r)

    # log-loss: mean negative log prob of the true class, clipped to [1e-15, 1-1e-15] (FR-020).
    proba_clipped = np.clip(proba, 1e-15, 1 - 1e-15)
    true_class_prob = proba_clipped[np.arange(n), y_true]
    out["logloss"] = float(-np.mean(np.log(true_class_prob)))

    aurocs = []
    for idx, key in enumerate(_CLASS_KEYS):
        a = _auroc_one_vs_rest(y_true, proba[:, idx], idx)
        out[f"auroc_{key}"] = a
        aurocs.append(a)
    out["auroc_macro"] = (
        float(np.nanmean(aurocs)) if not np.all(np.isnan(aurocs)) else float("nan")
    )

    # ECE: 15 equal-width bins on max(prob) (FR-022).
    conf = proba.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability_bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n_b = int(mask.sum())
        if n_b > 0:
            conf_b = float(conf[mask].mean())
            acc_b = float(correct[mask].mean())
            ece += (n_b / n) * abs(acc_b - conf_b)
        else:
            conf_b = float("nan")
            acc_b = float("nan")
        reliability_bins.append(
            {"lo": float(lo), "hi": float(hi), "conf": conf_b, "acc": acc_b, "n": n_b}
        )
    out["ece"] = float(ece)
    out["reliability_bins"] = reliability_bins

    out["confusion"] = confusion_matrix(y_true, y_pred, labels=_CLASS_LABELS).tolist()

    buy_mask = proba[:, 2] >= buy_threshold
    out["buy_prec_at_08"] = (
        float((y_true[buy_mask] == 2).mean()) if buy_mask.any() else float("nan")
    )
    out["buy_cov_at_08"] = float(buy_mask.mean()) if n > 0 else float("nan")

    pred_counts = np.bincount(y_pred, minlength=_N_CLASSES)[:_N_CLASSES]
    class_counts = np.bincount(y_true, minlength=_N_CLASSES)[:_N_CLASSES]
    out["pred_counts"] = pred_counts.tolist()
    out["class_counts"] = class_counts.tolist()
    out["collapsed"] = bool(n > 0 and pred_counts.max() / n >= collapse_frac)
    out["class_absent"] = bool((class_counts == 0).any())

    return out


def timing(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    batch_size: int = 256,
    n_b1: int = 200,
) -> dict:
    """Inference latency of `predict_fn` over `X_test` (FR-024).

    `predict_fn` takes a numpy array (a batch, or a single row) and returns predictions; only
    wall-clock time is measured, its return value is discarded.

    Returns dict with `infer_ms_per_window_b256` (full pass over X_test in batches of
    `batch_size`, total time / N rows * 1000) and `infer_ms_per_window_b1` (mean of `n_b1`
    single-row calls, after a 10-call warm-up).
    """
    n = X_test.shape[0]

    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        predict_fn(X_test[start : start + batch_size])
    t1 = time.perf_counter()
    infer_ms_per_window_b256 = (t1 - t0) / n * 1000.0

    for i in range(10):
        predict_fn(X_test[i % n : i % n + 1])

    b1_times = []
    for i in range(n_b1):
        idx = i % n
        t0 = time.perf_counter()
        predict_fn(X_test[idx : idx + 1])
        t1 = time.perf_counter()
        b1_times.append(t1 - t0)
    infer_ms_per_window_b1 = float(np.mean(b1_times) * 1000.0)

    return {
        "infer_ms_per_window_b256": float(infer_ms_per_window_b256),
        "infer_ms_per_window_b1": infer_ms_per_window_b1,
    }


def summarise_seeds(rows: list) -> dict:
    """Aggregate a list of `score()`-shaped dicts (one per seed) into mean/std per metric.

    Every key whose value is a scalar number (int/float, not bool) in `rows[0]` is aggregated:
    `{key}_mean` = nanmean over seeds, `{key}_std` = nanstd(ddof=1) over seeds (NaN when fewer
    than 2 seeds). Non-scalar keys (confusion, reliability_bins, pred_counts, ...) are ignored
    here; the leaderboard (T019) handles those separately. Also returns `n_seeds`.
    """
    if not rows:
        raise ValueError("rows must contain at least one row")

    numeric_keys = [
        k
        for k, v in rows[0].items()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    ]

    out: dict = {"n_seeds": len(rows)}
    for k in numeric_keys:
        vals = np.array([float(r[k]) for r in rows], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
        out[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) >= 2 else float("nan")
    return out
