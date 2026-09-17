"""Trivial baselines for the NNC classifier study (T016, research R6).

Both return `(proba, note)` with `proba` shaped `(n_test, 3)` (columns SELL, HOLD, BUY) so they
route through `metrics.score` unchanged, and a human-readable `note` describing what the baseline
actually does (recorded on the run row so a reader is never misled about what "beating baseline"
means).

Self-contained: numpy only, no dependency on any other module in this package.
"""
from __future__ import annotations

import numpy as np

_N_CLASSES = 3


def majority(train_labels: np.ndarray, n_test: int) -> tuple:
    """Predict the training-span class frequencies for every test row.

    `argmax(proba)` is therefore the majority class in `train_labels`; the non-degenerate
    per-class probabilities keep log-loss/AUROC/ECE defined (research R6).
    """
    y = np.asarray(train_labels, dtype=int)
    counts = np.bincount(y, minlength=_N_CLASSES)[:_N_CLASSES].astype(float)
    total = counts.sum()
    freqs = counts / total if total > 0 else np.full(_N_CLASSES, 1.0 / _N_CLASSES)
    proba = np.tile(freqs, (n_test, 1))
    note = (
        "MAJORITY: every test row is predicted the TRAINING-SPAN class frequencies "
        "(argmax = the majority training class); a live-usable, non-oracle baseline."
    )
    return proba, note


def persistence(test_labels: np.ndarray, eps: float = 1e-3) -> tuple:
    """Predict the previous TEST ROW'S TRUE label, smoothed.

    Row i (i >= 1) puts (1 - eps) on `test_labels[i - 1]` and eps / 2 on each other class; row 0
    (no previous label) is uniform. Because this consumes the true label stream at test time
    rather than only information available live, it is an ORACLE-ISH baseline, not a real
    trading-usable predictor (research R6) -- it is labelled as such here and wherever it is
    reported.
    """
    y = np.asarray(test_labels, dtype=int)
    n = y.shape[0]
    proba = np.full((n, _N_CLASSES), eps / 2.0, dtype=float)
    if n > 1:
        proba[np.arange(1, n), y[:-1]] = 1.0 - eps
    if n > 0:
        proba[0, :] = 1.0 / _N_CLASSES
    note = (
        "PERSISTENCE (oracle-ish, research R6): row i predicts the PREVIOUS TEST ROW'S TRUE "
        f"label smoothed with eps={eps}; row 0 is uniform. This consumes the true test-label "
        "stream, so it is not a live-usable predictor -- reported for scale only."
    )
    return proba, note
