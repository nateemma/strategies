"""Filtered vs smoothed state inference -- the one line this whole feature turns on (T009, T010).

    V_t(k) = loss_t(k) + min_j [ V_{t-1}(j) + lam * 1[j != k] ]

FILTERED (online, honest):  s_t = argmin_k V_t(k). Forward pass only. V_t depends on x_1..x_t
                            and nothing later. This is the deployed path (FR-004).
SMOOTHED (Viterbi, the leak reference):  run forward to T, take argmin_k V_T(k), then BACK-TRACK.
                            The traceback is what makes s_t depend on bars after t (FR-007).

The difference is a single backward loop, and keeping it to exactly that is deliberate: the
measured gap between the two arms is only a leak size if nothing else differs between them.

`jumpmodels` implements the same recursion; its `predict()` defaults to `use_viterbi=True`, so
the package's obvious method is the SMOOTHED one. See the package audit in `crosscheck.py`.
"""
from __future__ import annotations

import numpy as np


def loss_matrix(x: np.ndarray, centroids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(n, k) weighted squared distance from each bar to each centroid."""
    diff = x[:, None, :] - centroids[None, :, :]          # (n, k, d)
    return np.einsum("nkd,d->nk", diff * diff, weights)


def _values(loss: np.ndarray, lam: float) -> np.ndarray:
    """Forward DP value matrix. Row t uses rows <= t and nothing else."""
    n, k = loss.shape
    pen = lam * (1.0 - np.eye(k))
    v = np.empty((n, k), dtype=float)
    v[0] = loss[0]
    for t in range(1, n):
        v[t] = loss[t] + (v[t - 1][:, None] + pen).min(axis=0)
    return v


def filtered_states(x: np.ndarray, fit) -> np.ndarray:
    """FR-004. Guarantee: filtered_states(x[:m]) == filtered_states(x)[:m] for every m.

    That equality holds because `_values` is a forward recursion and the argmin is taken
    per row -- nothing downstream of t can move row t.
    """
    v = _values(loss_matrix(x, fit.centroids, fit.weights), fit.lam)
    return fit.remap(v.argmin(axis=1))


def smoothed_states(x: np.ndarray, fit) -> np.ndarray:
    """FR-007, the deliberate leak. The prefix equality above MUST FAIL for this function.

    A smoothed sequence that satisfies it is not running the traceback, which would mean the
    leak reference is not a leak reference and the measured gap is not a leak size.
    """
    loss = loss_matrix(x, fit.centroids, fit.weights)
    v = _values(loss, fit.lam)
    n, k = loss.shape
    pen = fit.lam * (1.0 - np.eye(k))
    s = np.empty(n, dtype=int)
    s[-1] = int(v[-1].argmin())
    for t in range(n - 1, 0, -1):                        # <-- the backward pass, the leak
        s[t - 1] = int((v[t - 1] + pen[:, s[t]]).argmin())
    return fit.remap(s)
