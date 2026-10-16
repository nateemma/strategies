"""037 scores (research R6, R7): rank AUC, binned normalised MI, signal lift, the stratified
(residual) AUC beyond bb_width + guard_metric, and window aggregation. Pure numpy; no I/O.

Every function drops rows where the column or the target is undefined; a score that cannot be
formed returns NaN (never 0 -- a broken probe must not look like a null).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _clean(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok] > 0.5


def auc(x, y) -> float:
    """Mann-Whitney AUC of `x` ranking y=1 above y=0, ties averaged."""
    x, y = _clean(x, y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_multi(x, ys) -> list[float]:
    """`auc(x, y)` for several targets with ONE sort of x (exact Mann-Whitney, ties averaged).
    Rows where x is not finite are dropped; per target, rows where y is not finite are ignored."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    xs_idx = np.flatnonzero(fin)
    order = xs_idx[np.argsort(x[fin], kind="mergesort")]
    xs = x[order]
    gid = np.cumsum(np.r_[True, xs[1:] != xs[:-1]]) - 1 if len(xs) else np.array([], int)
    out = []
    for y in ys:
        yo = np.asarray(y, dtype=float)[order]
        pos = (yo > 0.5) & np.isfinite(yo)
        neg = (yo <= 0.5) & np.isfinite(yo)
        n1, n0 = int(pos.sum()), int(neg.sum())
        if n1 == 0 or n0 == 0:
            out.append(float("nan"))
            continue
        pg = np.bincount(gid, weights=pos)
        ng = np.bincount(gid, weights=neg)
        below = np.cumsum(ng) - ng
        out.append(float((pg * (below + 0.5 * ng)).sum() / (n1 * n0)))
    return out


def direction(a: float) -> int:
    return 0 if not np.isfinite(a) or a == 0.5 else (1 if a > 0.5 else -1)


def strength(a: float) -> float:
    return abs(a - 0.5) if np.isfinite(a) else float("nan")


def quantile_edges(x, bins: int = 20) -> np.ndarray:
    """Interior quantile edges (unique) from `x` -- the TRAINING rows (Principle XI)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([])
    return np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))


def _entropy(p) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_norm(x, y, edges) -> float:
    """Mutual information (bits) between binned x and binary y, divided by H(y)."""
    x, y = _clean(x, y)
    if len(y) == 0 or y.all() or not y.any():
        return float("nan")
    b = np.searchsorted(np.asarray(edges, dtype=float), x, side="right")
    k = int(b.max()) + 1
    joint = np.zeros((k, 2))
    np.add.at(joint, (b, y.astype(int)), 1)
    joint /= joint.sum()
    hy = _entropy(joint.sum(0))
    mi = _entropy(joint.sum(1)) + hy - _entropy(joint.ravel())
    return float(max(mi, 0.0) / hy)


def lift(signal, y, min_fires: int, max_fire_rate: float) -> tuple[float, float, int]:
    """(lift, fire_rate, fires). lift = P(y | fires) / P(y); NaN when the signal fires fewer than
    `min_fires` times or on more than `max_fire_rate` of the rows, or the base rate is 0."""
    s = np.asarray(signal, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(s) & np.isfinite(yy)
    s, yy = s[ok] > 0.5, yy[ok] > 0.5
    n = len(s)
    if n == 0:
        return float("nan"), float("nan"), 0
    fires = int(s.sum())
    rate = fires / n
    base = yy.mean()
    if fires < min_fires or rate > max_fire_rate or base == 0:
        return float("nan"), rate, fires
    return float(yy[s].mean() / base), rate, fires


def stratified_auc(x, y, a, b, edges_a, edges_b) -> float:
    """AUC of x within cells of the a x b grid (edges from training rows), cells weighted by
    n_pos * n_neg (van Elteren style). Cells with only one class contribute nothing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    x, y, a, b = x[ok], y[ok] > 0.5, a[ok], b[ok]
    ca = np.searchsorted(np.asarray(edges_a, dtype=float), a, side="right")
    cb = np.searchsorted(np.asarray(edges_b, dtype=float), b, side="right")
    cell = ca * (len(edges_b) + 1) + cb
    num = den = 0.0
    for c in np.unique(cell):
        m = cell == c
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0
        num += w * auc(x[m], y[m])
        den += w
    return float(num / den) if den else float("nan")


def aggregate(window_scores: list[dict], min_rows: int, key: str = "auc") -> dict:
    """Median and same-direction share over computable windows (n_rows >= min_rows and a finite
    score), overall and per half. AUC-like keys use direction about 0.5; lift about 1.0."""
    centre = 1.0 if key == "lift" else 0.5
    use = [w for w in window_scores
           if w.get("n_rows", 0) >= min_rows and np.isfinite(w.get(key, float("nan")))]
    out = {"n_windows": len(use)}
    halves = {}
    for h in ("e0", "early", "late"):
        v = [w[key] for w in use if w["half"] == h]
        halves[h] = {"median": float(np.median(v)) if v else None, "n_windows": len(v)}
    out["halves"] = halves
    tests = [w[key] for w in use if w["half"] != "e0"]    # the 15 test windows; E0 reported apart
    if not tests:
        out.update(median=None, median_direction=0, same_direction_share=None)
        return out
    med = float(np.median(tests))
    d = 0 if med == centre else (1 if med > centre else -1)
    same = float(np.mean([(1 if v > centre else -1 if v < centre else 0) == d for v in tests]))
    out.update(median=med, median_direction=d, same_direction_share=same)
    return out
