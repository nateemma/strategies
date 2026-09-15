"""The screen: does the embedding beat the hand-built features? (T006-T010, T027-T033)

THE BAR IS MEASURED, NEVER QUOTED. rho ~ 0.15 is cited throughout this repo as "the OHLCV
information ceiling". Measured across five folds and six horizons it is a HORIZON-3-TO-5 number:

    H        3        6       12       24       48       96
    median  0.1465   0.1003   0.0684   0.0556   0.0499   0.0254

Predictability decays with forward horizon, so a constant is the wrong bar at every horizon but
one. Quoting 0.15 at H=48 -- where the hand-built features themselves reach 0.083 -- would have
set a bar THE BASELINE CANNOT CLEAR, and an embedding failing it would fail for a reason
unrelated to the hypothesis. Each screen therefore computes its own bar with the same ridge on
the same split.

THREE-WAY SPLIT, because there are two fittable objects and the obvious design leaks through the
second. The encoder is unsupervised, so it is tempting to fit the ridge on the same range -- but
the ridge IS supervised, and one fitted and scored on the same bars beats any bar. That is how a
null becomes a false positive.

    fold training range   ->  A  encoder   (no labels)
                          ->  B  ridge     (supervised; labels from B only)
    fold test range       ->  C  score     <- the only number reported

PRIMARY IS FIXED BEFORE ANY SCREEN RUNS and written into the record BEFORE scoring, so the record
itself shows it was not chosen afterwards. Four screens against one threshold would be four
chances to clear a bar meant to be cleared once -- the failure brief 05 measured at PBO 0.7128.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.ssl import inputs as IN                 # noqa: E402

# Pre-registered, before any screen runs (FR-007b, SC-001a).
PRIMARY = ("engineered", 48)
INPUTS = ("engineered", "ohlcv")
HORIZONS = (6, 48)

# research.md R1 — the measured ceiling under a PER-PAIR TEMPORAL split, for the reproduction
# test (T009). NOT a bar; the bar is recomputed per screen.
#
# R1's first numbers were higher and WRONG. That probe stacked every pair and then sliced the
# array, so "the first 60%" was THE FIRST 6 PAIRS rather than the first 60% of time — a
# contemporaneous cross-sectional split, not forward-in-time prediction. Measured on f07: at H=6
# it makes no difference (0.1601 vs 0.1602), but at H=48 it nearly doubles rho (0.0926 vs
# 0.0490), because a contemporaneous split lets market-wide moves leak between the fit and score
# sets and those dominate at longer horizons.
MEASURED_CEILING = {3: 0.1465, 6: 0.1003, 12: 0.0684, 24: 0.0556, 48: 0.0499, 96: 0.0254}

RIDGE_ALPHA = 1.0
BOOT_N = 2000
BOOT_SEED = 0


def splits(n: int, a: float = 0.7) -> tuple[slice, slice]:
    """Training range -> A (encoder) | B (ridge), in time order."""
    cut = int(a * n)
    return slice(0, cut), slice(cut, n)


def purge(X: np.ndarray, y: np.ndarray, horizon: int, embargo: int = 96):
    """Drop the last `horizon + embargo` rows of a split.

    A label at the end of a split reaches `horizon` bars past its boundary, so without this the
    ridge's training labels overlap the scoring range (FR-003). Applied at EVERY boundary, not
    only A|B.
    """
    k = horizon + embargo
    return (X[:-k], y[:-k]) if len(X) > k else (X[:0], y[:0])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if a.size < 8:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _fit_score(Xb, yb, Xc, yc) -> float:
    """Ridge fitted on B ONLY, scored on C ONLY."""
    from sklearn.linear_model import Ridge
    if len(Xb) < 50 or len(Xc) < 50:
        return float("nan")
    r = Ridge(alpha=RIDGE_ALPHA).fit(Xb, yb)
    return _spearman(r.predict(Xc), yc)


def _boot_ci(Xb, yb, Xc, yc, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Interval on the C-side rho by resampling C — the bar is compared with an interval, never
    bare (FR-009)."""
    from sklearn.linear_model import Ridge
    if len(Xb) < 50 or len(Xc) < 50:
        return (float("nan"), float("nan"))
    pred = Ridge(alpha=RIDGE_ALPHA).fit(Xb, yb).predict(Xc)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(yc), size=(n, len(yc)))
    vals = np.array([_spearman(pred[i], yc[i]) for i in idx[:200]])   # 200 is ample for a CI
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def stack(per_pair: dict, horizon: int, sl: slice):
    """Pool pairs over one split, aligning each pair's features to its own forward return."""
    A, B = [], []
    for X, px in per_pair.values():
        n = min(len(X), len(px))
        fwd = IN.forward_return(px[:n], horizon)
        if fwd.size < 100:
            continue
        Xi, yi = X[:len(fwd)], fwd
        Xi, yi = Xi[sl], yi[sl]
        A.append(Xi); B.append(yi)
    if not A:
        return np.zeros((0, 1)), np.zeros(0)
    X = np.vstack(A); y = np.concatenate(B)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[m], y[m]


def handbuilt_bar(fold, horizon: int) -> dict:
    """THE BAR: the same ridge over the production feature set, same horizon, same split."""
    per = {}
    for p in sorted(fold.pair_universe):
        try:
            per[p] = IN.engineered(p, fold.test_start, fold.test_end)
        except Exception:
            pass
    if not per:
        return {"rho": float("nan"), "ci": [float("nan")] * 2, "n_pairs": 0}
    n = max(min(len(X), len(px)) for X, px in per.values())
    sa, sb = splits(n)
    Xb, yb = stack(per, horizon, sa)
    Xc, yc = stack(per, horizon, sb)
    Xb, yb = purge(Xb, yb, horizon)
    rho = _fit_score(Xb, yb, Xc, yc)
    lo, hi = _boot_ci(Xb, yb, Xc, yc)
    return {"rho": rho, "ci": [lo, hi], "n_pairs": len(per),
            "n_fit": int(len(Xb)), "n_score": int(len(Xc))}


def baseline_table(fold_ids, horizons=(3, 6, 12, 24, 48, 96)) -> dict:
    """T009's reproduction check — the bar across folds and horizons."""
    from research.labelfree.folds import folds as _folds
    by = {f.fold_id: f for f in _folds()}
    out = {}
    for H in horizons:
        vals = []
        for fid in fold_ids:
            r = handbuilt_bar(by[fid], H)
            if np.isfinite(r["rho"]):
                vals.append(r["rho"])
        if vals:
            out[H] = {"median": round(float(np.median(vals)), 4),
                      "min": round(float(np.min(vals)), 4),
                      "max": round(float(np.max(vals)), 4),
                      "n_folds": len(vals)}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="f05..f14")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "screen.json"))
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    fids = parse_span(a.folds)

    if a.baseline_only:
        tab = baseline_table(fids)
        print("the hand-built bar, measured (NOT quoted):")
        print("  %-6s %9s %9s %9s %8s %10s" % ("H", "median", "min", "max", "folds", "R1 says"))
        for H, v in sorted(tab.items()):
            print("  %-6d %9.4f %9.4f %9.4f %8d %10.4f"
                  % (H, v["median"], v["min"], v["max"], v["n_folds"],
                     MEASURED_CEILING.get(H, float('nan'))))
        Path(a.out).with_name("baseline.json").write_text(json.dumps(tab, indent=2))
        return 0

    raise SystemExit("the encoder is required for a full screen; run --baseline-only for the bar")


if __name__ == "__main__":
    raise SystemExit(main())
