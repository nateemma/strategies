"""038 tier 1: the corrected statistical forecasters of utils/Forecasters.py (Story 3; research R5, R6).

At every 16th candle of the test windows: the trailing 64 log closes (AR: the trailing 64 one-candle
log returns, its path summed) -> forecast path -> f[n] = forecast log close n ahead - log close now.
Detrending and smoothing off. A window whose last candle is gap-filled, or with more than half its
candles gap-filled, is skipped (skipped_gap); a fit that raises is counted (failed), never zero.
The library returns the last len(window) values, so each window is forecast twice: 64 steps
(horizons 1-64) and 96 steps (horizon 96).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.stat_tier [--model holt] [--workers 4]
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research.forecast import common as C
from research.forecast import frames as F

import Forecasters as FC  # noqa: E402  (utils/ is on sys.path via common); imported here so a
#                            missing module fails loudly instead of being counted as forecast failures

WINDOW = 64
STRIDE = 16
MODELS = {"ar": "AR", "holt": "HOLT", "ses": "SIMPLE_EXPONENTAL", "expsm": "EXPONENTAL", "ets": "ETS",
          "ss_expsm": "SS_EXPONENTAL", "arima": "ARIMA", "theta": "THETA",
          # post-hoc (preregistration_harmonic.json): harmonic regression, smoothing / detrending variants
          "harm": "HARMONIC", "harm_s": "HARMONIC", "harm_d": "HARMONIC", "harm_sd": "HARMONIC"}
# post-hoc (preregistration_vote.json): members of the N-of-M votes
MODELS |= {f"harm_k{k}": "HARMONIC" for k in (1, 2, 4, 5)}
MODELS |= {f"holt_w{w}": "HOLT" for w in (32, 128, 256)}
WINDOWS = {f"holt_w{w}": w for w in (32, 128, 256)}
# post-hoc (preregistration_long_xasset.json): Holt on 3 / 7 / 14 days of candles, horizons up to 4 days
MODELS |= {f"holt_w{w}": "HOLT" for w in (288, 672, 1344)}
WINDOWS |= {f"holt_w{w}": w for w in (288, 672, 1344)}
LONG_MODELS = ("holt_w288", "holt_w672", "holt_w1344")


def horizons_of(model: str) -> tuple[int, ...]:
    return C.ALL_HORIZONS if model in LONG_MODELS else C.HORIZONS
DEFAULT = ("ar", "holt", "ses", "expsm", "ets", "ss_expsm", "arima", "theta")
SMOOTH_WINDOW = 4


def window_of(model: str) -> int:
    return WINDOWS.get(model, WINDOW)


def scored_rows(d: pd.DataFrame, stride: int = STRIDE, window: int = WINDOW) -> np.ndarray:
    """Frame rows forecast at this stride: in a fold or the forward window, with a full window."""
    idx = np.arange(len(d))
    return idx[(d["fold"].to_numpy() != "") & (idx % stride == 0) & (idx >= window)]


def _forecaster(model: str):
    f = FC.make_forecaster(FC.ForecasterType[MODELS[model]])
    harm = model.startswith("harm")
    variant = model.split("_")[-1] if harm and model != "harm" else ""
    # only the harmonic variants smooth / detrend, with their own causal implementations (the base
    # helpers are broken at the window edge: 038 D9/D10); harm_kK sets the number of sinusoids
    f.set_detrend("d" in variant and not variant.startswith("k"))
    f.set_smooth("s" in variant and not variant.startswith("k"), SMOOTH_WINDOW)
    if variant.startswith("k"):
        f.n_harmonics = int(variant[1:])
    return f


def _steps(model: str, x: np.ndarray) -> np.ndarray:
    """Forecasts 1..H ahead (H = the model's longest horizon). The library returns the last len(x)
    values of [x, forecasts], so a window shorter than H needs several calls (steps w, 2w, ... capped)."""
    top = max(horizons_of(model))
    w, out, covered = len(x), np.full(top, np.nan), 0
    while covered < top:
        s = min(covered + w, top)
        k = min(w, s)  # forecasts in the returned tail: steps s-k+1 .. s
        tail = np.asarray(_forecaster(model).forecast(x.copy(), s), float)[-k:]
        out[covered:s] = tail[covered - (s - k):]
        covered = s
    return out


def _path(model: str, lc: np.ndarray) -> np.ndarray:
    """Forecast log-close offsets for horizons 1..96 from a window of log closes ending at t (lc has
    window + 1 values, so AR gets window one-candle returns and its return path is summed)."""
    if model == "ar":
        return np.cumsum(_steps(model, np.diff(lc)))
    x = lc[1:]
    return _steps(model, x) - x[-1]


def forecast_rows(d: pd.DataFrame, rows, model: str) -> pd.DataFrame:
    lc = d["lc"].to_numpy(float)
    hd = d["has_data"].to_numpy(bool)
    hz = horizons_of(model)
    out = {f"f_{n}": [] for n in hz}
    failed, skipped, kinds = [], [], Counter()
    w = window_of(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in rows:
            path = np.full(max(hz), np.nan)
            gap = (not hd[t]) or hd[t - w + 1: t + 1].mean() < 0.5
            bad = False
            if not gap:
                try:
                    path = _path(model, lc[t - w: t + 1])
                    bad = not np.all(np.isfinite(path))
                except Exception as e:  # ForecastFailed or a statsmodels error: counted, never zero
                    kinds[type(e).__name__] += 1
                    bad = True
                if bad:
                    path = np.full(max(hz), np.nan)
            for n in hz:
                out[f"f_{n}"].append(path[n - 1])
            failed.append(bad)
            skipped.append(gap)
    df = pd.DataFrame({"date": d["date"].to_numpy()[list(rows)], **out, "failed": failed, "skipped_gap": skipped})
    df.attrs["fail_kinds"] = dict(kinds)
    return df


def _task(args) -> tuple[str, str, pd.DataFrame, dict, float]:
    pair, model = args
    t0 = time.time()
    d = F.read(pair, ["date", "fold", "lc", "has_data"])
    p = forecast_rows(d, scored_rows(d, window=window_of(model)), model).assign(pair=pair)
    return pair, model, p, p.attrs["fail_kinds"], time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    rc = C.refuse_unless_registered()
    models = a.model or list(DEFAULT)
    for amend in {C.POST_HOC[m] for m in models if m in C.POST_HOC}:
        rc = rc or C.refuse_unless_registered(prereg=amend)
    if rc:
        return rc
    tasks = [(p, m) for m in models for p in C.pairs()]
    res: dict[str, list] = {m: [] for m in models}
    t0 = time.time()
    with Pool(a.workers) as pool:
        for pair, model, p, kinds, sec in pool.imap_unordered(_task, tasks):
            res[model].append(p)
            print(f"{model:9s} {pair:10s} rows {len(p):6d} failed {p.failed.mean():.4f} "
                  f"skipped {p.skipped_gap.mean():.4f} {kinds or ''} {sec:6.1f}s  [{time.time() - t0:7.1f}s]",
                  flush=True)
            if len(res[model]) == len(C.pairs()):
                cols = ["pair", "date", *[f"f_{n}" for n in horizons_of(model)], "failed", "skipped_gap"]
                C.write_feather(pd.concat(res[model], ignore_index=True)[cols], C.PRED / f"{model}.feather")
                print(f"wrote {model}", flush=True)
    print(f"stat tier done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
