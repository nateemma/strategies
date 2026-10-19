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
          "ss_expsm": "SS_EXPONENTAL", "arima": "ARIMA", "theta": "THETA"}


def scored_rows(d: pd.DataFrame, stride: int = STRIDE) -> np.ndarray:
    """Frame rows forecast at this stride: in a fold or the forward window, with a full window."""
    idx = np.arange(len(d))
    return idx[(d["fold"].to_numpy() != "") & (idx % stride == 0) & (idx >= WINDOW)]


def _forecaster(model: str):
    f = FC.make_forecaster(FC.ForecasterType[MODELS[model]])
    f.set_detrend(False)
    f.set_smooth(False)
    return f


def _path(model: str, lc: np.ndarray) -> np.ndarray:
    """Forecast log-close offsets for horizons 1..96 from a WINDOW-long log-close window ending at t
    (lc has WINDOW + 1 values, so AR has WINDOW returns)."""
    path = np.full(96, np.nan)
    if model == "ar":
        x = np.diff(lc)
        a = np.asarray(_forecaster(model).forecast(x.copy(), WINDOW), float)  # returns 1..64
        b = np.asarray(_forecaster(model).forecast(x.copy(), 96), float)      # returns 33..96
        path[:WINDOW] = np.cumsum(a)
        path[WINDOW:] = path[WINDOW - 1] + np.cumsum(b[WINDOW - 32:])
        return path
    x = lc[1:]
    a = np.asarray(_forecaster(model).forecast(x.copy(), WINDOW), float)
    b = np.asarray(_forecaster(model).forecast(x.copy(), 96), float)
    path[:WINDOW] = a - x[-1]
    path[WINDOW:] = b[WINDOW - 32:] - x[-1]
    return path


def forecast_rows(d: pd.DataFrame, rows, model: str) -> pd.DataFrame:
    lc = d["lc"].to_numpy(float)
    hd = d["has_data"].to_numpy(bool)
    out = {f"f_{n}": [] for n in C.HORIZONS}
    failed, skipped, kinds = [], [], Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in rows:
            path = np.full(96, np.nan)
            gap = (not hd[t]) or hd[t - WINDOW + 1: t + 1].mean() < 0.5
            bad = False
            if not gap:
                try:
                    path = _path(model, lc[t - WINDOW: t + 1])
                    bad = not np.all(np.isfinite(path))
                except Exception as e:  # ForecastFailed or a statsmodels error: counted, never zero
                    kinds[type(e).__name__] += 1
                    bad = True
                if bad:
                    path = np.full(96, np.nan)
            for n in C.HORIZONS:
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
    p = forecast_rows(d, scored_rows(d), model).assign(pair=pair)
    return pair, model, p, p.attrs["fail_kinds"], time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    rc = C.refuse_unless_registered()
    if rc:
        return rc
    models = a.model or list(MODELS)
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
                cols = ["pair", "date", *[f"f_{n}" for n in C.HORIZONS], "failed", "skipped_gap"]
                C.write_feather(pd.concat(res[model], ignore_index=True)[cols], C.PRED / f"{model}.feather")
                print(f"wrote {model}", flush=True)
    print(f"stat tier done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
