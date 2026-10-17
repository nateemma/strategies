"""HGB sizer study: the per-fold HGB p_buy behind `cs_hgb` (preregistration.json). ONE fold per process.

The 037 label:buy model (research/indicator_screen/screen_models.py: candidate bars, every scored
column without btc:, the pre-registered HGB settings), fitted on the fold's training rows that END
a purge (48 bars) BEFORE the fold's validation tail, so the threshold pick on the tail reads bars
the model never fitted -- as ARO's threshold is picked on its own validation tail.

Validation tail per pair: the strategy's own split, rows [int(TRAIN_DATA_SPLIT * n), n) of the
pair's bars in [train_start, test_start) (research/width_relative/pick.py `_frames`). `pick.py`
re-derives it from the strategy frames and records any pair whose fit cutoff is later than the tail.

Output, user_data/data/hgb_sizer/ (not committed; sha256 in the plan's threshold_picks):
- HGB_fNN.parquet: date, pair, p_buy for every gbb candidate bar from the tail start to the fold's
  test end (the strategy reads it; bars without a row are scored 0 -> disagree);
- signals_fNN.parquet: the control's signal table ARO_fNN (test window) with p_buy replaced by
  the HGB's, for the instrument's entry and tag checks.

    PYTHONPATH=. ../../.venv/bin/python -m research.hgb_sizer.signals --fold f00
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import screen_models as SM
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel, scored_columns
from research.indicator_screen.screen_single import LabelTargets, prereg

HERE = Path(__file__).resolve().parent
OUT = H.ROOT / "user_data" / "data" / "hgb_sizer"
ARO_SIGNALS = HERE.parent / "nnnc_framework" / "signals"
TRAIN_DATA_SPLIT = 0.8          # Framework/BaseNNStrategy.TRAIN_DATA_SPLIT
TARGET = "label:buy"


def tail_starts(panel: Panel, fold: W.Window) -> dict[int, pd.Timestamp]:
    """{pair code: first validation-tail date} for the fold's training span."""
    out = {}
    for k in range(len(panel.pairs)):
        d = panel.date[(panel.pair == k) & (panel.date >= fold.train_start) & (panel.date < fold.start)]
        if len(d):
            out[k] = d[int(TRAIN_DATA_SPLIT * len(d))]
    return out


def fit_rows(panel: Panel, y: np.ndarray, fold: W.Window, tails: dict) -> np.ndarray:
    cut = pd.Series(tails).reindex(range(len(panel.pairs)))
    stop = pd.DatetimeIndex(cut.to_numpy()[panel.pair]) - W.purge(TARGET) * W.BAR
    m = (panel.data_ok & panel.candidate & np.isfinite(y)
         & np.asarray(panel.date >= fold.train_start) & np.asarray(panel.date < stop))
    return np.flatnonzero(m)


def score_rows(panel: Panel, fold: W.Window, tails: dict) -> np.ndarray:
    cut = pd.Series(tails).reindex(range(len(panel.pairs)))
    start = pd.DatetimeIndex(cut.to_numpy()[panel.pair])
    return np.flatnonzero(panel.candidate & np.asarray(panel.date >= start) & np.asarray(panel.date < fold.end))


def run(fid: str, panel: Panel | None = None, out: Path = OUT) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    pr = prereg()
    fold = next(f for f in W.folds() if f.id == fid)
    panel = panel or Panel()
    y = LabelTargets(panel).y(TARGET, fold)
    tails = tail_starts(panel, fold)
    tr, te = fit_rows(panel, y, fold, tails), score_rows(panel, fold, tails)
    cols = scored_columns(include_btc=False)
    Xtr = SM.matrix(panel, cols, tr)
    usable = np.isfinite(Xtr).any(axis=0)
    m = HistGradientBoostingClassifier(**pr["models"]["hgb"]).fit(Xtr[:, usable], y[tr] > 0.5)
    p = m.predict_proba(SM.matrix(panel, cols, te)[:, usable])[:, 1]
    names = np.asarray(panel.pairs)[panel.pair[te]]
    tab = pd.DataFrame({"date": panel.date[te].tz_convert("UTC"), "pair": names, "p_buy": p})
    out.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out / f"HGB_{fid}.parquet", index=False)

    aro = pd.read_parquet(ARO_SIGNALS / f"ARO_{fid}.parquet")
    d = pd.to_datetime(aro["date"])
    d = d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")
    sig = aro.assign(date=d).drop(columns="p_buy").merge(tab, on=["date", "pair"], how="left")
    sig.to_parquet(out / f"signals_{fid}.parquet", index=False)
    return {"fold": fid, "n_fit": int(len(tr)), "n_scored": int(len(te)),
            "fit_positive_rate": round(float(np.mean(y[tr] > 0.5)), 4),
            "dropped_all_missing": [c for c, u in zip(cols, usable) if not u],
            "tail_starts": {panel.pairs[k]: str(v) for k, v in sorted(tails.items())},
            "sha256": hashlib.sha256((out / f"HGB_{fid}.parquet").read_bytes()).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run(a.fold)
    (OUT / f"HGB_{a.fold}.json").write_text(json.dumps(res, indent=1))
    print(f"{a.fold}: fit {res['n_fit']} rows, scored {res['n_scored']} ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
