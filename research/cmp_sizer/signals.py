"""Composite sizer study: the per-fold 037 'dip' composite behind `cs_cmp` (preregistration.json).

The composite of 037 addendum 4 (research/indicator_screen/composite.py, members and signs from
vol/composite.json): each member z-scored with the mean / sd of the fold's tradeable bars (data_ok,
volume > 0, per-pair warm-up excluded) from train_start to the pair's validation-tail start, signed,
missing -> 0, averaged. No label is used, so there is no fit and no purge; the threshold pick on the
tail reads bars the normalisation never saw. Tail starts: research/hgb_sizer/signals.tail_starts.

Output, user_data/data/cmp_sizer/ (not committed; sha256 in the plan's threshold_picks). The files
keep the HGB sizer's names so its strategy, pick and instrument read them unchanged:
- HGB_fNN.parquet: date, pair, p_buy (= the composite) for every gbb candidate bar from the tail
  start to the fold's test end; HGB_fNN.json: meta (fit_last is empty: nothing is fitted);
- signals_fNN.parquet: the control's ARO_fNN table with p_buy replaced, for the instrument.

    PYTHONPATH=. ../../.venv/bin/python -m research.cmp_sizer.signals
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from research.hgb_sizer import signals as HS
from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W
from research.indicator_screen.composite import OUT as COMPOSITE
from research.indicator_screen.panel import Panel

OUT = H.ROOT / "user_data" / "data" / "cmp_sizer"


def members() -> tuple[list[str], np.ndarray]:
    mem = json.loads(COMPOSITE.read_text())["members"]
    return [m["column"] for m in mem], np.array([m["dir"] for m in mem], float)


def norm_rows(panel: Panel, fold: W.Window, tails: dict) -> np.ndarray:
    cut = pd.Series(tails).reindex(range(len(panel.pairs)))
    stop = pd.DatetimeIndex(cut.to_numpy()[panel.pair])
    ok = panel.data_ok & np.asarray(panel.date >= fold.train_start) & np.asarray(panel.date < stop)
    s = 0
    for n in panel.lengths:
        ok[s: s + W.WARMUP_BARS] = False
        s += n
    return ok


def composite(X: np.ndarray, sign: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(X[rows], 0), np.nanstd(X[rows], 0)
    return np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1) * sign).mean(1)


def run_all(out=OUT) -> dict:
    panel = Panel(tradeable_only=True)
    cols, sign = members()
    d = panel.columns(cols)
    X = np.column_stack([d[c] for c in cols]).astype(float)
    rd = HS.raw_dates(panel.pairs)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in W.folds():
        t0 = time.perf_counter()
        tails = HS.tail_starts(panel, f, rd)
        nr = norm_rows(panel, f, tails)
        c = composite(X, sign, nr)
        te = HS.score_rows(panel, f, tails)
        tab = pd.DataFrame({"date": panel.date[te].tz_convert("UTC"),
                            "pair": np.asarray(panel.pairs)[panel.pair[te]], "p_buy": c[te]})
        tab.to_parquet(out / f"HGB_{f.id}.parquet", index=False)
        aro = pd.read_parquet(HS.ARO_SIGNALS / f"ARO_{f.id}.parquet")
        dd = pd.to_datetime(aro["date"])
        dd = dd.dt.tz_localize("UTC") if dd.dt.tz is None else dd.dt.tz_convert("UTC")
        aro.assign(date=dd).drop(columns="p_buy").merge(tab, on=["date", "pair"], how="left") \
            .to_parquet(out / f"signals_{f.id}.parquet", index=False)
        last = {panel.pairs[k]: str(panel.date[nr & (panel.pair == k)].max())
                for k in range(len(panel.pairs)) if (nr & (panel.pair == k)).any()}
        meta = {"fold": f.id, "members": cols, "n_norm": int(nr.sum()), "n_scored": int(len(te)),
                "fit_last": {}, "norm_last": last,
                "tail_starts": {panel.pairs[k]: str(v) for k, v in sorted(tails.items())},
                "sha256": hashlib.sha256((out / f"HGB_{f.id}.parquet").read_bytes()).hexdigest()}
        (out / f"HGB_{f.id}.json").write_text(json.dumps(meta, indent=1))
        res[f.id] = meta
        print(f"{f.id}: normalised on {meta['n_norm']} rows, scored {meta['n_scored']} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    run_all()
