"""Live-buffer check for the in-strategy transition stand-in (NNNC/_transition_standin.py).

A live bot computes indicators on its ~920-candle buffer, a backtest on the whole range. The long windows
(z-scores, data-ok) come from the history store in both, so they are equal by construction; what can
differ is the 24 indicators and raw8 under a short warm-up. Per pair, at sampled candles of [start, end),
this recomputes the inputs on the trailing BUFFER candles only (harvest.populate_path, the strategy's
indicator steps) and compares them, the stand-in's probabilities and its p80 fire flags with the
full-range values.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.live_check --cutoff 20260614 \
        --start 2026-06-14 --end 2026-09-07 --n 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNNC")]
HERE = Path(__file__).resolve().parent
BUFFER = 920


def main(argv=None) -> int:
    import _transition_standin as TS
    from freqtrade.data import history
    from research.indicator_screen import harvest as H
    from research.nnmt.transition import _pairs
    from utils.CloseHistoryStore import _grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--buffer", type=int, default=BUFFER)
    a = ap.parse_args(argv)
    cfg, aro = H.base_config(), H.load_aro()
    st = TS.StandIn(a.cutoff)
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")

    def load(pair, filled):
        df = history.load_pair_history(pair=pair, timeframe="15m", datadir=cfg["datadir"], fill_up_missing=filled,
                                       data_format=cfg.get("dataformat_ohlcv", "feather"))
        return df[(df["date"] >= pd.Timestamp(H.W.DATA_START, tz="UTC")) & (df["date"] < end)].reset_index(drop=True)

    btc = load(TS.CONTEXT_PAIR, True)
    hb = _grid(load(TS.CONTEXT_PAIR, False), 15)
    btc_pos = {d: i for i, d in enumerate(btc["date"])}
    rng = np.random.default_rng(0)
    feats = TS.feature_names()
    diffs = {c: [] for c in feats}
    rows = []
    for pair in _pairs():
        ohlcv = load(pair, True)
        hp = _grid(load(pair, False), 15)
        full = H.populate_path(aro, ohlcv, pair)
        Xf, trf = TS.features(full, btc, hp, hb)
        cand = np.flatnonzero(trf & (full["date"] >= start).to_numpy() & (full["date"] < end).to_numpy())
        cand = cand[cand >= a.buffer]
        if not len(cand):
            continue
        for t in np.sort(rng.choice(cand, min(a.n, len(cand)), replace=False)):
            sl = H.populate_path(aro, ohlcv.iloc[t - a.buffer + 1:t + 1].reset_index(drop=True), pair)
            bi = btc_pos.get(ohlcv["date"].iloc[t])
            bsl = btc.iloc[max(0, bi - a.buffer + 1):bi + 1].reset_index(drop=True)
            Xs, trs = TS.features(sl, bsl, hp, hb)
            row_f, row_s = Xf.iloc[[t]], Xs.iloc[[len(Xs) - 1]]
            for c in feats:
                diffs[c].append(abs(float(row_f[c].iloc[0]) - float(row_s[c].iloc[0])))
            rec = {"pair": pair, "date": str(ohlcv["date"].iloc[t]), "tradeable_buffer": bool(trs[-1])}
            for cell in st.meta["cells"]:
                pf = st.predict(cell, row_f, np.array([True]))[0]
                ps = st.predict(cell, row_s, np.array([bool(trs[-1])]))[0]
                thr = st.threshold(cell, 80)
                rec[cell] = {"p_full": pf, "p_buffer": ps, "fire_full": bool(pf >= thr),
                             "fire_buffer": bool(np.nan_to_num(ps, nan=-np.inf) >= thr)}
            rows.append(rec)
        print(pair, len(rows), flush=True)
    out = {"buffer": a.buffer, "cutoff": a.cutoff, "samples": len(rows),
           "feature_max_abs_diff": {c: (max(v) if v else None) for c, v in diffs.items()},
           "features_with_diff": sorted([c for c, v in diffs.items() if v and max(v) > 1e-9]),
           "cells": {}}
    for cell in st.meta["cells"]:
        pf = np.array([r[cell]["p_full"] for r in rows]); ps = np.array([r[cell]["p_buffer"] for r in rows], float)
        ff = np.array([r[cell]["fire_full"] for r in rows]); fb = np.array([r[cell]["fire_buffer"] for r in rows])
        out["cells"][cell] = {"max_abs_p_diff": float(np.nanmax(np.abs(pf - ps))), "fire_agreement": float((ff == fb).mean()),
                              "fire_full": int(ff.sum()), "fire_buffer": int(fb.sum())}
    out["tradeable_lost_in_buffer"] = int(sum(not r["tradeable_buffer"] for r in rows))
    (HERE / "live_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "feature_max_abs_diff"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
