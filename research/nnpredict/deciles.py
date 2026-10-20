"""039 decile check (user request, 2026-10-19): does NNPredict's predicted gain RANK outcomes? If the
strongest predictions are not clearly the best trades, no entry threshold can rescue the standalone
entry; if they are, a rate-matched threshold (NNNC's method) is worth a pre-registered amendment.

Diagnostic only (no verdict): early half (f00-f07) only, so the late half stays unread for any later
threshold amendment. Per fold, the model's predicted gain on every test-window candle (signal tables,
research/nnnc_framework/signals.py --models NNP_RIDGE) is ranked into deciles WITHIN the fold; each
candle's outcome is 038's realised trade return (entry next open, exit H candles later, eligible
candles only: research/forecast frames), H = 4 (the default model horizon), 8 and 32 (~ the hold).

Reported per decile: candles, mean gross return (bp), mean net of the 30 bp round trip, and the edge
over the fold's all-candle mean (random entry); plus Spearman(decile, mean return), the top-decile
minus bottom-decile spread, and the same for the gated entries (enter_long) vs all candles.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.deciles [--model NNP_RIDGE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"
EARLY = [f"f{i:02d}" for i in range(8)]
HORIZONS = (4, 8, 32)  # 32 = 8 h, the stored horizon nearest the median hold (frames have no 24)
COST_BP = 30.0


def table(model: str) -> pd.DataFrame:
    from research.forecast import frames as F
    from research.forecast import common as C

    parts = []
    for f in EARLY:
        t = pd.read_parquet(SIG / f"{model}_{f}.parquet")[["pair", "date", "predict_gain", "enter_long"]]
        t["date"] = pd.to_datetime(t["date"], utc=True)
        t["fold"] = f
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    out = []
    for pair, g in t.groupby("pair"):
        fr = F.read(pair, ["date", *[c for n in HORIZONS for c in (f"trade_ret_{n}", f"eligible_{n}")]])
        fr["date"] = pd.to_datetime(fr["date"], utc=True)
        out.append(g.merge(fr, on="date", how="left"))
    return pd.concat(out, ignore_index=True)


def deciles(t: pd.DataFrame, n: int) -> dict:
    d = t[t[f"eligible_{n}"].fillna(False).astype(bool) & np.isfinite(t["predict_gain"])].copy()
    d["ret_bp"] = d[f"trade_ret_{n}"] * 1e4
    d["dec"] = d.groupby("fold")["predict_gain"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    base = d.groupby("fold")["ret_bp"].transform("mean")
    d["edge"] = d["ret_bp"] - base
    tab = d.groupby("dec").agg(candles=("ret_bp", "size"), gross_bp=("ret_bp", "mean"), edge_bp=("edge", "mean"))
    tab["net_bp"] = tab["gross_bp"] - COST_BP
    per_fold_top = d[d["dec"] == 9].groupby("fold")["edge"].mean()
    gated = d[d["enter_long"] == 1]
    return {"horizon": n, "candles": int(len(d)),
            "deciles": {int(k): {c: round(float(v), 2) for c, v in r.items()} for k, r in tab.iterrows()},
            "spearman_decile_return": round(float(spearmanr(tab.index, tab["gross_bp"]).statistic), 3),
            "top_minus_bottom_bp": round(float(tab.loc[9, "gross_bp"] - tab.loc[0, "gross_bp"]), 2),
            "top_decile_edge_bp_per_fold": {f: round(float(v), 2) for f, v in per_fold_top.items()},
            "top_decile_edge_positive_folds": int((per_fold_top > 0).sum()),
            "gated": {"candles": int(len(gated)), "gross_bp": round(float(gated["ret_bp"].mean()), 2),
                      "edge_bp": round(float(gated["edge"].mean()), 2)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NNP_RIDGE")
    a = ap.parse_args(argv)
    t = table(a.model)
    out = {"model": a.model, "folds": EARLY, "note": "diagnostic, early half only; no verdict",
           "results": [deciles(t, n) for n in HORIZONS]}
    (HERE / f"deciles_{a.model.lower()}.json").write_text(json.dumps(out, indent=1))
    for r in out["results"]:
        print(f"H={r['horizon']}: candles {r['candles']}, spearman(decile, return) {r['spearman_decile_return']}, "
              f"top-bottom {r['top_minus_bottom_bp']} bp, top-decile edge >0 in {r['top_decile_edge_positive_folds']}/8 folds; "
              f"gated entries {r['gated']}")
        for k, v in r["deciles"].items():
            print(f"   decile {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
