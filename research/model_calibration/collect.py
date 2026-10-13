"""Model probability bunching / calibration: DATA COLLECTION ONLY (2026-10-13).

Reads the per-bar test-window signal tables written by `research/nnnc_framework/signals.py`
(p_buy per pair/bar, f00-f14, four NNNC models) and the models' own training labels, computed
through each model class's own label methods (`get_training_labels`, bound on an uninitialised
instance -- no model, no config). No inference, no backtest.

    cd user_data/strategies && PYTHONPATH=.:NNNC ../../.venv/bin/python -m research.model_calibration.collect

Positive class = BUY (2) after conflict resolution, exactly as the model was trained. Labels are
computed per pair on the full history (training uses the fold's training slice; only the last
HORIZON bars of a slice can differ). Forward metrics use close[t] as the reference price.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "NNNC"))

HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"
MODELS = {"ARO": "NNNC_A_Raw_Open", "GBR": "NNNC_GbbRaw_Open", "GFM": "NNNC_GbbFwdMae_Open",
          "MLX": "NNNC_MLX"}
FOLDS = [f"f{i:02d}" for i in range(15)]
EARLY = set(FOLDS[:8])
PLAN = STRAT / "evaluation" / "folds" / "plan_wf_aro_g24.json"
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
FIXED_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0001)
TP, H = 0.022, 48


def labels_for(cls_name: str, pairs) -> dict:
    from evaluation.fingerprint import load_strategy_class
    from research.clfstudy.data import load_pair, populate

    k = load_strategy_class(cls_name)
    s = k.__new__(k)
    s.dbg_verbose, s.dbg_curr_df = False, None
    out = {}
    for p in pairs:
        df = populate(load_pair(p))
        lab = np.asarray(s.get_training_labels(df.copy()))
        c, hi = df["close"].to_numpy(float), df["high"].to_numpy(float)
        n = len(c)
        fwd = np.full(n, np.nan)
        fwd[:-H] = c[H:] / c[:-H] - 1
        # max high over the next H bars (t+1..t+H), and over 96 bars (the 24 h grace window)
        mh = {}
        for w in (H, 96):
            m = np.full(n, np.nan)
            s_hi = pd.Series(hi[::-1]).rolling(w, min_periods=w).max().to_numpy()[::-1]
            m[:-1] = s_hi[1:]
            mh[w] = m / c - 1
        out[p] = pd.DataFrame({"date": pd.to_datetime(df["date"]).astype("datetime64[ns, UTC]"), "label": (lab == 2).astype(int), "fwd": fwd,
                               "hit48": (mh[H] >= TP).astype(float), "hit96": (mh[96] >= TP).astype(float)})
        out[p].loc[np.isnan(mh[96]), "hit96"] = np.nan
        out[p].loc[np.isnan(mh[H]), "hit48"] = np.nan
    return out


def auc(y, p) -> float | None:
    y = np.asarray(y)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return None
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bucket_table(d: pd.DataFrame, bins) -> list:
    g = d.groupby(pd.cut(d["p_buy"], bins, include_lowest=True, duplicates="drop"), observed=True)
    return [{"bin": str(b), "n": int(len(x)), "p_mean": round(float(x.p_buy.mean()), 4),
             "label_rate": round(float(x.label.mean()), 4), "fwd_mean": round(float(x.fwd.mean()), 5),
             "hit48": round(float(x.hit48.mean()), 4), "hit96": round(float(x.hit96.mean()), 4)}
            for b, x in g]


def stats(d: pd.DataFrame, thr: float) -> dict:
    p, y = d["p_buy"].to_numpy(float), d["label"].to_numpy(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ok = d["fwd"].notna()
    return {
        "n": int(len(d)), "mean": round(float(p.mean()), 5), "std": round(float(p.std()), 5),
        "q": {str(q): round(float(np.quantile(p, q)), 4) for q in QS},
        "min": round(float(p.min()), 5), "max": round(float(p.max()), 5),
        "threshold": thr, "frac_ge_thr": round(float((p >= thr).mean()), 5),
        "n_ge_thr": int((p >= thr).sum()),
        "label_rate_test": round(float(y.mean()), 5),
        "label_rate_ge_thr": round(float(y[p >= thr].mean()), 4) if (p >= thr).any() else None,
        "auc": auc(y, p), "brier": round(float(((p - y) ** 2).mean()), 5),
        "logloss": round(float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()), 5),
        "spearman_p_fwd": round(float(pd.Series(p[ok]).corr(d.loc[ok, "fwd"].reset_index(drop=True),
                                                         method="spearman")), 4),
        "spearman_p_fwd_top5pct": None,
    }


def main() -> int:
    t0 = time.time()
    meta = json.loads((SIG / "meta.json").read_text())
    plan = {f["fold_id"]: f for f in json.loads(PLAN.read_text())["folds"]}
    res = {"label": "DATA ONLY -- PROVISIONAL; p_buy from research/nnnc_framework/signals "
                    "(strict model-key tables); labels = each model's own training label",
           "notes": [], "models": {}}
    for m, cls in MODELS.items():
        tabs = {f: pd.read_parquet(SIG / f"{m}_{f}.parquet") for f in FOLDS}
        pairs = sorted(set().union(*[set(t["pair"].unique()) for t in tabs.values()]))
        L = labels_for(cls, pairs)
        mres = {"class": cls, "folds": {}, "per_pair_median": {}, "halves": {}}
        allrows = []
        for f, t in tabs.items():
            thr = round(float(meta[f"{m}_{f}"]["threshold"]), 2)
            parts = []
            t = t.assign(date=pd.to_datetime(t["date"]).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")
                         if pd.to_datetime(t["date"]).dt.tz is None else pd.to_datetime(t["date"]).astype("datetime64[ns, UTC]"))
            for p, g in t.groupby("pair"):
                parts.append(g[["date", "pair", "p_buy"]].merge(L[p], on="date", how="left"))
            d = pd.concat(parts, ignore_index=True)
            miss = int(d["label"].isna().sum())
            d = d.dropna(subset=["label"])
            d["label"] = d["label"].astype(int)
            st = stats(d, thr)
            st["unmatched_bars"] = miss
            # training-window label base rate (anchored, data start -> train_end)
            tr_end = pd.Timestamp(plan[f]["train_end"], tz="UTC")
            tr_start = pd.Timestamp(plan[f]["train_start"], tz="UTC")
            tr = pd.concat([x.loc[(x.date >= tr_start) & (x.date <= tr_end), "label"] for x in L.values()])
            st["label_rate_train"] = round(float(tr.mean()), 5)
            st["train_bars"] = int(len(tr))
            st["train_days"] = int((tr_end - tr_start).days)
            deciles = np.unique(np.quantile(d["p_buy"], np.linspace(0, 1, 11)))
            st["reliability_deciles"] = bucket_table(d, deciles)
            st["reliability_fixed"] = bucket_table(d, FIXED_BINS)
            top = d[d.p_buy >= d.p_buy.quantile(0.95)]
            okt = top["fwd"].notna()
            st["spearman_p_fwd_top5pct"] = round(float(top.loc[okt, "p_buy"].corr(top.loc[okt, "fwd"], method="spearman")), 4)
            mres["folds"][f] = st
            d["fold"] = f
            allrows.append(d)
        A = pd.concat(allrows, ignore_index=True)
        A["half"] = np.where(A["fold"].isin(EARLY), "early", "late")
        for h, d in A.groupby("half"):
            thr_d = {f: mres["folds"][f]["threshold"] for f in FOLDS}
            ge = d["p_buy"] >= d["fold"].map(thr_d)
            mres["halves"][h] = {
                "n": int(len(d)), "label_rate": round(float(d.label.mean()), 5),
                "auc": auc(d.label.to_numpy(), d.p_buy.to_numpy()),
                "frac_ge_thr": round(float(ge.mean()), 5),
                "label_rate_ge_thr": round(float(d.label[ge].mean()), 4) if ge.any() else None,
                "fwd_ge_thr": round(float(d.fwd[ge].mean()), 5) if ge.any() else None,
                "hit48_ge_thr": round(float(d.hit48[ge].mean()), 4) if ge.any() else None,
                "reliability_fixed": bucket_table(d, FIXED_BINS),
            }
        for p, g in A.groupby("pair"):
            mres["per_pair_median"][p] = {
                "median": round(float(g.p_buy.median()), 4), "p95": round(float(g.p_buy.quantile(0.95)), 4),
                "label_rate": round(float(g.label.mean()), 4),
                "by_fold_median": {f: round(float(x.p_buy.median()), 4) for f, x in g.groupby("fold")}}
        res["models"][m] = mres
        print(m, "done %.0fs" % (time.time() - t0), flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    (HERE / "data.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
