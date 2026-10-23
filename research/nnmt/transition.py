"""040 regime-transition target and its screen (research R11; preregistration_transition.json).

Causal regime state S_L(t) of a close series:
    z_L(t) = log(close_t / SMA_L(t)) / (sigma(t) * sqrt(L)),  sigma = trailing std of 1-bar log returns
    BULL if z >= +DEADBAND, BEAR if z <= -DEADBAND, else SIDEWAYS
Transition label at horizon H (L = H), a label (it reads t+H):
    UP   if S_H(t) != BULL and S_H(t+H) == BULL
    DOWN if S_H(t) != BEAR and S_H(t+H) == BEAR
Levels: "pair" (the pair's own close) and "btc" (BTC's close, joined by date).

Steps (one process each; the fits never run alongside an MLX trainer):
  labels      labels + base rates per fold and half -> user_data/data/nnmt_transition/labels.parquet
  causality   16-cut truncation test of z_L, the state, the label and every stand-in input -> exit 1 on a mismatch
  screen      --fold fNN: per (direction, H, level) a HistGradientBoosting one-vs-rest stand-in, fit on the
              fold's training window minus its last 90 days (purge H), scored on that held-out tail (the
              arms' threshold source) and on the test window -> pred_fNN.parquet
  judge       AUC vs the base rate and the trailing-trend baseline, events, the P&L proxy, pass, the
              selected cell per direction                                  -> screens_transition.json

Regime-state amendment (preregistration_regime_state.json): the FUTURE state Y = S_H(t+H), 3 classes.
  state-screen  --fold fNN: one multiclass HGB per (H, level), same inputs/fit/tail as `screen`, plus the
                Markov baseline P(Y | S_H(t)) fit on the same rows                -> state_pred_fNN.parquet
  state-judge   AUC per class (BULL, BEAR) vs max(0.5, Markov, trailing z_H), events, proxy, pass, selection
                                                                                    -> screens_regime_state.json

  state-nn      --models MT_LR_RI_RG,MT_LR_RI_RGM --cell pair_288: each NN's regime head (signal tables) scored with
                the same rule on that cell (Markov fit on the fold's training rows), plus shared-vs-separate
                                                                                    -> screens_regime_state.json["nn"]

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.transition labels|causality|judge|state-judge
    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.transition screen|state-screen --fold f00
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_transition.json"
PREREG_STATE = HERE / "preregistration_regime_state.json"
CLASSES = {"BEAR": -1.0, "SIDEWAYS": 0.0, "BULL": 1.0}
OUT = STRAT.parents[1] / "user_data" / "data" / "nnmt_transition"
PLAN = STRAT / "evaluation" / "folds" / "plan_nnp_lstm_ind_op.json"   # the 15 fold windows
BAR = pd.Timedelta(minutes=15)


def _pr() -> dict:
    return json.loads(PREREG.read_text())


# ---------------------------------------------------------------------------- the state and the label


def zscore(close: pd.Series, L: int, sigma_window: int, sigma_min: int) -> pd.Series:
    lc = np.log(close.astype(float))
    sma = close.astype(float).rolling(L, min_periods=L).mean()
    sig = lc.diff().rolling(sigma_window, min_periods=sigma_min).std()
    return (lc - np.log(sma)) / (sig * np.sqrt(L))


def state(z: pd.Series, deadband: float) -> pd.Series:
    """+1 BULL, -1 BEAR, 0 SIDEWAYS; NaN where z is NaN."""
    s = pd.Series(np.where(z >= deadband, 1.0, np.where(z <= -deadband, -1.0, 0.0)), index=z.index)
    return s.where(z.notna())


def labels(close: pd.Series, H: int, pr: dict) -> pd.DataFrame:
    """UP / DOWN indicators (float; NaN where either end's state is unknown, incl. within H of the end)."""
    d = pr["definition"]
    s = state(zscore(close, H, d["sigma_window"], d["sigma_min_periods"]), d["deadband"])
    fut = s.shift(-H)
    ok = s.notna() & fut.notna()
    up = ((s != 1) & (fut == 1)).astype(float).where(ok)
    down = ((s != -1) & (fut == -1)).astype(float).where(ok)
    return pd.DataFrame({"up": up, "down": down, "state": s})


def future_state(close: pd.Series, H: int, pr: dict) -> pd.DataFrame:
    """Regime-state amendment: the current state S_H(t), z_H(t), and the target Y = S_H(t+H) (NaN where either
    end's state is unknown, incl. within H of the end)."""
    d = pr["definition"]
    z = zscore(close, H, d["sigma_window"], d["sigma_min_periods"])
    s = state(z, d["deadband"])
    fut = s.shift(-H)
    return pd.DataFrame({"state": s, "z": z, "fstate": fut.where(s.notna() & fut.notna())})


def markov_scores(cur_fit: np.ndarray, fut_fit: np.ndarray, cur_eval: np.ndarray) -> pd.DataFrame:
    """P(Y = k | S(t) = j) from the FIT rows only, looked up by each evaluated row's current state."""
    cur_fit, fut_fit = np.asarray(cur_fit, float), np.asarray(fut_fit, float)
    ok = np.isfinite(cur_fit) & np.isfinite(fut_fit)
    m = pd.crosstab(cur_fit[ok], fut_fit[ok], normalize="index").reindex(
        index=list(CLASSES.values()), columns=list(CLASSES.values())).fillna(0.0)
    e = pd.Series(np.asarray(cur_eval, float))
    return pd.DataFrame({k: e.map(m[v]).to_numpy(float) for k, v in CLASSES.items()})


# ---------------------------------------------------------------------------- data


def _pairs() -> list[str]:
    from research.indicator_screen import panel as P

    return P.screen_pairs()


def _read(pair: str, cols: list[str]) -> pd.DataFrame:
    from research.indicator_screen import harvest as H

    return H.read_cache(pair, cols)


def _folds() -> list[dict]:
    return json.loads(PLAN.read_text())["folds"]


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _half(fid: str, pr: dict) -> str:
    return "early" if fid in pr["early"] else "late"


def build_labels() -> pd.DataFrame:
    """Per pair: date, pair, tradeable, close, then for every H and level: up/down labels, and the trailing
    states the baselines and the proxy need (z of S_{H/4}, forward H-bar log return)."""
    pr = _pr()
    d = pr["definition"]
    btc = _read("BTC/USDT", ["close"]).set_index("date")["close"]
    frames = []
    for pair in _pairs():
        df = _read(pair, ["close", "volume", "meta:data_ok"])
        out = pd.DataFrame({"date": df["date"], "pair": pair,
                            "tradeable": (df["meta:data_ok"].to_numpy() > 0) & (df["volume"].to_numpy() > 0)})
        close = df["close"].astype(float)
        bclose = df["date"].map(btc).astype(float)
        for H in pr["grid"]["H"]:
            for level, c in (("pair", close), ("btc", bclose)):
                lab = labels(c, H, pr)
                out[f"up_{level}_{H}"] = lab["up"].to_numpy()
                out[f"down_{level}_{H}"] = lab["down"].to_numpy()
                out[f"ztrend_{level}_{H}"] = zscore(c, max(1, H // 4), d["sigma_window"], d["sigma_min_periods"]).to_numpy()
                out[f"fwd_{level}_{H}"] = (np.log(c.shift(-H)) - np.log(c)).to_numpy()
                fs = future_state(c, H, pr)  # regime-state amendment
                out[f"state_{level}_{H}"] = fs["state"].to_numpy()
                out[f"z_{level}_{H}"] = fs["z"].to_numpy()
                out[f"fstate_{level}_{H}"] = fs["fstate"].to_numpy()
            # the pair's own forward return (what a long position in THIS pair earns), for the proxy
            out[f"fwd_trade_{H}"] = (np.log(close.shift(-H)) - np.log(close)).to_numpy()
        frames.append(out)
    t = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT / "labels.parquet", index=False)
    return t


def base_rates(t: pd.DataFrame) -> dict:
    pr = _pr()
    rates = {}
    for f in _folds():
        m = t["tradeable"] & (t["date"] >= _ts(f["test_start"])) & (t["date"] < _ts(f["test_end"]))
        g = t[m]
        r = {}
        for H in pr["grid"]["H"]:
            for level in pr["grid"]["level"]:
                for dirn in ("up", "down"):
                    c = f"{dirn}_{level}_{H}"
                    y = g[c].dropna()
                    r[c] = {"rate": float(y.mean()) if len(y) else None, "events": int(_events(g, c))}
        rates[f["fold_id"]] = r
    return rates


def _events(g: pd.DataFrame, col: str) -> int:
    """Run starts of a positive label, per pair (a run = consecutive positive bars)."""
    n = 0
    for _, s in g.groupby("pair")[col]:
        y = s.fillna(0).to_numpy() > 0.5
        n += int(np.sum(y & ~np.r_[False, y[:-1]]))
    return n


# ---------------------------------------------------------------------------- causality (SC-005)


def causality(cuts: int = 16, warmup: int = 3000) -> dict:
    """Truncation test: z_L and the state from a prefix equal the full series' on the overlap; the label
    from a prefix ending at a cut equals the full label for t <= cut - H (and is NaN after); every
    stand-in input column from the panel cache is re-checked the same way via 037's causality table."""
    pr = _pr()
    d = pr["definition"]
    res = {"z_state": {"pass": True, "comparisons": 0}, "label": {"pass": True, "comparisons": 0}}
    for pair in pr["causality"]["pairs"]:
        close = _read(pair, ["close"])["close"].astype(float).reset_index(drop=True)
        n = len(close)
        for cut in np.linspace(warmup, n - 1, cuts).astype(int):
            pre = close.iloc[: cut + 1]
            for H in pr["grid"]["H"]:
                for L in (H, max(1, H // 4)):
                    a = zscore(close, L, d["sigma_window"], d["sigma_min_periods"]).to_numpy()[: cut + 1]
                    b = zscore(pre, L, d["sigma_window"], d["sigma_min_periods"]).to_numpy()
                    res["z_state"]["comparisons"] += 1
                    if not _equal(a, b):
                        res["z_state"]["pass"] = False
                        res["z_state"].setdefault("first", f"{pair} cut {cut} L {L}")
                full, part = labels(close, H, pr), labels(pre, H, pr)
                k = cut + 1 - H
                for c in ("up", "down"):
                    res["label"]["comparisons"] += 1
                    if not _equal(full[c].to_numpy()[:k], part[c].to_numpy()[:k]) or part[c].iloc[k:].notna().any():
                        res["label"]["pass"] = False
                        res["label"].setdefault("first", f"{pair} cut {cut} H {H} {c}")
    # every stand-in input must carry an explicit PASS in 037's truncation tables (btc: columns are the
    # same harvest on BTC's series, so their pair-level verdict covers them)
    verdicts = {}
    for f in ("causality.json", "causality_extras.json"):
        cz = json.loads((STRAT / "research" / "indicator_screen" / f).read_text())
        verdicts |= {c: r.get("verdict") for c, r in cz.get("columns", {}).items() if isinstance(r, dict)}
    own = sorted({c[4:] if c.startswith("btc:") else c for c in stand_in_columns(pr)})
    not_pass = {c: verdicts.get(c, "UNTESTED") for c in own if verdicts.get(c) != "PASS"}
    res["inputs"] = {"pass": not not_pass, "not_pass": not_pass, "checked": len(own),
                     "source": "research/indicator_screen/causality{,_extras}.json"}
    res["pass"] = all(r["pass"] for r in res.values() if isinstance(r, dict) and "pass" in r)
    return res


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isnan(a) & np.isnan(b)
    return bool(np.all(both | np.isclose(a, b, rtol=1e-9, atol=0.0)))


# ---------------------------------------------------------------------------- the stand-in screen


def stand_in_columns(pr: dict) -> list[str]:
    cols = [f"base:{c}" for c in pr["stand_in"]["indicators"]] + [f"raw8:{c}" for c in pr["stand_in"]["raw8"]]
    return cols + [f"btc:{c}" for c in cols if c.startswith("raw8:")]


def _features(pair: str, pr: dict, btc: pd.DataFrame) -> pd.DataFrame:
    own = [c for c in stand_in_columns(pr) if not c.startswith("btc:")]
    df = _read(pair, own)
    X = df[["date", *own]].copy()
    for c in [c for c in stand_in_columns(pr) if c.startswith("btc:")]:
        X[c] = df["date"].map(btc[c[4:]]).astype(float).to_numpy()
    d = pr["definition"]
    close = _read(pair, ["close"]).set_index("date")["close"].astype(float)
    bclose = btc["close"]
    for L in pr["stand_in"]["state_lengths"]:
        X[f"z_pair_{L}"] = zscore(close, L, d["sigma_window"], d["sigma_min_periods"]).reindex(X["date"]).to_numpy()
        X[f"z_btc_{L}"] = X["date"].map(zscore(bclose, L, d["sigma_window"], d["sigma_min_periods"])).astype(float).to_numpy()
    return X


def screen_fold(fid: str) -> Path:
    from sklearn.ensemble import HistGradientBoostingClassifier

    pr = _pr()
    f = next(x for x in _folds() if x["fold_id"] == fid)
    lab = pd.read_parquet(OUT / "labels.parquet")
    btc_cols = ["close", *[c[4:] for c in stand_in_columns(pr) if c.startswith("btc:")]]
    btc = _read("BTC/USDT", btc_cols).set_index("date")
    X = pd.concat([_features(p, pr, btc).assign(pair=p) for p in _pairs()], ignore_index=True)
    t = lab.drop(columns=[c for c in X.columns if c in lab.columns and c not in ("date", "pair")]).merge(X, on=["date", "pair"], how="left")
    feats = [c for c in X.columns if c not in ("date", "pair")]
    ts, te = _ts(f["test_start"]), _ts(f["test_end"])
    tail0 = ts - pd.Timedelta(days=pr["stand_in"]["tail_days"])
    rows = []
    for H in pr["grid"]["H"]:
        purge = H * BAR
        tr = t["tradeable"] & (t["date"] < tail0 - purge)
        tr &= (t["date"].dt.minute == 0) if pr["stand_in"]["hourly_train"] else True
        tail = t["tradeable"] & (t["date"] >= tail0) & (t["date"] < ts - purge)
        test = t["tradeable"] & (t["date"] >= ts) & (t["date"] < te)
        for level in pr["grid"]["level"]:
            for dirn in ("up", "down"):
                col = f"{dirn}_{level}_{H}"
                m = tr & t[col].notna()
                y = t.loc[m, col].to_numpy() > 0.5
                if y.all() or not y.any():
                    continue
                clf = HistGradientBoostingClassifier(**pr["stand_in"]["hgb"])
                clf.fit(t.loc[m, feats].to_numpy(np.float32), y)
                for part, mask in (("tail", tail), ("test", test)):
                    g = t.loc[mask, ["date", "pair"]].copy()
                    g["cell"] = col
                    g["part"] = part
                    g["p"] = clf.predict_proba(t.loc[mask, feats].to_numpy(np.float32))[:, 1]
                    rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"pred_{fid}.parquet"
    out.to_parquet(p, index=False)
    print(fid, len(out), "rows", flush=True)
    return p


def _auc(score: np.ndarray, y: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(score) & np.isfinite(y)
    yy = y[ok] > 0.5
    if yy.all() or not yy.any():
        return None
    return float(roc_auc_score(yy, score[ok]))


def judge() -> dict:
    pr = _pr()
    lab = pd.read_parquet(OUT / "labels.parquet")
    preds = pd.concat([pd.read_parquet(OUT / f"pred_{f['fold_id']}.parquet").assign(fold=f["fold_id"])
                       for f in _folds()], ignore_index=True)
    test = preds[preds["part"] == "test"].merge(lab, on=["date", "pair"], how="left")
    rule = pr["screen_rule"]
    cells = {}
    for cell, g in test.groupby("cell"):
        dirn, level, H = cell.split("_")
        H = int(H)
        r = {"auc": {}, "baseline_trend_auc": {}, "margin": {}, "events": {}, "proxy_bp": {}}
        for half in ("early", "late"):
            h = g[g["fold"].map(lambda x: _half(x, pr)) == half]
            y = h[cell].to_numpy(float)
            trend = h[f"ztrend_{level}_{H}"].to_numpy(float) * (1 if dirn == "up" else -1)
            a, b = _auc(h["p"].to_numpy(float), y), _auc(trend, y)
            r["auc"][half], r["baseline_trend_auc"][half] = a, b
            r["margin"][half] = None if a is None else a - max(0.5, b if b is not None else 0.5)
            r["events"][half] = int(_events(h, cell))
            r["proxy_bp"][half] = _proxy(h, dirn, H, rule)
        ok = all(r["margin"][x] is not None and r["margin"][x] >= rule["min_auc_margin"] for x in ("early", "late"))
        ok &= all(r["events"][x] >= rule["min_events_per_half"] for x in ("early", "late"))
        ok &= all(r["proxy_bp"][x] is not None and r["proxy_bp"][x]["pass"] for x in ("early", "late"))
        r["pass"] = bool(ok)
        cells[cell] = r
    sel = {}
    for dirn in ("up", "down"):
        passing = {c: r for c, r in cells.items() if c.startswith(dirn) and r["pass"]}
        sel[dirn] = max(passing, key=lambda c: passing[c]["margin"]["late"]) if passing else None
    out = {"cells": cells, "selected": sel, "pass": any(sel.values())}
    path = HERE / "screens_transition.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**prev, **out}, indent=1, default=str))
    return out


def _proxy(h: pd.DataFrame, dirn: str, H: int, rule: dict) -> dict | None:
    """Top decile of scored bars per fold. UP: the pair's forward H-bar return net of the round trip > 0.
    DOWN: the top decile's forward return is below the fold's mean by >= the round trip."""
    tops, alls = [], []
    for _, g in h.groupby("fold"):
        fwd = g[f"fwd_trade_{H}"].to_numpy(float)
        ok = np.isfinite(fwd)
        if ok.sum() < 10:
            continue
        p = g["p"].to_numpy(float)[ok]
        cut = np.quantile(p, 1 - rule["proxy_top_share"])
        tops.append(fwd[ok][p >= cut])
        alls.append(fwd[ok])
    if not tops:
        return None
    top, al = np.concatenate(tops), np.concatenate(alls)
    rt = rule["round_trip_bp"]
    top_bp, all_bp = float(np.mean(top) * 1e4), float(np.mean(al) * 1e4)
    ok = (top_bp - rt > 0) if dirn == "up" else (top_bp - all_bp <= -rt)
    return {"top_bp": top_bp, "all_bp": all_bp, "pass": bool(ok), "n_top": int(len(top))}


# ---------------------------------------------------------------------------- regime-state amendment


def state_screen_fold(fid: str) -> Path:
    """One 3-class HGB per (H, level) on Y = S_H(t+H), with transition's inputs, fit rows (purge H, hourly) and
    tail; the Markov baseline is fit on the same rows. Writes P(BULL), P(BEAR) and the Markov scores."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    pr = _pr()
    f = next(x for x in _folds() if x["fold_id"] == fid)
    lab = pd.read_parquet(OUT / "labels.parquet")
    btc_cols = ["close", *[c[4:] for c in stand_in_columns(pr) if c.startswith("btc:")]]
    btc = _read("BTC/USDT", btc_cols).set_index("date")
    X = pd.concat([_features(p, pr, btc).assign(pair=p) for p in _pairs()], ignore_index=True)
    t = lab.drop(columns=[c for c in X.columns if c in lab.columns and c not in ("date", "pair")]).merge(X, on=["date", "pair"], how="left")
    feats = [c for c in X.columns if c not in ("date", "pair")]
    ts, te = _ts(f["test_start"]), _ts(f["test_end"])
    tail0 = ts - pd.Timedelta(days=pr["stand_in"]["tail_days"])
    rows = []
    for H in pr["grid"]["H"]:
        purge = H * BAR
        tr = t["tradeable"] & (t["date"] < tail0 - purge)
        tr &= (t["date"].dt.minute == 0) if pr["stand_in"]["hourly_train"] else True
        tail = t["tradeable"] & (t["date"] >= tail0) & (t["date"] < ts - purge)
        test = t["tradeable"] & (t["date"] >= ts) & (t["date"] < te)
        for level in pr["grid"]["level"]:
            col, cur = f"fstate_{level}_{H}", f"state_{level}_{H}"
            m = tr & t[col].notna()
            y = t.loc[m, col].to_numpy(float)
            if len(np.unique(y)) < 3:
                continue
            clf = HistGradientBoostingClassifier(**pr["stand_in"]["hgb"])
            clf.fit(t.loc[m, feats].to_numpy(np.float32), y)
            ix = {c: list(clf.classes_).index(v) for c, v in CLASSES.items()}
            for part, mask in (("tail", tail), ("test", test)):
                g = t.loc[mask, ["date", "pair"]].copy()
                g["cell"] = f"{level}_{H}"
                g["part"] = part
                pp = clf.predict_proba(t.loc[mask, feats].to_numpy(np.float32))
                g["p_bull"], g["p_bear"] = pp[:, ix["BULL"]], pp[:, ix["BEAR"]]
                mk = markov_scores(t.loc[m, cur].to_numpy(), y, t.loc[mask, cur].to_numpy())
                g["markov_bull"], g["markov_bear"] = mk["BULL"].to_numpy(), mk["BEAR"].to_numpy()
                rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"state_pred_{fid}.parquet"
    out.to_parquet(p, index=False)
    print(fid, len(out), "rows", flush=True)
    return p


def state_judge() -> dict:
    """screen_rule of preregistration_regime_state.json per (cell, class); selection per class."""
    pr = _pr()
    rule = json.loads(PREREG_STATE.read_text())["screen_rule"]
    lab = pd.read_parquet(OUT / "labels.parquet")
    preds = pd.concat([pd.read_parquet(OUT / f"state_pred_{f['fold_id']}.parquet").assign(fold=f["fold_id"])
                       for f in _folds()], ignore_index=True)
    test = preds[preds["part"] == "test"].merge(lab, on=["date", "pair"], how="left")
    cells = {}
    for cell, g in test.groupby("cell"):
        level, H = cell.split("_")
        H = int(H)
        for cls, sign, dirn in (("BULL", 1.0, "up"), ("BEAR", -1.0, "down")):
            k = cls.lower()
            r = {"auc": {}, "markov_auc": {}, "trailing_z_auc": {}, "margin": {}, "events": {}, "proxy_bp": {}}
            for half in ("early", "late"):
                h = g[g["fold"].map(lambda x: _half(x, pr)) == half].copy()
                y = (h[f"fstate_{level}_{H}"] == CLASSES[cls]).astype(float).where(h[f"fstate_{level}_{H}"].notna())
                h["y"] = y
                a = _auc(h[f"p_{k}"].to_numpy(float), y.to_numpy(float))
                bm = _auc(h[f"markov_{k}"].to_numpy(float), y.to_numpy(float))
                bz = _auc(sign * h[f"z_{level}_{H}"].to_numpy(float), y.to_numpy(float))
                r["auc"][half], r["markov_auc"][half], r["trailing_z_auc"][half] = a, bm, bz
                base = max(0.5, bm if bm is not None else 0.5, bz if bz is not None else 0.5)
                r["margin"][half] = None if a is None else a - base
                r["events"][half] = int(_events(h, "y"))
                r["proxy_bp"][half] = _proxy(h.assign(p=h[f"p_{k}"]), dirn, H, rule)
            ok = all(r["margin"][x] is not None and r["margin"][x] >= rule["min_auc_margin"] for x in ("early", "late"))
            ok &= all(r["events"][x] >= rule["min_events_per_half"] for x in ("early", "late"))
            ok &= all(r["proxy_bp"][x] is not None and r["proxy_bp"][x]["pass"] for x in ("early", "late"))
            r["pass"] = bool(ok)
            cells[f"{cls}:{cell}"] = r
    sel = {}
    for cls in ("BULL", "BEAR"):
        passing = {c: r for c, r in cells.items() if c.startswith(cls) and r["pass"]}
        sel[cls] = max(passing, key=lambda c: passing[c]["margin"]["late"]) if passing else None
    out = {"cells": cells, "selected": sel, "pass": any(sel.values())}
    (HERE / "screens_regime_state.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def _score_state(h: pd.DataFrame, level: str, H: int, rule: dict, k: str, sign: float, dirn: str) -> dict:
    y = (h[f"fstate_{level}_{H}"] == CLASSES[k.upper()]).astype(float).where(h[f"fstate_{level}_{H}"].notna())
    h = h.assign(y=y)
    a = _auc(h[f"p_{k}"].to_numpy(float), y.to_numpy(float))
    bm = _auc(h[f"markov_{k}"].to_numpy(float), y.to_numpy(float))
    bz = _auc(sign * h[f"z_{level}_{H}"].to_numpy(float), y.to_numpy(float))
    base = max(0.5, bm if bm is not None else 0.5, bz if bz is not None else 0.5)
    return {"auc": a, "markov_auc": bm, "trailing_z_auc": bz, "margin": None if a is None else a - base,
            "events": int(_events(h, "y")), "proxy_bp": _proxy(h.assign(p=h[f"p_{k}"]), dirn, H, rule)}


def state_nn(models: list[str], cell: str) -> dict:
    """The regime NNs' heads on one cell, scored like the stand-in (screen_rule); Markov fit per fold on the
    fold's training rows (dates < test_start - H), the NN's own training range."""
    pr = _pr()
    rule = json.loads(PREREG_STATE.read_text())["screen_rule"]
    level, H = cell.split("_")
    H = int(H)
    lab = pd.read_parquet(OUT / "labels.parquet")
    sig = STRAT / "research" / "nnnc_framework" / "signals"
    out = {"cell": cell, "models": {}}
    for m in models:
        parts = []
        for f in _folds():
            t = pd.read_parquet(sig / f"{m}_{f['fold_id']}.parquet")[["date", "pair", "p_regime_bull", "p_regime_bear"]]
            t["date"] = pd.to_datetime(t["date"], utc=True)
            g = t.merge(lab, on=["date", "pair"], how="inner")
            g = g[g["tradeable"]].rename(columns={"p_regime_bull": "p_bull", "p_regime_bear": "p_bear"})
            fit = lab[lab["tradeable"] & (lab["date"] < _ts(f["test_start"]) - H * BAR)]
            mk = markov_scores(fit[f"state_{level}_{H}"].to_numpy(), fit[f"fstate_{level}_{H}"].to_numpy(),
                               g[f"state_{level}_{H}"].to_numpy())
            g["markov_bull"], g["markov_bear"] = mk["BULL"].to_numpy(), mk["BEAR"].to_numpy()
            parts.append(g.assign(fold=f["fold_id"]))
        g = pd.concat(parts, ignore_index=True)
        res = {}
        for cls, sign, dirn in (("BULL", 1.0, "up"), ("BEAR", -1.0, "down")):
            r = {half: _score_state(g[g["fold"].map(lambda x: _half(x, pr)) == half], level, H, rule, cls.lower(), sign, dirn)
                 for half in ("early", "late")}
            ok = all(r[x]["margin"] is not None and r[x]["margin"] >= rule["min_auc_margin"] for x in r)
            ok &= all(r[x]["events"] >= rule["min_events_per_half"] for x in r)
            ok &= all(r[x]["proxy_bp"] is not None and r[x]["proxy_bp"]["pass"] for x in r)
            res[cls] = {"halves": r, "pass": bool(ok)}
        out["models"][m] = res
    if len(models) == 2:
        sep, shared = models  # (<M>_RG, <M>_RGM)
        out["shared_vs_separate"] = {}
        for cls in ("BULL", "BEAR"):
            d = {h: (out["models"][shared][cls]["halves"][h]["auc"] or np.nan) - (out["models"][sep][cls]["halves"][h]["auc"] or np.nan)
                 for h in ("early", "late")}
            verdict = ("sharing helps" if all(v >= 0.01 for v in d.values()) else
                       "sharing hurts" if all(v <= -0.01 for v in d.values()) else "no difference")
            out["shared_vs_separate"][cls] = {"delta_auc": d, "verdict": verdict}
    path = HERE / "screens_regime_state.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev["nn"] = out
    path.write_text(json.dumps(prev, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------- CLI


def _refusals() -> list[str]:
    import subprocess

    out = []
    for pre in (PREREG, PREREG_STATE):
        rel = pre.relative_to(STRAT)
        if subprocess.run(["git", "-C", str(STRAT), "status", "--porcelain", "--", str(rel)],
                          capture_output=True, text=True).stdout.strip():
            out.append(f"{rel} has uncommitted changes")
        if not subprocess.run(["git", "-C", str(STRAT), "log", "-1", "--format=%ct", "--", str(rel)],
                              capture_output=True, text=True).stdout.strip():
            out.append(f"{rel} is not committed")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("labels", "causality", "screen", "judge", "state-screen", "state-judge", "state-nn"))
    ap.add_argument("--fold")
    ap.add_argument("--models")
    ap.add_argument("--cell")
    a = ap.parse_args(argv)
    bad = _refusals()
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    path = HERE / "screens_transition.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    if a.step == "labels":
        t = build_labels()
        prev["base_rates"] = base_rates(t)
        path.write_text(json.dumps(prev, indent=1))
        print("labels", len(t), "rows", flush=True)
        return 0
    if a.step == "causality":
        r = causality()
        prev["truncation"] = r
        path.write_text(json.dumps(prev, indent=1))
        print(json.dumps(r, indent=1))
        return 0 if r["pass"] else 1
    if a.step in ("screen", "state-screen"):
        if not (prev.get("truncation") or {}).get("pass"):
            print("REFUSED: the truncation test has not passed")
            return 2
        (screen_fold if a.step == "screen" else state_screen_fold)(a.fold)
        return 0
    if a.step == "state-nn":
        r = state_nn(a.models.split(","), a.cell)
        print(json.dumps(r, indent=1, default=str))
        return 0
    if a.step == "state-judge":
        r = state_judge()
        print(json.dumps({"selected": r["selected"], "pass": r["pass"]}, indent=1))
        return 0
    r = judge()
    print(json.dumps({"selected": r["selected"], "pass": r["pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
