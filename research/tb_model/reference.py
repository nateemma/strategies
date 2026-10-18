"""ATR-barrier study, DIAGNOSTIC (not a gate; after S1): what is learnable from arm A/B's inputs?

Per fold (f00-f14), simple models on the LAST bar's inputs (no sequence) -- ARO's 24 include_list
columns (037 cache `base:*`, pre-scaler) for A, plus the 7 composite members for B:
- logistic regression (standardised on the training rows) and HistGradientBoosting;
- training rows: tradeable (data_ok AND volume > 0), hourly, from the fold's train_start to 48 bars
  before its test start; test rows: the fold's test window, tradeable.
- targets: `dir` = label:tbatr_buy (1 up first, 0 down first; timeouts excluded), the label S1 scores;
  `hit` = barrier hit vs timeout (what the 3-class model's p_buy actually tracked).
Reports test AUC on candidate and all bars, next to the composite (0.554/0.552) and the trained arms.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.reference
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.cmp_sizer.signals import members
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel

HERE = Path(__file__).resolve().parent
TARGET = "label:tbatr_buy"
ARO24 = ["adx_scaled", "aroonosc_scaled", "atr_norm", "bb_position", "bb_width", "cci_scaled",
         "di_diff_scaled", "ema_fast_norm", "fast_diff", "fastk_scaled", "fisher_ss", "cg_ss",
         "gain_norm", "guard_metric_pos", "guard_metric_neg", "macd_pos", "macd_neg", "macdhist_norm",
         "mfi_scaled", "rsi_scaled", "sar_ratio", "spread_ma", "vwap_pos", "vwap_neg"]


def fit_score(kind: str, X: np.ndarray, y: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        mu, sd = X.mean(0), X.std(0) + 1e-9
        return LogisticRegression(max_iter=1000).fit((X - mu) / sd, y).predict_proba((Xt - mu) / sd)[:, 1]
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y).predict_proba(Xt)[:, 1]


def main() -> int:
    panel = Panel(tradeable_only=True)
    mem, _ = members()
    sets = {"A": ["base:" + c for c in ARO24]}
    sets["B"] = sets["A"] + list(mem)
    d = panel.columns(sorted(set(sets["B"])) + [TARGET])
    y_dir = d[TARGET].astype(float)
    targets = {"dir": y_dir, "hit": np.isfinite(y_dir).astype(float)}
    hourly = np.asarray(panel.date.minute == 0)
    ok = panel.data_ok & np.all([np.isfinite(d[c]) for c in sets["B"]], axis=0)
    rows = {}
    for f in W.folds():
        tr_win = np.asarray((panel.date >= f.train_start) & (panel.date < f.start - W.purge(TARGET) * W.BAR))
        te_win = f.mask(panel.date) & ok
        r = {"half": f.half}
        for t, y in targets.items():
            fin = np.isfinite(y)
            tr = np.flatnonzero(ok & hourly & tr_win & fin)
            te = np.flatnonzero(te_win & fin)
            cand = panel.candidate[te]
            for s, cols in sets.items():
                X = np.column_stack([d[c] for c in cols])
                for kind in ("logit", "hgb"):
                    p = fit_score(kind, X[tr], y[tr] > 0.5, X[te])
                    r[f"{t}_{s}_{kind}_cand"] = S.auc(p[cand], y[te][cand])
                    r[f"{t}_{s}_{kind}_all"] = S.auc(p, y[te])
            r[f"n_train_{t}"] = int(len(tr))
        rows[f.id] = r
        print(f.id, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    keys = [k for k in rows["f00"] if k != "half" and not k.startswith("n_")]
    summary = {h: {k: float(np.nanmedian([r[k] for r in rows.values() if r["half"] == h])) for k in keys}
               for h in ("early", "late")}
    out = {"label": "DIAGNOSTIC after S1 (not a gate)", "inputs": sets, "summary": summary, "folds": rows}
    (HERE / "reference.json").write_text(json.dumps(out, indent=1))
    for h in ("early", "late"):
        print(h, {k: round(v, 3) for k, v in summary[h].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
