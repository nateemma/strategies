"""040 auxiliary-head skill screen (research R10 screen 1; preregistration_screens.json).

For each auxiliary head of the heads-trained model <M>_ah, on its test windows: the one-vs-rest AUC of the
head's probability for its declared class against the head's OWN training label, compared with a CAUSAL
baseline -- the same quantity smoothed over a TRAILING window ending at t ("the head only recovers the
current state"). Labels come from NNMT's own target methods (get_market_regime / get_risk_level /
get_flow / get_momentum, then the 5-bar forward shift; get_profit_target), run on each pair's cached
indicators, so the screen scores exactly what the head was trained to predict.

  margin = AUC(head) - AUC(baseline), per half; skill passes if margin >= min_auc_margin in both halves.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.head_screen
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNMT")]

HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"
NEED = ["close", "high", "low", "volume", "base:atr_norm", "base:di_diff_scaled", "base:aroonosc_scaled"]


def _strategy():
    from NNMT.NNMT_MLX import NNMT_MLX

    s = NNMT_MLX.__new__(NNMT_MLX)
    s.include_list = list(NNMT_MLX.include_list)
    s.debug_print = lambda *a, **k: None
    return s


def head_labels(df: pd.DataFrame, s) -> pd.DataFrame:
    """The training labels of the five auxiliary heads (class ints), via NNMT's own methods."""
    d = df.rename(columns={c: c[5:] for c in df.columns if c.startswith("base:")}).reset_index(drop=True)
    d["regime"] = s.get_market_regime(d)
    d["risk"] = s.get_risk_level(d)
    d["flow"] = s.get_flow(d)
    d["momentum"] = s.get_momentum(d)
    return pd.DataFrame({
        "regime": s.get_market_target(d), "risk": s.get_risk_target(d), "flow": s.get_flow_target(d),
        "momentum": s.get_momentum_target(d), "profit": s.get_profit_target(d)})


def baselines(df: pd.DataFrame, s, pr: dict) -> pd.DataFrame:
    """Causal trailing analogues (continuous scores, higher = the declared class)."""
    w = pr["baseline_windows"]
    close = df["close"].astype(float).reset_index(drop=True)
    tr = lambda x, n: pd.Series(x).reset_index(drop=True).rolling(n, min_periods=1).mean()  # noqa: E731
    return pd.DataFrame({
        "regime": close / tr(close, w["regime"]) - 1.0,
        "risk": tr(df["base:atr_norm"].astype(float), w["risk"]),
        "flow": tr(df["base:di_diff_scaled"].astype(float), w["flow"]),
        "momentum": tr(df["base:aroonosc_scaled"].astype(float), w["momentum"]),
        "profit": close / close.shift(w["profit"]) - 1.0,
    })


def _auc(score, y) -> float | None:
    from sklearn.metrics import roc_auc_score

    score, y = np.asarray(score, float), np.asarray(y, float)
    ok = np.isfinite(score) & np.isfinite(y)
    yy = y[ok] > 0.5
    if yy.all() or not yy.any():
        return None
    return float(roc_auc_score(yy, score[ok]))


def main(argv=None) -> int:
    from research.indicator_screen import harvest as H
    from research.nnmt import judge as J

    path, pr = J._prereg("screens")
    bad = J._refused(path, {**pr, "folds": [], "arms": {}})
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    sel = json.loads((HERE / "verdicts_models.json").read_text())["selected"]
    model = f"{sel}_ah".upper()
    s = _strategy()
    folds = json.loads((STRAT / "evaluation" / "folds" / f"plan_{sel}_ah.json").read_text())["folds"]
    tables = pd.concat([pd.read_parquet(SIG / f"{model}_{f['fold_id']}.parquet").assign(fold=f["fold_id"])
                        for f in folds], ignore_index=True)
    tables["date"] = pd.to_datetime(tables["date"], utc=True)
    rows = []
    for pair, g in tables.groupby("pair"):
        c = H.read_cache(pair, NEED)
        c["date"] = pd.to_datetime(c["date"], utc=True)
        lab, base = head_labels(c, s), baselines(c, s, pr)
        lab.columns = [f"lab_{x}" for x in lab.columns]
        base.columns = [f"base_{x}" for x in base.columns]
        full = pd.concat([c[["date", "volume"]].reset_index(drop=True), lab, base], axis=1)
        rows.append(g.merge(full, on="date", how="left", suffixes=("", "_c")))
    t = pd.concat(rows, ignore_index=True)
    t = t[t["volume"] > 0]
    out = {}
    for head, spec in pr["heads"].items():
        cls = spec["class_index"]
        r = {"column": spec["column"], "class_index": cls, "auc": {}, "baseline_auc": {}, "margin": {}}
        for half in ("early", "late"):
            h = t[t["fold"].isin(pr[half])]
            y = (h[f"lab_{head}"] == cls).astype(float)
            a = _auc(h[spec["column"]], y)
            # the baseline's sign follows the class: a high trailing value predicts the "up" class
            b = _auc(h[f"base_{head}"] * (1.0 if cls == 2 else -1.0), y)
            r["auc"][half], r["baseline_auc"][half] = a, b
            r["margin"][half] = None if a is None or b is None else a - b
        r["skill_pass"] = all(r["margin"][x] is not None and r["margin"][x] >= pr["skill_rule"]["min_auc_margin"]
                              for x in ("early", "late"))
        out[head] = r
    p = HERE / "screens_heads.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev["skill"] = out
    rel = prev.get("relevance") or {}
    prev["pass"] = {h: bool(out[h]["skill_pass"] and (rel.get(h) or {}).get("pass")) for h in out}
    p.write_text(json.dumps(prev, indent=1, default=str))
    print(json.dumps({h: {"auc": r["auc"], "baseline": r["baseline_auc"], "skill": r["skill_pass"]}
                      for h, r in out.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
