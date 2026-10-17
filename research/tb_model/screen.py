"""ATR-barrier study, stage S1 (preregistration.json): does a model trained on 037's tbatr learn
direction where we trade? Per fold (f00-f14), test windows, TRADEABLE bars (037 panel, data_ok AND
volume > 0), label:tbatr_buy (1 = up first, 0 = down first; timeouts / ties excluded):
- AUC of p_buy on gbb candidate bars (the decision population) and on all bars, for each arm and ARO;
- MCC of BUY-argmax vs the label on candidate bars (the honest MCC: no gbb-setup shortcut);
- the 037 composite's candidate AUC per fold (vol/composite.json) as the hand-built reference.
S1 pass per arm: candidate median AUC above ARO's in BOTH halves. Arm B vs A answers whether the
composite's member inputs help the model learn the label.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.screen
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.aro_volume.auc import SIG
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel

HERE = Path(__file__).resolve().parent
ARMS = {"cs_tba": "CS_TBA", "cs_tbb": "CS_TBB"}
REF = "ARO"
TARGET = "label:tbatr_buy"


def tables(panel: Panel, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    p, a = np.full(len(panel), np.nan), np.zeros(len(panel), bool)
    key = pd.DataFrame({"date": panel.date.tz_convert("UTC"), "pair": np.asarray(panel.pairs)[panel.pair],
                        "i": np.arange(len(panel))})
    for f in W.folds():
        t = pd.read_parquet(SIG / f"{prefix}_{f.id}.parquet", columns=["date", "pair", "p_buy", "buy_argmax"])
        t["date"] = pd.to_datetime(t["date"], utc=True)
        m = key.merge(t, on=["date", "pair"], how="inner")
        p[m["i"].to_numpy()] = m["p_buy"].to_numpy(float)
        a[m["i"].to_numpy()] = m["buy_argmax"].to_numpy(bool)
    return p, a


def mcc(pred: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0 or pred.all() or (~pred).all():
        return 0.0
    return float(np.corrcoef(pred.astype(float), y.astype(float))[0, 1])


def main() -> int:
    panel = Panel(tradeable_only=True)
    y = panel.columns([TARGET])[TARGET].astype(float)
    models = {REF: REF, **ARMS}
    sc = {k: tables(panel, v) for k, v in models.items()}
    comp = json.loads((HERE.parent / "indicator_screen" / "vol" / "composite.json").read_text())
    comp = comp["per_window"][TARGET]["candidate"]
    rows = {}
    for f in W.folds():
        base = f.mask(panel.date) & panel.data_ok & np.isfinite(y) & np.all([np.isfinite(v[0]) for v in sc.values()], axis=0)
        r = {"half": f.half, "composite_cand_auc": comp[f.id]["auc"]}
        for pop, m in (("cand", base & panel.candidate), ("all", base)):
            i = np.flatnonzero(m)
            r[f"n_{pop}"] = int(len(i))
            for k, (p, a) in sc.items():
                r[f"{k}_{pop}_auc"] = S.auc(p[i], y[i])
                if pop == "cand":
                    r[f"{k}_cand_mcc"] = mcc(a[i], y[i] > 0.5)
        rows[f.id] = r
    med = lambda key, h: float(np.nanmedian([r[key] for r in rows.values() if r["half"] == h]))
    summary = {h: {k: med(k, h) for k in rows["f00"] if k not in ("half",) and not k.startswith("n_")}
               for h in ("early", "late")}
    s1 = {arm: all(summary[h][f"{arm}_cand_auc"] > summary[h][f"{REF}_cand_auc"] for h in ("early", "late"))
          for arm in ARMS}
    b_vs_a = all(summary[h]["cs_tbb_cand_auc"] > summary[h]["cs_tba_cand_auc"] for h in ("early", "late"))
    out = {"label": "S1 screen (pre-registered); descriptive beyond the pass rule", "summary": summary,
           "s1_pass": s1, "inputs_help_both_halves": b_vs_a, "folds": rows}
    (HERE / "screen.json").write_text(json.dumps(out, indent=1))
    for h in ("early", "late"):
        print(h, {k: round(v, 3) for k, v in summary[h].items()})
    print("S1 pass:", s1, "| B > A both halves:", b_vs_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
