"""ARO volume-filter retrain: does the retrained p_buy rank tradeable gbb candidates better? Descriptive.

Per fold test window, rows = gbb candidate bars with data_ok and volume > 0 (037 Panel, tradeable_only),
target = label:buy (ARO's own label): AUC of the retrained p_buy (signals/AROV_fNN) vs the control's
(signals/ARO_fNN, = cs_aro's weights). Medians per half.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.aro_volume.auc
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel

HERE = Path(__file__).resolve().parent
SIG = HERE.parent / "nnnc_framework" / "signals"


def p_for(panel: Panel, prefix: str) -> np.ndarray:
    p = np.full(len(panel), np.nan)
    key = pd.DataFrame({"date": panel.date.tz_convert("UTC"), "pair": np.asarray(panel.pairs)[panel.pair],
                        "i": np.arange(len(panel))})
    for f in W.folds():
        t = pd.read_parquet(SIG / f"{prefix}_{f.id}.parquet", columns=["date", "pair", "p_buy"])
        t["date"] = pd.to_datetime(t["date"], utc=True)
        m = key.merge(t, on=["date", "pair"], how="inner")
        p[m["i"].to_numpy()] = m["p_buy"].to_numpy(float)
    return p


def main() -> int:
    panel = Panel(tradeable_only=True)
    y = panel.core["label:buy"]
    ps = {"ARO": p_for(panel, "ARO"), "AROV": p_for(panel, "AROV")}
    rows = {}
    for f in W.folds():
        i = np.flatnonzero(f.mask(panel.date) & panel.data_ok & panel.candidate
                           & np.isfinite(ps["ARO"]) & np.isfinite(ps["AROV"]))
        rows[f.id] = {"half": f.half, "n": int(len(i)), **{k: S.auc(v[i], y[i]) for k, v in ps.items()}}
    med = {h: {k: float(np.nanmedian([r[k] for r in rows.values() if r["half"] == h])) for k in ps}
           for h in ("early", "late")}
    out = {"label": "descriptive; candidate tradeable bars, label:buy", "median": med, "folds": rows}
    (HERE / "auc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(med))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
