"""Existing triple-barrier model (034 `LS_TB`: ARO retrained on triple_barrier, pt 2% / sl 5%, folds
f00-f07) vs ARO (`ARO`, the control weights): test-window AUC on TRADEABLE bars (037 panel,
data_ok AND volume > 0), all bars and gbb candidate bars, on ARO's label (label:buy), the fixed-%
barrier (label:tb_buy) and the ATR-scaled one (label:tbatr_buy). Medians over f00-f07. Descriptive.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.aro_volume.auc import p_for_folds
from research.indicator_screen import scores as S
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel

HERE = Path(__file__).resolve().parent
MODELS = ("ARO", "LS_TB")
LABELS = ("label:buy", "label:tb_buy", "label:tbatr_buy")


def main() -> int:
    panel = Panel(tradeable_only=True)
    folds = [f for f in W.folds() if f.half == "early"]
    ps = {m: p_for_folds(panel, m, folds) for m in MODELS}
    ys = {t: panel.columns([t])[t].astype(float) for t in LABELS}
    out = {}
    for pop in ("all", "candidate"):
        for t, y in ys.items():
            per = []
            for f in folds:
                m = f.mask(panel.date) & panel.data_ok & np.isfinite(y) & np.isfinite(ps["ARO"]) & np.isfinite(ps["LS_TB"])
                i = np.flatnonzero(m & panel.candidate if pop == "candidate" else m)
                per.append({k: S.auc(v[i], y[i]) for k, v in ps.items()})
            out[f"{t} [{pop}]"] = {k: float(np.nanmedian([r[k] for r in per])) for k in MODELS}
            print(f"{t:18s} {pop:9s}", {k: round(v, 3) for k, v in out[f"{t} [{pop}]"].items()}, flush=True)
    (HERE / "check.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
