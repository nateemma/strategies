"""Option 3 of the model check: width-book variants (2026-10-13; simulator screen, no training).

The width book (width_twin `width_causal`) still borrows one thing from the model: its cut is set
so the fold's bar count matches the MODEL's. Two families, both causal (set from the previous fold;
f00 uses its own):

  quantile cut  q50 / q70 / q80 / q90: cut = that quantile of bb_width over the previous fold's
                rule-eligible bars (no model input at all)
  second feature  the width book AND feature >= the previous fold's median of that feature over
                the width book's signal bars (about halves the book): rvol, adx, atr_pct_roll,
                depth (-guard_metric), low close_norm (-close_norm)

READING (declared before the run): in each family the variant with the largest EARLY-half pp summed
over the four models is the candidate; it replaces the width book as the baseline if its LATE-half
pp beats the width book's on >= 3 of 4 models. Early half selects; late half (seen for the width
book itself, not for any variant) checks.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.model_calibration.width_variants
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.model_calibration.width_twin import _gates  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
QUANTILES = {"q50": 0.5, "q70": 0.7, "q80": 0.8, "q90": 0.9}
FEATURES = {"rvol": ("rvol", 1), "adx": ("adx", 1), "atr_pct": ("atr_pct_roll", 1),
            "depth": ("guard_metric", -1), "low_close_norm": ("close_norm", -1)}
FAMILIES = {"quantile": tuple(QUANTILES), "second_feature": tuple(f"w+{k}" for k in FEATURES)}


def _feat(t, name):
    col, sign = FEATURES[name]
    return sign * t[col].to_numpy(float)


def main() -> int:
    tw = json.loads((HERE / "width_twin.json").read_text())["models"]
    names = ("width",) + FAMILIES["quantile"] + FAMILIES["second_feature"]
    res = {}
    for m in ("ARO", "GBR", "GFM", "MLX"):
        model = E.Model(m)
        model.params = dataclasses.replace(model.params, stoploss_grace_hours=24.0)
        folds, cuts = sorted(model.folds), tw[m]["cuts"]
        acc = {n: {"early": [0.0, 0], "late": [0.0, 0]} for n in names}
        prev = None  # previous fold's (eligible bb_width, width-book feature medians)
        for i, f in enumerate(folds):
            t = model.tables[f]
            g = _gates(t, E.fold_params(model.prefix, f))
            w = t["bb_width"].to_numpy(float)
            width = g & (w >= (cuts[folds[i - 1]] if i else cuts[f]))
            here = (w[g], {k: float(np.median(_feat(t, k)[width])) for k in FEATURES})
            src = prev or here
            masks = {"width": width}
            masks |= {n: g & (w >= float(np.quantile(src[0], q))) for n, q in QUANTILES.items()}
            masks |= {f"w+{k}": width & (_feat(t, k) >= src[1][k]) for k in FEATURES}
            prev = here
            h = F.half_of(f)
            for n, mk in masks.items():
                d = model.simulate(f, mk)
                acc[n][h][0] += float(d["pp"].sum()) if not d.empty else 0.0
                acc[n][h][1] += len(d)
        res[m] = {n: {h: {"pp": round(v[0], 3), "trades": v[1]} for h, v in a.items()} for n, a in acc.items()}
        print(m, json.dumps(res[m]), flush=True)
    verdict = {}
    for fam, members in FAMILIES.items():
        cand = max(members, key=lambda n: sum(res[m][n]["early"]["pp"] for m in res))
        beats = {m: res[m][cand]["late"]["pp"] > res[m]["width"]["late"]["pp"] for m in res}
        verdict[fam] = {"candidate": cand, "late_beats_width": beats,
                        "replaces_width_book": sum(beats.values()) >= 3}
    print("VERDICT", json.dumps(verdict))
    (HERE / "width_variants.json").write_text(json.dumps(
        {"label": "EXPLORATORY screen (early selects, late checks)", "models": res, "verdict": verdict},
        indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
