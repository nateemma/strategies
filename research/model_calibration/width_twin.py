"""Book-level test of the within-candidate finding (2026-10-13; offline, no training, no backtest).

within_candidates.py found that, at matched selectivity, the widest-band rule-eligible bars match
the model's selection on 48-bar forward return (ARO, GBR) and beat it (GFM). Forward return is not
trade P&L (stops, take-profit, one trade per pair), so this rebuilds the BOOKS with the study
simulator (`research.nnnc_framework.entries.Model.simulate`, ~99.7-100% faithful to the recorded
backtests, 24 h grace, no held-pair add) for:

  model     the model's own entries (enter_long)
  width     the SAME non-model gates (volume, rvol, guards as declared), the gbb rule's
            eligibility (guard_metric < 0.3, bb_width > 0.035), and instead of p_buy >= threshold:
            bb_width >= a per-fold cut
  * matched   cut set so the fold's bar count equals the model's (uses the test fold's own count:
              a ranking-quality diagnostic, NOT tradeable)
  * causal    cut = the previous fold's matched cut (f00 uses its own); tradeable

Reported per model and half: pp, trades. PROVISIONAL diagnostic; no adoption rule.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.model_calibration.width_twin
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
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
GUARD, BBW = 0.3, 0.035


def _gates(t, buy) -> np.ndarray:
    """base_mask without the model term (argmax and p_buy)."""
    m = t["volume"].to_numpy() > 0
    m &= t["rvol"].to_numpy() > float(buy["entry_rvol_threshold"])
    if buy["entry_enable_guards"]:
        for k, (col, op, _) in E.GUARDS_ON.items():
            m &= E._cmp(t[col].to_numpy(float), op, float(buy[E.GUARD_PARAM[k]]))
    m &= (t["guard_metric"].to_numpy(float) < GUARD) & (t["bb_width"].to_numpy(float) > BBW)
    return m


def main() -> int:
    out = {"label": "DIAGNOSTIC -- PROVISIONAL (matched variant uses the test fold's own count)",
           "models": {}}
    for m in ("ARO", "GBR", "GFM", "MLX"):
        model = E.Model(m)
        model.params = dataclasses.replace(model.params, stoploss_grace_hours=24.0)
        cuts, prev = {}, None
        books = {"model": {}, "width_matched": {}, "width_causal": {}}
        for f in sorted(model.folds):
            t = model.tables[f]
            buy = E.fold_params(model.prefix, f)
            mm = t["enter_long"].to_numpy() == 1
            g = _gates(t, buy)
            w = t["bb_width"].to_numpy(float)
            k = int(mm.sum())
            cand = np.sort(w[g])[::-1]
            cut = cand[k - 1] if 0 < k <= len(cand) else np.inf
            cuts[f] = float(cut)
            causal = cut if prev is None else prev
            prev = cut
            books["model"][f] = model.simulate(f, mm)
            books["width_matched"][f] = model.simulate(f, g & (w >= cut))
            books["width_causal"][f] = model.simulate(f, g & (w >= causal))
        res = {}
        for name, b in books.items():
            res[name] = {h: {"pp": round(float(sum(b[f]["pp"].sum() for f in b
                                                  if F.half_of(f) == h and not b[f].empty)), 3),
                             "trades": int(sum(len(b[f]) for f in b if F.half_of(f) == h))}
                         for h in ("early", "late")}
        res["cuts"] = cuts
        out["models"][m] = res
        print(m, json.dumps({k: v for k, v in res.items() if k != "cuts"}), flush=True)
    (HERE / "width_twin.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
