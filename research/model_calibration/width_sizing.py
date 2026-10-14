"""Option 2 of the model check: the model as a SIZER on the width book (2026-10-13; simulator screen,
no training, no backtest).

The width book (width_twin `width_causal`: the gbb rule + bb_width >= the previous fold's cut) picks
the trades; the model only sets the stake. Each width-book trade keeps its simulated exit and is
re-weighted by the model at its signal bar (entry bar - 1):

  agree_0.5   1.0 if the model's own entry fires (enter_long), else 0.5
  agree_0.25  1.0 if enter_long, else 0.25
  rank        0.5 + 0.5 * the trade's p_buy percentile among the fold's width-book trades

Re-weighting does not move entries or exits (the simulator has no wallet shrinkage), so this prices
sizing alone. Reported per model and half: agree / disagree trades and pp per trade; each sized
book's pp, capital (sum of weight, as a share of the unsized book) and pp per unit capital.

READING (declared before the run): the model is a useful sizer if agree pp/trade > disagree pp/trade
in BOTH halves on >= 3 of 4 models, AND agree_0.5 earns more pp per unit capital than the unsized
width book in both halves on >= 3 of 4. EXPLORATORY: width_plus_model.py already showed the
width AND model subset's pp (the agree side), so the split is not unseen.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.model_calibration.width_sizing
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.model_calibration.width_twin import _gates  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
SCHEMES = ("agree_0.5", "agree_0.25", "rank")


def _signal_rows(t: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """The signal-table row (p_buy, enter_long) behind each simulated trade."""
    dt = pd.DatetimeIndex(t["date"])
    key = pd.DataFrame({"pair": t["pair"].to_numpy(),
                        "date": dt.tz_localize("UTC") if dt.tz is None else dt,
                        "p_buy": t["p_buy"].to_numpy(float),
                        "agree": t["enter_long"].to_numpy() == 1})
    sd = [pd.Timestamp(F.bars_for(r.pair)[1][r.e - 1]) for r in d.itertuples()]
    q = pd.DataFrame({"pair": d["pair"].to_numpy(), "date": pd.DatetimeIndex(sd).tz_convert("UTC")})
    out = q.merge(key, on=["pair", "date"], how="left")
    assert out["p_buy"].notna().all(), "a simulated trade has no signal row"
    return out


def main() -> int:
    tw = json.loads((HERE / "width_twin.json").read_text())["models"]
    out = {"label": "EXPLORATORY screen (agree side seen in width_plus_model.py)", "models": {}}
    for m in ("ARO", "GBR", "GFM", "MLX"):
        model = E.Model(m)
        model.params = dataclasses.replace(model.params, stoploss_grace_hours=24.0)
        folds, cuts = sorted(model.folds), tw[m]["cuts"]
        rows = []
        for i, f in enumerate(folds):
            t = model.tables[f]
            g = _gates(t, E.fold_params(model.prefix, f))
            cut = cuts[folds[i - 1]] if i else cuts[f]
            d = model.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
            if d.empty:
                continue
            s = _signal_rows(t, d)
            d = d.assign(p_buy=s["p_buy"].to_numpy(), agree=s["agree"].to_numpy(), half=F.half_of(f))
            d["w_agree_0.5"] = np.where(d["agree"], 1.0, 0.5)
            d["w_agree_0.25"] = np.where(d["agree"], 1.0, 0.25)
            d["w_rank"] = 0.5 + 0.5 * d["p_buy"].rank(pct=True)
            rows.append(d)
        a = pd.concat(rows, ignore_index=True)
        res = {}
        for h, d in a.groupby("half"):
            ag, dis = d[d["agree"]], d[~d["agree"]]
            base = float(d["pp"].sum())
            r = {"trades": len(d), "unsized_pp": round(base, 3),
                 "agree": {"n": len(ag), "pp_per_trade": round(float(ag["pp"].mean()), 4) if len(ag) else None},
                 "disagree": {"n": len(dis),
                              "pp_per_trade": round(float(dis["pp"].mean()), 4) if len(dis) else None}}
            for sch in SCHEMES:
                w = d[f"w_{sch}"].to_numpy()
                pp, cap = float((d["pp"] * w).sum()), float(w.mean())
                r[sch] = {"pp": round(pp, 3), "capital": round(cap, 4),
                          "pp_per_capital": round(pp / cap, 3), "beats_unsized_per_capital": pp / cap > base}
            res[h] = r
        out["models"][m] = res
        print(m, json.dumps(res), flush=True)
    M = out["models"]

    def _both(fn):
        return {m: all(fn(M[m][h]) for h in ("early", "late")) for m in M}

    split = _both(lambda r: (r["disagree"]["pp_per_trade"] is not None
                             and r["agree"]["pp_per_trade"] > r["disagree"]["pp_per_trade"]))
    cap = _both(lambda r: r["agree_0.5"]["beats_unsized_per_capital"])
    out["verdict"] = {"agree_beats_disagree": split, "agree_0.5_beats_unsized_per_capital": cap,
                      "useful_sizer": sum(split.values()) >= 3 and sum(cap.values()) >= 3}
    print("VERDICT", json.dumps(out["verdict"]))
    (HERE / "width_sizing.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
