"""031 follow-up screen (2026-10-13): conditioned and risk-matched held-pair adds.

The judged DCA arms (research/held_pair/report.md) showed the add mostly scales exposure: Sharpe
UNCHANGED everywhere, drawdown up 1.3-3.3x, and the late half / 2022 lost. Two fix-oriented
variants, priced on the NNNC signal tables with `screen.simulate` (24 h grace, last fill >= 1 h):

  condition  all | loss (held trade under water) | profit (held trade in profit: pyramiding)
  add size   1.0 | 0.5 | 0.25 x the fresh-entry stake (then the liquidity and wallet caps)

Per variant and model:
  * gain vs the no-add book per half (EARLY = selection; LATE = EXPLORATORY: the judged DCA arms'
    late half has been seen, so no variant here is confirmed by it);
  * max drawdown and Sharpe of the daily pp series (15 folds), arm vs base;
  * leverage control: the adds' stake-hours x the base book's pp per stake-hour;
  * risk-matched gain: arm pp - base pp x (arm maxDD / base maxDD), i.e. the gain over simply
    levering the base book up to the arm's drawdown. > 0 = the add earns more than leverage at
    equal drawdown.

Calibration: DCA:all:1.0 here must reproduce the judged arms (early ~ fidelity run; late ARO
-0.97, GBR -7.74, GFM +1.75, MLX +5.39 within the screen's usual few-pp tolerance).

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.screen2
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

from evaluation.risk import max_drawdown, sharpe  # noqa: E402
from research.exit_coupling import feasibility as F  # noqa: E402
from research.held_pair.screen import simulate  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
CONDS = ("all", "loss", "profit")
FRACS = (1.0, 0.5, 0.25)
GRACE = 24.0
JUDGED_LATE = {"ARO": -0.97, "GBR": -7.74, "GFM": 1.75, "MLX": 5.39}
HOURS_PER_BAR = 0.25


def _book(model, sims: dict) -> pd.DataFrame:
    rows = []
    for f, df in sims.items():
        if df.empty:
            continue
        for t in df.itertuples():
            _, dates = F.bars_for(t.pair)
            add_e = getattr(t, "add_e", -1)
            add_stake = getattr(t, "add_stake", 0.0)
            first = t.stake - add_stake
            sh = first * (t.x - t.e) * HOURS_PER_BAR
            extra = add_stake * (t.x - add_e) * HOURS_PER_BAR if add_e >= 0 else 0.0
            rows.append({"fold": f, "half": F.half_of(f), "close_day": dates[t.x].floor("D"),
                         "pp": t.pp, "stake_hours": sh + extra, "extra_stake_hours": extra,
                         "adds": int(add_e >= 0)})
    return pd.DataFrame(rows)


def _risk(book: pd.DataFrame, span) -> dict:
    s = book.groupby("close_day")["pp"].sum().reindex(span, fill_value=0.0).to_numpy()
    return {"maxdd": max_drawdown(s), "sharpe": sharpe(s)}


def main() -> int:
    out = {"label": "EXPLORATORY -- early half selects; late half already seen by the judged DCA "
                    "arms; no fresh NNNC window remains for the held-pair question",
           "grace": GRACE, "models": {}}
    for m in ("ARO", "GBR", "GFM", "MLX"):
        model = E.Model(m)
        params = dataclasses.replace(model.params, stoploss_grace_hours=GRACE)
        masks = {f: model.tables[f]["enter_long"].to_numpy() == 1 for f in model.folds}
        base = _book(model, {f: simulate(model, f, masks[f], params, "NONE", None, 0.0)
                             for f in model.folds})
        days = pd.to_datetime(base["close_day"])
        span = pd.date_range(days.min(), days.max() + pd.Timedelta(days=30), freq="D")
        rb = _risk(base, span)
        rate = base["pp"].sum() / base["stake_hours"].sum()
        res = {"base": {"early_pp": base.loc[base.half == "early", "pp"].sum(),
                        "late_pp": base.loc[base.half == "late", "pp"].sum(), **rb},
               "variants": {}}
        for cond in CONDS:
            for frac in FRACS:
                arm = _book(model, {f: simulate(model, f, masks[f], params, "DCA", cond, 1.0,
                                                add_frac=frac) for f in model.folds})
                ra = _risk(arm, span)
                gain = {h: arm.loc[arm.half == h, "pp"].sum() - base.loc[base.half == h, "pp"].sum()
                        for h in ("early", "late")}
                lev = arm["extra_stake_hours"].sum() * rate
                dd_ratio = ra["maxdd"] / rb["maxdd"] if rb["maxdd"] > 0 else float("nan")
                risk_matched = arm["pp"].sum() - base["pp"].sum() * dd_ratio
                res["variants"][f"{cond}:{frac:g}"] = {
                    "early_pp": round(gain["early"], 3), "late_pp": round(gain["late"], 3),
                    "adds": int(arm["adds"].sum()), "maxdd": round(ra["maxdd"], 3),
                    "dd_ratio": round(dd_ratio, 3), "sharpe": round(ra["sharpe"], 3),
                    "leverage_pp": round(lev, 3),
                    "beats_leverage": bool(gain["early"] + gain["late"] > lev),
                    "risk_matched_pp": round(risk_matched, 3)}
            print(m, cond, json.dumps({k: v for k, v in res["variants"].items()
                                       if k.startswith(cond)}), flush=True)
        res["calibration"] = {"screen_all_1_late_pp": res["variants"]["all:1"]["late_pp"],
                              "judged_arm_late_pp": JUDGED_LATE[m]}
        out["models"][m] = res
        print(m, "base", json.dumps({k: round(v, 3) for k, v in res["base"].items()}),
              "calibration", res["calibration"], flush=True)
    (HERE / "screen2.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
