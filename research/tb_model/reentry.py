"""Held-pair re-entry with the 2xATR(14) take-profit (preregistration_reentry.json).

Arms: w_aro_hp (width book + 031's in-profit add) and w_aro_tp2_hp (+ TP 2xATR14). Instrument per
arm (no P&L read): adds in both halves; every add is on a trade in profit at the add (add fill
above the first fill's price, net of round-trip fees) -- >= 95%; no exit on an add candle; for
w_aro_tp2_hp, trades WITHOUT an add replay under the TP2 exit to >= 90% of recorded exits.
Primary: judge_one(w_aro_tp2_hp, w_aro_hp) -- the take-profit with re-entry on in both. Reported:
judge_one(w_aro_hp, w_aro), judge_one(w_aro_tp2_hp, w_aro), trade / add counts.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.reentry
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model import replay as R  # noqa: E402  (patches feasibility.exit_setup)

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_reentry.json"
ARMS = ("w_aro_hp", "w_aro_tp2_hp")
FEE = 0.001


def _adds(tag: str) -> tuple[list[dict], dict]:
    from research.exit_coupling import feasibility as F

    rows, single = [], {}
    for t in F.archive_trades(tag):
        fills = sorted((o["order_filled_timestamp"], float(o.get("safe_price") or o.get("average") or o.get("price")))
                       for o in t.get("orders") or [] if o.get("ft_is_entry") and o.get("order_filled_timestamp"))
        if len(fills) <= 1:
            single[(t["pair"], str(pd.Timestamp(t["open_date"])))] = True
            continue
        first = fills[0][1]
        close = pd.Timestamp(t["close_date"])
        close = close.tz_localize("UTC") if close.tz is None else close
        for ts, px in fills[1:]:
            add = pd.Timestamp(ts, unit="ms", tz="UTC")
            rows.append({"fold": t["fold"], "pair": t["pair"],
                         "in_profit": px * (1 - FEE) / (first * (1 + FEE)) - 1 > 0,
                         "exit_on_add_candle": bool(close == add), "exit_reason": t["exit_reason"]})
    return rows, single


def instrument(tag: str, pr: dict) -> dict:
    from research.exit_coupling import feasibility as F
    from research.exit_coupling import paths as P

    adds, single = _adds(tag)
    a = pd.DataFrame(adds)
    out = {"adds": {h: int(a["fold"].isin(pr[h]).sum()) if len(a) else 0 for h in ("early", "late")},
           "in_profit_share": round(float(a["in_profit"].mean()), 4) if len(a) else None,
           "exits_on_add_candle": int(a["exit_on_add_candle"].sum()) if len(a) else 0}
    ok = all(out["adds"].values()) and (out["in_profit_share"] or 0) >= 0.95 and out["exits_on_add_candle"] == 0
    if tag == "w_aro_tp2_hp":
        df, _ = F.trade_rows(tag, "NNNC_W_ARO_TP2_HP_f00", "reversion")
        cls, params, roi = R.exit_setup_json("NNNC_W_ARO_TP2_HP_f00")
        params = dataclasses.replace(F.family_params(cls, params, "reversion"), take_profit_atr_mult=None)
        match = []
        for r in df.itertuples():
            b, dates = F.bars_for(r.pair)
            if (r.pair, str(dates[r.e])) not in single:
                continue
            tp = 2.0 * float(R.atr14(r.pair)[r.e - 1])
            x = P.replay(b, r.e, r.open_rate, dataclasses.replace(params, take_profit=tp), roi=roi,
                         roi_suppressed=r.roi_suppressed, end=r.end, tick=r.tick)
            match.append(x.reason == r.rec_reason and x.idx == r.rec_idx)
        out["single_fill_replay_agreement"] = round(float(np.mean(match)), 4) if match else None
        out["single_fill_trades"] = len(match)
        ok = ok and (out["single_fill_replay_agreement"] or 0) >= 0.90
    out["pass"] = bool(ok)
    return out


def main() -> int:
    from research.control_sizer import judge as CJ
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PREREG.read_text())
    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        print(f"REFUSED: {rel} has uncommitted changes")
        return 2
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    bad = [x for a in ARMS for x in CJ.refusals({**pr, "arm": a}, commit_ts=ts)] if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = {a: instrument(a, pr) for a in ARMS}
    out = {"label": pr["label"], "instrument": inst}
    if not all(i["pass"] for i in inst.values()):
        out["decision"] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
    else:
        v = judge_one(pr, "w_aro_tp2_hp", "w_aro_hp")
        small = v["verdict"] == "PASS" and min(v["early_gain_pp"], v["late_gain_pp"]) < pr["small_margin_pp"]
        out["decision"] = {"decision": "INCONCLUSIVE" if small else
                           ("PASS (PROVISIONAL)" if v["verdict"] == "PASS" else v["verdict"]),
                           "reason": "small margin" if small else f"judge_one(w_aro_tp2_hp, w_aro_hp) {v['verdict']}"}
        out["primary"] = v
        out["hp_vs_w_aro"] = judge_one(pr, "w_aro_hp", "w_aro")
        out["tp2_hp_vs_w_aro"] = judge_one(pr, "w_aro_tp2_hp", "w_aro")
    (HERE / "verdicts_reentry.json").write_text(json.dumps(out, indent=1, default=str))
    for a, i in inst.items():
        print(a, "instrument", json.dumps(i))
    print("decision", out["decision"])
    for c in ("primary", "hp_vs_w_aro", "tp2_hp_vs_w_aro"):
        v = out.get(c)
        if v:
            print(c, v["arm"], "vs", v["control"], v["verdict"], round(v["arm_pp"], 2), round(v["ctl_pp"], 2),
                  round(v["early_gain_pp"], 2), round(v["late_gain_pp"], 2), v["paired"]["verdict"], v["paired"]["p"],
                  v["risk"]["verdict_sharpe"], v["risk"]["verdict_maxdd"], v["arm_trades"], v["ctl_trades"], v["failed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
