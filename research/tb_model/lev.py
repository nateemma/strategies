"""Backtested leverage control for cs_aro_hp (preregistration_lev.json).

Arm cs_aro_lev = cs_aro (add off) with every stake x STAKE_SCALE 1.3187, cs_aro_hp's committed
capital over cs_aro's. Instrument (no P&L read): on trades shared with cs_aro (pair + open date) the
median first-fill stake ratio is within 0.02 of k; time-weighted average committed stake / cs_aro_hp's
within [0.90, 1.10] per half; 0 adds. Primary: judge_one(cs_aro_hp, cs_aro_lev). Reported:
judge_one(cs_aro_lev, cs_aro), capped share, exposure.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.lev
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_lev.json"
ARM, K = "cs_aro_lev", 1.3187
EXPOSURE = ("cs_aro_lev", "cs_aro_hp", "cs_aro")


def _pooled(tag: str) -> list[dict]:
    from evaluation.pool import pool

    return pool(json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text()))["trades"]


def _first_fills(tag: str) -> dict:
    from research.exit_coupling import feasibility as F

    return {(t["pair"], str(pd.Timestamp(t["open_date"]))): float(t["stake_amount"]) for t in F.archive_trades(tag)
            if sum(bool(o.get("ft_is_entry")) for o in t.get("orders") or []) <= 1}


def instrument(pr: dict) -> dict:
    from research.held_pair import exposure as X

    a, c = _first_fills(ARM), _first_fills("cs_aro")
    ratio = np.array([a[k] / c[k] for k in a.keys() & c.keys() if c[k] > 0])
    pooled = {t: _pooled(t) for t in EXPOSURE}
    exp = {t: {h: X.exposure([x for x in pooled[t] if x["fold_id"] in pr[h]]) for h in ("early", "late")}
           for t in EXPOSURE}
    vs_hp = {h: exp[ARM][h]["avg_stake"] / exp["cs_aro_hp"][h]["avg_stake"] for h in ("early", "late")}
    adds = sum((t.get("n_entries") or 1) - 1 for t in pooled[ARM])
    out = {"shared_trades": int(len(ratio)), "median_stake_ratio": round(float(np.median(ratio)), 4),
           "capped_share": round(float((ratio < K - 0.02).mean()), 4),
           "avg_stake_vs_cs_aro_hp": {h: round(v, 4) for h, v in vs_hp.items()}, "adds": int(adds),
           "exposure": {t: {h: {k: round(v, 1) for k, v in e.items()} for h, e in d.items()} for t, d in exp.items()}}
    out["pass"] = bool(abs(out["median_stake_ratio"] - K) <= 0.02 and all(0.90 <= v <= 1.10 for v in vs_hp.values())
                       and adds == 0)
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
    bad = CJ.refusals({**pr, "arm": ARM}, commit_ts=ts) if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = instrument(pr)
    out = {"label": pr["label"], "instrument": inst}
    if not inst["pass"]:
        out["decision"] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
    else:
        v = judge_one(pr, "cs_aro_hp", ARM)
        small = v["verdict"] == "PASS" and min(v["early_gain_pp"], v["late_gain_pp"]) < pr["small_margin_pp"]
        out["decision"] = {"decision": "INCONCLUSIVE" if small else
                           ("PASS (PROVISIONAL)" if v["verdict"] == "PASS" else v["verdict"]),
                           "reason": "small margin" if small else f"judge_one(cs_aro_hp, cs_aro_lev) {v['verdict']}"}
        out["primary"] = v
        out["lev_vs_cs_aro"] = judge_one(pr, ARM, "cs_aro")
    (HERE / "verdicts_lev.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps({k: v for k, v in inst.items() if k != "exposure"}))
    print("decision", out["decision"])
    for c in ("primary", "lev_vs_cs_aro"):
        v = out.get(c)
        if v:
            print(c, v["arm"], "vs", v["control"], v["verdict"], round(v["arm_pp"], 2), round(v["ctl_pp"], 2),
                  round(v["early_gain_pp"], 2), round(v["late_gain_pp"], 2), v["paired"]["verdict"], v["paired"]["p"],
                  v["risk"]["verdict_sharpe"], v["risk"]["verdict_maxdd"], round(v["risk"]["arm"]["sharpe"], 2),
                  round(v["risk"]["ctl"]["sharpe"], 2), round(v["risk"]["arm"]["maxdd"], 2),
                  round(v["risk"]["ctl"]["maxdd"], 2), v["arm_trades"], v["ctl_trades"], v["failed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
