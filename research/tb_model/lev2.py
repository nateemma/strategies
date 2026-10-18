"""Two controls for cs_aro_hp's in-profit add (preregistration_lev2.json).

cs_aro_lev2 = cs_aro with stakes x1.7432 (capital-matched); cs_aro_hpany = cs_aro + the any-signal add.
Instruments per arm (no P&L read for a failed arm); primaries judge_one(cs_aro_hp, cs_aro_lev2) and
judge_one(cs_aro_hp, cs_aro_hpany), two separate decisions. Reported: each arm vs cs_aro, exposure.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.lev2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.lev import _pooled  # noqa: E402
from research.tb_model.reentry import _adds  # noqa: E402
from research.tb_model.stack import _agree  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_lev2.json"
K, BASE_STAKE, HEADROOM = 1.7432, 900.0, 10.0
WEIGHTS = {"agree": 1.0 / 0.75, "disagree": 0.5 / 0.75}
ARMS = ("cs_aro_lev2", "cs_aro_hpany")
EXPOSURE = ("cs_aro_lev2", "cs_aro_hpany", "cs_aro_hp", "cs_aro")
_QV: dict = {}


def _cap(pair: str, ts: pd.Timestamp) -> float:
    """Fillable stake of the candle before the entry open (BaseStrategy._fillable_stake)."""
    from research.clfstudy.data import load_pair

    if pair not in _QV:
        d = load_pair(pair)
        _QV[pair] = pd.Series((d["volume"] * d["close"]).to_numpy(float) / HEADROOM, index=pd.DatetimeIndex(d["date"]))
    s = _QV[pair]
    return float(s.iloc[s.index.searchsorted(ts) - 1])


def _exposure(pr: dict) -> dict:
    from research.held_pair import exposure as X

    pooled = {t: _pooled(t) for t in EXPOSURE}
    return {t: {h: {k: round(v, 1) for k, v in X.exposure([x for x in pooled[t] if x["fold_id"] in pr[h]]).items()}
                for h in ("early", "late")} for t in EXPOSURE}


def inst_lev2(pr: dict, exp: dict) -> dict:
    from research.exit_coupling import feasibility as F

    shared = {(t["pair"], str(pd.Timestamp(t["open_date"]))) for t in F.archive_trades("cs_aro")}
    ok = []
    for t in F.archive_trades("cs_aro_lev2"):
        if (t["pair"], str(pd.Timestamp(t["open_date"]))) not in shared:
            continue
        intended = BASE_STAKE * WEIGHTS[str(t["enter_tag"]).rsplit(":", 1)[-1]]
        model = min(K * intended, _cap(t["pair"], pd.Timestamp(t["open_date"])))
        ok.append(abs(float(t["stake_amount"]) / model - 1) <= 0.01)
    vs_hp = {h: round(exp["cs_aro_lev2"][h]["avg_stake"] / exp["cs_aro_hp"][h]["avg_stake"], 4) for h in ("early", "late")}
    adds = sum((t.get("n_entries") or 1) - 1 for t in _pooled("cs_aro_lev2"))
    out = {"avg_stake_vs_cs_aro_hp": vs_hp, "adds": int(adds), "shared_trades": len(ok),
           "stake_model_share": round(float(np.mean(ok)), 4) if ok else None}
    out["pass"] = bool(all(0.90 <= v <= 1.10 for v in vs_hp.values()) and adds == 0
                       and (out["stake_model_share"] or 0) >= 0.95)
    return out


def inst_hpany(pr: dict) -> dict:
    a = pd.DataFrame(_adds("cs_aro_hpany")[0])
    _, share = _agree("cs_aro_hpany")
    _, ctl = _agree("cs_aro")
    out = {"adds": {h: int(a["fold"].isin(pr[h]).sum()) if len(a) else 0 for h in ("early", "late")},
           "adds_not_in_profit": int((~a["in_profit"]).sum()) if len(a) else 0,
           "in_profit_share": round(float(a["in_profit"].mean()), 4) if len(a) else None,
           "exits_on_add_candle": int(a["exit_on_add_candle"].sum()) if len(a) else 0,
           "agree_share": round(share, 4), "cs_aro_agree_share": round(ctl, 4)}
    out["pass"] = bool(all(out["adds"].values()) and out["adds_not_in_profit"] > 0
                       and out["exits_on_add_candle"] == 0 and abs(share - ctl) <= 0.05)
    return out


def decide(pr: dict, v: dict) -> dict:
    small = v["verdict"] == "PASS" and min(v["early_gain_pp"], v["late_gain_pp"]) < pr["small_margin_pp"]
    return {"decision": "INCONCLUSIVE" if small else ("PASS (PROVISIONAL)" if v["verdict"] == "PASS" else v["verdict"]),
            "reason": "small margin" if small else f"judge_one({v['arm']}, {v['control']}) {v['verdict']}"}


def _line(v: dict) -> str:
    r = v["risk"]
    return (f"{v['arm']} vs {v['control']} {v['verdict']} {v['arm_pp']:.2f} {v['ctl_pp']:.2f} "
            f"{v['early_gain_pp']:+.2f} / {v['late_gain_pp']:+.2f} paired {v['paired']['verdict']} p {v['paired']['p']} "
            f"sharpe {r['arm']['sharpe']:.2f}/{r['ctl']['sharpe']:.2f} {r['verdict_sharpe']} "
            f"maxdd {r['arm']['maxdd']:.2f}/{r['ctl']['maxdd']:.2f} {r['verdict_maxdd']} "
            f"trades {v['arm_trades']}/{v['ctl_trades']} {v['failed']}")


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
    exp = _exposure(pr)
    inst = {"cs_aro_lev2": inst_lev2(pr, exp), "cs_aro_hpany": inst_hpany(pr)}
    out = {"label": pr["label"], "instrument": inst, "exposure": exp, "decisions": {}}
    for arm, q in (("cs_aro_lev2", "lev2"), ("cs_aro_hpany", "hpany")):
        if not inst[arm]["pass"]:
            out["decisions"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        v = judge_one(pr, "cs_aro_hp", arm)
        out["decisions"][q] = decide(pr, v)
        out[f"primary_{q}"] = v
        out[f"{q}_vs_cs_aro"] = judge_one(pr, arm, "cs_aro")
    (HERE / "verdicts_lev2.json").write_text(json.dumps(out, indent=1, default=str))
    for a, i in inst.items():
        print(a, "instrument", json.dumps(i))
    print("decisions", out["decisions"])
    for c in ("primary_lev2", "lev2_vs_cs_aro", "primary_hpany", "hpany_vs_cs_aro"):
        if c in out:
            print(c, _line(out[c]))
    print("avg stake", {t: (e["early"]["avg_stake"], e["late"]["avg_stake"]) for t, e in exp.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
