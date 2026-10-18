"""ARO sizer x in-profit held-pair add (preregistration_stack.json).

Arm cs_aro_hp = cs_aro (ARO as a capital-neutral width-book sizer, cs_aro's own thresholds) with
031's in-profit add on. Instrument (no P&L read): adds in both halves; >= 95% of adds on a trade in
profit; no exit on an add candle; >= 99% of trades tagged :agree / :disagree, agree share within 0.05
of cs_aro's. Co-primary: judge_one(cs_aro_hp, w_aro_hp) AND judge_one(cs_aro_hp, cs_aro). Reported:
judge_one(cs_aro_hp, w_aro); trade / add counts; capital deployed (stake x hours held, scaled).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.stack
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.reentry import _adds  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_stack.json"
ARM = "cs_aro_hp"
COMPARATORS = ("w_aro_hp", "cs_aro")
CAPITAL = ("cs_aro_hp", "cs_aro", "w_aro_hp", "w_aro")


def _agree(tag: str) -> tuple[float, float]:
    from research.exit_coupling import feasibility as F

    side = [str(t["enter_tag"]).rsplit(":", 1)[-1] for t in F.archive_trades(tag)]
    tagged = [s for s in side if s in ("agree", "disagree")]
    return len(tagged) / max(len(side), 1), sum(s == "agree" for s in tagged) / max(len(tagged), 1)


def instrument(pr: dict) -> dict:
    adds, _ = _adds(ARM)
    a = pd.DataFrame(adds)
    tagged, share = _agree(ARM)
    _, ctl_share = _agree("cs_aro")
    out = {"adds": {h: int(a["fold"].isin(pr[h]).sum()) if len(a) else 0 for h in ("early", "late")},
           "in_profit_share": round(float(a["in_profit"].mean()), 4) if len(a) else None,
           "exits_on_add_candle": int(a["exit_on_add_candle"].sum()) if len(a) else 0,
           "tagged_share": round(tagged, 4), "agree_share": round(share, 4), "cs_aro_agree_share": round(ctl_share, 4)}
    out["pass"] = bool(all(out["adds"].values()) and (out["in_profit_share"] or 0) >= 0.95
                       and out["exits_on_add_candle"] == 0 and tagged >= 0.99 and abs(share - ctl_share) <= 0.05)
    return out


def capital(pr: dict) -> dict:
    from research.exit_coupling import feasibility as F

    out = {}
    for tag in CAPITAL:
        t = pd.DataFrame(F.archive_trades(tag))
        sh = t["stake_amount"] * t["scale"] * t["trade_duration"] / 60.0
        out[tag] = {h: round(float(sh[t["fold"].isin(pr[h])].sum()), 1) for h in ("early", "late")}
    return out


def decide(pr: dict, v: dict) -> dict:
    verdicts = {c: v[c]["verdict"] for c in COMPARATORS}
    gains = {f"{c}_{h}": v[c][f"{h}_gain_pp"] for c in COMPARATORS for h in ("early", "late")}
    if "FAIL" in verdicts.values():
        return {"decision": "FAIL", "reason": f"judge_one {verdicts}", "gains": gains}
    if any(x != "PASS" for x in verdicts.values()):
        return {"decision": "INCONCLUSIVE", "reason": f"judge_one {verdicts}", "gains": gains}
    if min(gains.values()) < pr["small_margin_pp"]:
        return {"decision": "INCONCLUSIVE", "reason": "small margin", "gains": gains}
    return {"decision": "PASS (PROVISIONAL)", "reason": f"judge_one {verdicts}", "gains": gains}


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
        v = {c: judge_one(pr, ARM, c) for c in COMPARATORS}
        out["decision"] = decide(pr, v)
        out.update({f"vs_{c}": v[c] for c in COMPARATORS})
        out["vs_w_aro"] = judge_one(pr, ARM, "w_aro")
        out["capital_stake_hours"] = capital(pr)
    (HERE / "verdicts_stack.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps(inst))
    print("decision", out["decision"])
    for c in ("vs_w_aro_hp", "vs_cs_aro", "vs_w_aro"):
        v = out.get(c)
        if v:
            print(c, v["arm"], "vs", v["control"], v["verdict"], round(v["arm_pp"], 2), round(v["ctl_pp"], 2),
                  round(v["early_gain_pp"], 2), round(v["late_gain_pp"], 2), v["paired"]["verdict"], v["paired"]["p"],
                  v["risk"]["verdict_sharpe"], v["risk"]["verdict_maxdd"], v["arm_trades"], v["ctl_trades"], v["failed"])
    if "capital_stake_hours" in out:
        print("capital", json.dumps(out["capital_stake_hours"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
