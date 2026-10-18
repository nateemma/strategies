"""More capital through the liquidity cap (preregistration_cap.json).

Q1 cs_aro_hpany_lev2 (any-signal add + first fills x1.7432) vs cs_aro_hpany AND vs cs_aro_lev2;
Q2 cs_aro_hpany_m2 (2 adds) vs cs_aro_hpany; Q3 cs_aro_hpany_m3 vs cs_aro_hpany_m2. Instrument per arm
(no P&L read for a failed arm). Decision = judge_one; RETURNS-ONLY secondary ignores Sharpe / maxDD.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.cap
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.lev import _pooled  # noqa: E402
from research.tb_model.lev2 import BASE_STAKE, WEIGHTS, _cap  # noqa: E402
from research.tb_model.reentry import _adds  # noqa: E402
from research.tb_model.stack import _agree  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_cap.json"
AMEND = HERE / "preregistration_cap_amend.json"
K = 1.7432
ARMS = {"cs_aro_hpany_lev2": None, "cs_aro_hpany_m2": 2, "cs_aro_hpany_m3": 3}
QUESTIONS = {"Q1": ("cs_aro_hpany_lev2", ("cs_aro_hpany", "cs_aro_lev2")),
             "Q2": ("cs_aro_hpany_m2", ("cs_aro_hpany",)),
             "Q3": ("cs_aro_hpany_m3", ("cs_aro_hpany_m2",))}
EXPOSURE = ("cs_aro_hpany_lev2", "cs_aro_hpany_m2", "cs_aro_hpany_m3", "cs_aro_hpany", "cs_aro_lev2", "cs_aro")


def _n_adds(tag: str) -> pd.DataFrame:
    from research.exit_coupling import feasibility as F

    return pd.DataFrame([{"fold": t["fold"], "n": sum(bool(o.get("ft_is_entry")) for o in t.get("orders") or []) - 1}
                         for t in F.archive_trades(tag)])


def _first_fill_ok(tag: str) -> float | None:
    from research.exit_coupling import feasibility as F

    shared = {(t["pair"], str(pd.Timestamp(t["open_date"]))) for t in F.archive_trades("cs_aro")}
    ok = []
    for t in F.archive_trades(tag):
        if (t["pair"], str(pd.Timestamp(t["open_date"]))) not in shared:
            continue
        o = min((o for o in t["orders"] if o.get("ft_is_entry")), key=lambda o: o["order_filled_timestamp"])
        first = float(o["amount"]) * float(o["safe_price"])
        model = min(K * BASE_STAKE * WEIGHTS[str(t["enter_tag"]).rsplit(":", 1)[-1]],
                    _cap(t["pair"], pd.Timestamp(t["open_date"])))
        ok.append(abs(first / model - 1) <= 0.01)
    return round(float(np.mean(ok)), 4) if ok else None


def instrument(pr: dict, tag: str, max_adds: int | None) -> dict:
    a = pd.DataFrame(_adds(tag)[0])
    _, share = _agree(tag)
    _, ctl = _agree("cs_aro")
    n = _n_adds(tag)
    out = {"adds": {h: int(a["fold"].isin(pr[h]).sum()) if len(a) else 0 for h in ("early", "late")},
           "adds_not_in_profit": int((~a["in_profit"]).sum()) if len(a) else 0,
           "in_profit_share": round(float(a["in_profit"].mean()), 4) if len(a) else None,
           "exits_on_add_candle": int(a["exit_on_add_candle"].sum()) if len(a) else 0,
           "agree_share": round(share, 4), "adds_per_trade": dict(sorted(Counter(n["n"]).items()))}
    on_add = a[a["exit_on_add_candle"]] if len(a) else a
    if max_adds is not None and AMEND.is_file():  # preregistration_cap_amend.json: stops inside the candle allowed
        from research.exit_coupling.feasibility import STOP_REASONS

        stops = on_add["exit_reason"].isin(STOP_REASONS)
        out["stop_exits_on_add_candle"] = int(stops.sum())
        out["exits_on_add_candle"] = int((~stops).sum())
    ok = (all(out["adds"].values()) and out["adds_not_in_profit"] > 0 and out["exits_on_add_candle"] == 0
          and abs(share - ctl) <= 0.05)
    if max_adds is None:
        out["first_fill_model_share"] = _first_fill_ok(tag)
        ok = ok and (out["first_fill_model_share"] or 0) >= 0.95
    else:
        top = {h: int(((n["n"] == max_adds) & n["fold"].isin(pr[h])).sum()) for h in ("early", "late")}
        out[f"trades_with_{max_adds}_adds"] = top
        ok = ok and all(top.values()) and int(n["n"].max()) <= max_adds
    out["pass"] = bool(ok)
    return out


def _decide(pr: dict, vs: list[dict], returns_only: bool) -> dict:
    def ok(v):
        if not returns_only:
            return v["verdict"]
        if v["verdict"] in ("INCONCLUSIVE", "MISSING"):
            return v["verdict"]
        g = v["early_gain_pp"] > 0 and v["late_gain_pp"] > 0
        p = pr["rule"]["paired_map"].get(v["paired"]["verdict"], "inconclusive")
        return "FAIL" if not g or p == "fail" else ("INCONCLUSIVE" if p == "inconclusive" else "PASS")

    verdicts = [ok(v) for v in vs]
    gains = [g for v in vs for g in (v["early_gain_pp"], v["late_gain_pp"])]
    if "FAIL" in verdicts:
        d = "FAIL"
    elif any(x != "PASS" for x in verdicts):
        d = "INCONCLUSIVE"
    elif min(gains) < pr["small_margin_pp"]:
        d = "INCONCLUSIVE"
    else:
        d = "PASS (PROVISIONAL)"
    return {"decision": d, "verdicts": verdicts}


def _line(v: dict) -> str:
    r = v["risk"]
    return (f"{v['arm']} vs {v['control']} {v['verdict']} {v['arm_pp']:.2f} {v['ctl_pp']:.2f} "
            f"{v['early_gain_pp']:+.2f} / {v['late_gain_pp']:+.2f} paired {v['paired']['verdict']} p {v['paired']['p']} "
            f"sharpe {r['arm']['sharpe']:.2f}/{r['ctl']['sharpe']:.2f} {r['verdict_sharpe']} "
            f"maxdd {r['arm']['maxdd']:.2f}/{r['ctl']['maxdd']:.2f} {r['verdict_maxdd']} "
            f"trades {v['arm_trades']}/{v['ctl_trades']} {v['failed']}")


def main() -> int:
    from research.control_sizer import judge as CJ
    from research.held_pair import exposure as X
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
    inst = {a: instrument(pr, a, m) for a, m in ARMS.items()}
    pooled = {t: _pooled(t) for t in EXPOSURE}
    exp = {t: {h: {k: round(v, 1) for k, v in X.exposure([x for x in pooled[t] if x["fold_id"] in pr[h]]).items()}
               for h in ("early", "late")} for t in EXPOSURE}
    out = {"label": pr["label"], "instrument": inst, "exposure": exp, "decisions": {}, "returns_only": {}, "judged": {}}
    for q, (arm, ctls) in QUESTIONS.items():
        if not inst[arm]["pass"] or any(c in inst and not inst[c]["pass"] for c in ctls):
            out["decisions"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        vs = [judge_one(pr, arm, c) for c in ctls]
        out["decisions"][q] = _decide(pr, vs, False)
        out["returns_only"][q] = _decide(pr, vs, True)
        for v in vs:
            out["judged"][f"{arm}|{v['control']}"] = v
    for arm in ARMS:
        if inst[arm]["pass"]:
            out["judged"][f"{arm}|cs_aro"] = judge_one(pr, arm, "cs_aro")
    (HERE / "verdicts_cap.json").write_text(json.dumps(out, indent=1, default=str))
    for a, i in inst.items():
        print(a, "instrument", json.dumps(i))
    print("decisions", out["decisions"])
    print("returns_only", out["returns_only"])
    for v in out["judged"].values():
        print(_line(v))
    print("avg stake", {t: (e["early"]["avg_stake"], e["late"]["avg_stake"]) for t, e in exp.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
