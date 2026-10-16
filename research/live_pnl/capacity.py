"""036 T031: judge the capacity curve on net P&L (research R8; preregistration_capacity.json).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.capacity

Each arm (cap_aro_b5 … b40) is compared with cap_aro_ref on NET P&L. Trades are re-stated with
the committed cost model, and every pooled trade's profit_abs and every fold's return_pct are
replaced by their net values. The comparison then uses the harness's own paired test
(evaluation.compare_pooled) and risk verdict (evaluation.risk), with the decision rule of
research/nnnc_switches (judge_one's maps).

An arm whose stakes are not different from the reference (two-sample KS p >= 0.01) is INERT.

The tool refuses while the recording gate is closed, or while the pre-registration is uncommitted
or modified. It withholds any adoption offer while model_check.json has an open bias flag.
Writes research/live_pnl/capacity.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_capacity.json"
FOLDS = STRAT / "evaluation" / "folds"
REF = "cap_aro_ref"
BUDGETS = (5, 10, 20, 40)
EARLY = tuple(f"f{i:02d}" for i in range(8))
WALLET = 10000.0
KS_P = 0.01


def refusals(gate_open: bool, prereg_ok: bool) -> list[str]:
    out = []
    if not gate_open:
        out.append("recording gate is closed")
    if not prereg_ok:
        out.append("preregistration_capacity.json is uncommitted or modified")
    return out


def inert(arm_stakes, ref_stakes) -> tuple[bool, float]:
    from scipy.stats import ks_2samp

    p = float(ks_2samp(list(arm_stakes), list(ref_stakes)).pvalue)
    return p >= KS_P, p


def decide(ge: float, gl: float, paired: str, risk_s: str, risk_dd: str, rule: dict,
           n_ctl: int, min_ctl: int) -> tuple[str, list]:
    p = rule["paired_map"][paired]
    r = [rule["risk_map"][risk_s], rule["risk_map"][risk_dd]]
    if n_ctl < min_ctl or p == "inconclusive" or "inconclusive" in r:
        return "INCONCLUSIVE", []
    failed = [n for n, bad in (("early gain", not ge > 0), ("late gain", not gl > 0),
                               ("paired", p == "fail"), ("risk", "fail" in r)) if bad]
    return ("FAIL" if failed else "PASS"), failed


def net_pool(arm: str, cost_fn, qv_at) -> dict:
    from evaluation.pool import pool
    from research.live_pnl.restate import price_trades

    pl = pool(json.loads((FOLDS / f"plan_{arm}.json").read_text()))
    t = pd.DataFrame(pl["trades"])
    t["open_date"] = pd.to_datetime(t.open_date, utc=True)
    t["close_date"] = pd.to_datetime(t.close_date, utc=True)
    priced = price_trades(t, cost_fn, qv_at)
    for row, net in zip(pl["trades"], priced.net_abs):
        row["profit_abs"] = float(net)
    by_fold = priced.groupby("fold_id").net_abs.sum()
    for f in pl["per_fold"]:
        f["return_pct"] = 100 * float(by_fold.get(f["fold_id"], 0.0)) / WALLET
    pl["stakes"] = t.stake_amount.tolist()
    return pl


def main() -> int:
    from evaluation import risk
    from evaluation.compare_pooled import compare_data
    from research.live_pnl import cost_model as CM
    from research.live_pnl import gate

    rel = PREREG.relative_to(STRAT)
    dirty = subprocess.run(["git", "-C", str(STRAT), "status", "--porcelain", "--", str(rel)],
                           capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "-C", str(STRAT), "ls-files", "--", str(rel)],
                             capture_output=True, text=True).stdout.strip()
    bad = refusals(gate.is_open(), bool(tracked) and not dirty)
    if bad:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(bad))
    rule_src = json.loads((STRAT / "research" / "nnnc_switches" / "preregistration.json").read_text())
    rule, min_ctl = rule_src["rule"], rule_src["min_control_trades"]
    model, sha = CM.load()
    vol = {}

    def qv_at(pair, ts):
        if pair not in vol:
            vol[pair] = CM.qv24(pair)
        return float(vol[pair].asof(pd.Timestamp(ts).floor("15min") - pd.Timedelta(minutes=15)))

    def cost_fn(p, s, q, sd):
        return 0.0 if p in model["excluded_pairs"] else CM.cost_bps(model, p, s, q, sd)

    ref = net_pool(REF, cost_fn, qv_at)
    mc = HERE / "model_check.json"
    hold = mc.is_file() and json.loads(mc.read_text()).get("bias_flag", False)
    out = {"label": json.loads(PREREG.read_text())["label"], "cost_model_sha": sha,
           "adoption_offer": not hold, "reference": REF, "points": []}

    def half(pl, ids):
        return sum(f["return_pct"] for f in pl["per_fold"] if f["fold_id"] in ids)

    late = tuple(f["fold_id"] for f in ref["per_fold"] if f["fold_id"] not in EARLY)
    out["points"].append({"arm": REF, "budget_bps": None, "net_early_pct": round(half(ref, EARLY), 3),
                          "net_late_pct": round(half(ref, late), 3), "trades": ref["n_trades_pooled"],
                          "median_stake_ratio": round(float(pd.Series(ref["stakes"]).median()) / 900, 3)})
    for b in BUDGETS:
        arm = f"cap_aro_b{b}"
        ap = net_pool(arm, cost_fn, qv_at)
        is_inert, ks_p = inert(ap["stakes"], ref["stakes"])
        ge, gl = half(ap, EARLY) - half(ref, EARLY), half(ap, late) - half(ref, late)
        cmp = compare_data(ap, ref)
        rk = risk.compare(ap["trades"], ref["trades"])
        verdict, failed = ("INERT", []) if is_inert else decide(
            ge, gl, cmp["verdict"], rk["verdict_sharpe"], rk["verdict_maxdd"], rule,
            ref["n_trades_pooled"], min_ctl)
        out["points"].append({
            "arm": arm, "budget_bps": b, "verdict": verdict, "failed": failed, "ks_p": round(ks_p, 5),
            "net_early_pct": round(half(ap, EARLY), 3), "net_late_pct": round(half(ap, late), 3),
            "early_gain_pp": round(ge, 3), "late_gain_pp": round(gl, 3),
            "paired": cmp["verdict"], "risk": [rk["verdict_sharpe"], rk["verdict_maxdd"]],
            "trades": ap["n_trades_pooled"],
            "median_stake_ratio": round(float(pd.Series(ap["stakes"]).median()) / 900, 3)})
    (HERE / "capacity.json").write_text(json.dumps(out, indent=1) + "\n")
    for pt in out["points"]:
        print(pt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
