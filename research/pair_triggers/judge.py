"""037 signal pairs as triggers: the pre-registered verdicts (preregistration.json).

Per arm: instrument (trigger causality PASS, trades >= min_arm_trades, no zero-trade half), then
  - alone: judge_one(arm, st_fam_gbb);
  - beside the width book: [w_aro, arm] through one shared wallet vs w_aro alone -- PASS = early
    AND late totals above w_aro's and the whole-period paired verdict not WORSENED.
Refuses while the pre-registration is uncommitted/dirty or any arm record predates its commit.

    PYTHONPATH=. ../../.venv/bin/python -m research.pair_triggers.judge
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.control_sizer import judge as CJ  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"
ARMS_DIR = STRAT / "regime" / "arms"


def refusals(pr: dict) -> list[str]:
    rel = PREREG.relative_to(STRAT)
    out = []
    if CJ._git("status", "--porcelain", "--", str(rel)):
        out.append(f"{rel} has uncommitted changes")
    ts = CJ._git("log", "-1", "--format=%ct", "--", str(rel))
    if not ts:
        return out + [f"{rel} is not committed"]
    for arm in pr["arms"]:
        for f in pr["folds"]:
            p = ARMS_DIR / f"{arm}_{f}_{f.upper()}.json"
            if p.is_file() and p.stat().st_mtime <= float(ts):
                out.append(f"record older than the pre-registration commit: {p.name}")
    return out


def beside_width(arm: str, bar: str = "w_aro") -> dict:
    from evaluation import books as B
    from evaluation.fingerprint import load_config
    from evaluation.halves import split

    cfg = load_config(str(STRAT / "config" / "config_wf.json"))
    stake = float(cfg["stake_amount"])
    pooled = {bar: B._pooled(bar), arm: B._pooled(arm)}
    books = B.shared_wallet(B.combine(pooled, 1.0, None, stake), stake * int(cfg["max_open_trades"]), stake)
    h, hb = B._halves(books), B._halves(pooled[bar])
    whole = split(books, pooled[bar], split_after="f14")["early"]
    overlap = B.capacity(books["trades"], int(cfg["max_open_trades"]))["same_pair_other_book"]
    ok = h["early"] > hb["early"] and h["late"] > hb["late"] and whole["verdict"] != "WORSENED"
    return {"verdict": "PASS" if ok else "FAIL", "combined": h, "width_book": hb,
            "early_gain_pp": round(h["early"] - hb["early"], 3), "late_gain_pp": round(h["late"] - hb["late"], 3),
            "paired_whole": {k: whole.get(k) for k in ("verdict", "delta_pp", "p")},
            "shared_wallet": {k: v for k, v in books.get("shared_wallet", {}).items() if not isinstance(v, (list, dict))},
            "same_pair_overlap_trades": overlap}


def instrument(pr: dict, arm: str, trigger: str) -> dict:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan
    from research.outcomes.causality import check_runnable

    ok_c, why = check_runnable(trigger)
    p = pool(_plan(arm))
    per = {f["fold_id"]: f.get("n_trades", 0) for f in p["per_fold"]}
    halves = {h: sum(per.get(f, 0) for f in pr[h]) for h in ("early", "late")}
    n = p["n_trades_pooled"]
    res = {"causality": ok_c, "causality_note": why, "trades": n, "trades_by_half": halves,
           "folds_missing": p["folds_missing"]}
    res["pass"] = bool(ok_c and not p["folds_missing"] and all(halves.values()) and n >= pr["min_arm_trades"])
    return res


def main() -> int:
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "arms": {}}
    for arm, spec in pr["arms"].items():
        inst = instrument(pr, arm, spec["trigger"])
        r = {"instrument": inst}
        if inst["pass"]:
            r["alone_vs_gbb"] = judge_one(pr, arm, "st_fam_gbb")
            r["beside_width_book"] = beside_width(arm)
            r["decision"] = ("candidate trigger" if "PASS" in (r["alone_vs_gbb"]["verdict"], r["beside_width_book"]["verdict"])
                             else "not a candidate")
        else:
            r["decision"] = "INCONCLUSIVE (instrument)"
        out["arms"][arm] = r
        a, b = r.get("alone_vs_gbb", {}), r.get("beside_width_book", {})
        print(arm, r["decision"], "| alone", a.get("verdict"), a.get("arm_pp"), "vs", a.get("ctl_pp"),
              a.get("early_gain_pp"), a.get("late_gain_pp"), (a.get("paired") or {}).get("verdict"),
              "| beside w_aro", b.get("verdict"), b.get("early_gain_pp"), b.get("late_gain_pp"),
              (b.get("paired_whole") or {}).get("verdict"), "| trades", inst["trades"], flush=True)
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
