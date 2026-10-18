"""Momentum book with a pair-relative liquidity guard (preregistration_liq.json).

QL1 judge_one(st_g24_mom_liq, st_g24_mom_s4k); QL2 judge_one(st_g24_mom_liq3k, st_g24_mom_liq);
RETURNS-ONLY secondary; instrument per question first (no P&L read on a failure). Reported: warm-up
effect, liq3k vs st_g24_mom, slippage re-score at 0 / 2.5 / 5 bps per side.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.liq
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.cap import _decide, _line  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_liq.json"
ARMS = ("st_g24_mom_s4k", "st_g24_mom_liq", "st_g24_mom_liq3k")
WINDOW, RATIO, LIMIT = 2880, 1.0, 3000.0
_SIG: dict = {}


def _signal(pair: str, ts: pd.Timestamp) -> pd.Series:
    """Signal candle (the one before the entry open): quote volume and its trailing-median ratio."""
    from research.clfstudy.data import load_pair

    if pair not in _SIG:
        d = load_pair(pair)
        qv = pd.Series((d["volume"] * d["close"]).to_numpy(float), index=pd.DatetimeIndex(d["date"]))
        _SIG[pair] = pd.DataFrame({"qv": qv, "med": qv.rolling(WINDOW, min_periods=WINDOW).median()})
    s = _SIG[pair]
    return s.iloc[s.index.searchsorted(ts) - 1]


def _rows(tag: str) -> pd.DataFrame:
    from research.exit_coupling import feasibility as F

    out = []
    for t in F.archive_trades(tag):
        s = _signal(t["pair"], pd.Timestamp(t["open_date"]))
        out.append({"key": (t["pair"], str(pd.Timestamp(t["open_date"]))), "fold": t["fold"],
                    "stake": float(t["stake_amount"]), "pr": float(t["profit_ratio"]),
                    "pa": float(t["profit_abs"]), "qv": s.qv, "passes": bool(s.qv > RATIO * s.med),
                    "adds": sum(bool(o.get("ft_is_entry")) for o in t.get("orders") or []) - 1})
    return pd.DataFrame(out)


def instrument(pr: dict, d: dict) -> dict:
    liq, s4k, l3k = d["st_g24_mom_liq"], d["st_g24_mom_s4k"], d["st_g24_mom_liq3k"]
    out = {"trades": {t: len(x) for t, x in d.items()},
           "liq_pass_share": round(float(liq.passes.mean()), 4),
           "s4k_fail_share": round(float((~s4k.passes).mean()), 4),
           "liq_halves": {h: int(liq.fold.isin(pr[h]).sum()) for h in ("early", "late")},
           "liq_adds": int(liq.adds.clip(lower=0).sum())}
    out["QL1_pass"] = bool(out["liq_pass_share"] >= 0.99 and out["s4k_fail_share"] > 0
                           and all(out["liq_halves"].values()) and out["liq_adds"] == 0)
    model = np.minimum(LIMIT, l3k.qv / 10)
    out["liq3k_model_share"] = round(float((abs(l3k.stake / model - 1) <= 0.01).mean()), 4)
    m = l3k.merge(liq, on="key", suffixes=("", "_l"))
    room = m[m.qv / 10 > 900]
    out["liq3k_room_trades"] = len(room)
    out["liq3k_room_median_ratio"] = round(float((room.stake / room.stake_l).median()), 4) if len(room) else None
    out["QL2_pass"] = bool(out["QL1_pass"] and out["liq3k_model_share"] >= 0.80
                           and (out["liq3k_room_median_ratio"] or 0) > 1)
    return out


def slippage(pr: dict, d: dict) -> dict:
    from evaluation.pool import pool
    from research.nnnc_switches import judge as J

    out = {}
    for tag, x in d.items():
        p = pool(J._plan(tag))
        base = {f["fold_id"]: f["return_pct"] or 0.0 for f in p["per_fold"]}
        t = pd.DataFrame(p["trades"])
        turn = t.stake_amount.astype(float) * (2 + t.profit_ratio.astype(float))
        row = {}
        for bps in (0.0, 2.5, 5.0):
            cost = (turn * bps / 1e4 * t.scale_pp_per_abs).groupby(t.fold_id).sum()
            adj = {f: v - cost.get(f, 0.0) for f, v in base.items()}
            row[f"{bps}bps"] = {h: round(sum(v for f, v in adj.items() if f in pr[h]), 2) for h in ("early", "late")}
        row["break_even_pct_per_side"] = round(100 * float(t.profit_abs.sum()) / float(turn.sum()), 4)
        row["median_stake"] = round(float(x.stake.median()), 1)
        out[tag] = row
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
    d = {t: _rows(t) for t in ("st_g24_mom",) + ARMS}
    inst = instrument(pr, d)
    out = {"label": pr["label"], "instrument": inst, "decisions": {}, "returns_only": {}, "judged": {}}
    for q, (arm, ctl) in {"QL1": ("st_g24_mom_liq", "st_g24_mom_s4k"),
                          "QL2": ("st_g24_mom_liq3k", "st_g24_mom_liq")}.items():
        if not inst[f"{q}_pass"]:
            out["decisions"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        v = judge_one(pr, arm, ctl)
        out["decisions"][q] = _decide(pr, [v], False)
        out["returns_only"][q] = _decide(pr, [v], True)
        out["judged"][f"{arm}|{ctl}"] = v
    out["judged"]["st_g24_mom_s4k|st_g24_mom"] = judge_one(pr, "st_g24_mom_s4k", "st_g24_mom")
    if inst["QL2_pass"]:
        out["judged"]["st_g24_mom_liq3k|st_g24_mom"] = judge_one(pr, "st_g24_mom_liq3k", "st_g24_mom")
    out["slippage"] = slippage(pr, d)
    (HERE / "verdicts_liq.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps(inst))
    print("decisions", out["decisions"])
    print("returns_only", out["returns_only"])
    for v in out["judged"].values():
        print(_line(v))
    for t, r in out["slippage"].items():
        print("slippage", t, json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
