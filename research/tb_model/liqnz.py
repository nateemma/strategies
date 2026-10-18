"""Liquidity guard on the pair's traded candles (preregistration_liqnz.json).

QN1 judge_one(st_g24_mom_liqnz, st_g24_mom_s4k); QN2 judge_one(cs_aro_liqnz, cs_aro_s4k);
QN3 judge_one(cs_aro_all3k_liqnz, cs_aro_all3k_s4k); RETURNS-ONLY secondary; instrument per question
first (no P&L read on a failure). Reported: warm-up nulls, guarded arms vs 64-bar parents, slippage.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.liqnz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.cap import _decide, _line  # noqa: E402
from research.tb_model.stack import _agree  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_liqnz.json"
ARMS = ("st_g24_mom_liqnz", "cs_aro_s4k", "cs_aro_liqnz", "cs_aro_all3k_s4k", "cs_aro_all3k_liqnz")
QUESTIONS = {"QN1": ("st_g24_mom_liqnz", "st_g24_mom_s4k"), "QN2": ("cs_aro_liqnz", "cs_aro_s4k"),
             "QN3": ("cs_aro_all3k_liqnz", "cs_aro_all3k_s4k")}
NULLS = {"cs_aro_s4k": "cs_aro", "cs_aro_all3k_s4k": "cs_aro_all3k"}
PARENTS = {"st_g24_mom_liqnz": "st_g24_mom", "cs_aro_liqnz": "cs_aro", "cs_aro_all3k_liqnz": "cs_aro_all3k"}
_SIG: dict = {}


def _signal(pair: str, ts: pd.Timestamp, g: dict) -> pd.Series:
    """Signal candle (the one before the entry open): quote volume and the non-zero trailing median."""
    from research.clfstudy.data import load_pair

    if pair not in _SIG:
        d = load_pair(pair)
        qv = pd.Series((d["volume"] * d["close"]).to_numpy(float), index=pd.DatetimeIndex(d["date"]))
        med = qv.where(qv > 0).rolling(g["window"], min_periods=1).median()
        med.iloc[: g["window"] - 1] = float("nan")
        _SIG[pair] = pd.DataFrame({"qv": qv, "med": med})
    s = _SIG[pair]
    return s.iloc[s.index.searchsorted(ts) - 1]


def _rows(tag: str, g: dict) -> pd.DataFrame:
    from research.exit_coupling import feasibility as F

    out = []
    for t in F.archive_trades(tag):
        s = _signal(t["pair"], pd.Timestamp(t["open_date"]), g)
        out.append({"fold": t["fold"], "stake": float(t["stake_amount"]),
                    "passes": bool(s.qv > g["min_ratio"] * s.med),
                    "adds": sum(bool(o.get("ft_is_entry")) for o in t.get("orders") or []) - 1})
    return pd.DataFrame(out)


def instrument(pr: dict, d: dict) -> dict:
    out = {"trades": {t: len(x) for t, x in d.items()}}
    _, cshare = _agree("cs_aro")
    for tag in NULLS:
        _, share = _agree(tag)
        out[f"{tag}_agree_share"] = round(share, 4)
        out[f"{tag}_pass"] = bool(abs(share - cshare) <= 0.05)
    out["cs_aro_agree_share"] = round(cshare, 4)
    for q, (arm, ctl) in QUESTIONS.items():
        a, c = d[arm], d[ctl]
        i = {"pass_share": round(float(a.passes.mean()), 4), "control_fail_share": round(float((~c.passes).mean()), 4),
             "halves": {h: int(a.fold.isin(pr[h]).sum()) for h in ("early", "late")},
             "adds": int(a.adds.clip(lower=0).sum())}
        i["pass"] = bool(i["pass_share"] >= 0.99 and i["control_fail_share"] >= 0.10 and all(i["halves"].values())
                         and i["adds"] == 0 and out.get(f"{ctl}_pass", True))
        out[q] = i
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
        for bps in (0.0, 2.5, 5.0, 10.0):
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
    g = pr["guard"]
    tags = ("st_g24_mom_s4k",) + ARMS + tuple(NULLS.values()) + ("st_g24_mom",)
    d = {t: _rows(t, g) for t in tags}
    inst = instrument(pr, d)
    out = {"label": pr["label"], "instrument": inst, "decisions": {}, "returns_only": {}, "judged": {}}
    for q, (arm, ctl) in QUESTIONS.items():
        if not inst[q]["pass"]:
            out["decisions"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        v = judge_one(pr, arm, ctl)
        out["decisions"][q] = _decide(pr, [v], False)
        out["returns_only"][q] = _decide(pr, [v], True)
        out["judged"][f"{arm}|{ctl}"] = v
    for a, c in {**NULLS, **{a: p for a, p in PARENTS.items() if inst[next(q for q, v in QUESTIONS.items() if v[0] == a)]["pass"]}}.items():
        out["judged"][f"{a}|{c}"] = judge_one(pr, a, c)
    out["slippage"] = slippage(pr, d)
    (HERE / "verdicts_liqnz.json").write_text(json.dumps(out, indent=1, default=str))
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
