"""Seed-2 repeat and the all-available + any-signal add arm (preregistration_seed2.json).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.seed2 seed2   # Q1s2, Q4s2, Q5s2
    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.seed2 combo   # Q6

Seed-2 records are named `<tag>_fNN_s2_FNN.json` (run_folds --seed 2); the judge plans
plan_<arm>_s2.json carry those arm tags, and judge._rec_path is mapped to the same names here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.cap import _decide, _line  # noqa: E402
from research.tb_model.lev2 import WEIGHTS, _cap  # noqa: E402
from research.tb_model.reentry import _adds  # noqa: E402
from research.tb_model.stack import _agree  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_seed2.json"
PICKS = HERE / "seed2_picks.json"
BOOKS = ("cs_aro", "cs_aro_all2k", "cs_aro_all3k", "cs_aro_hpany_lev2", "cs_aro_hpany", "cs_aro_lev2")
LIMITS = {"cs_aro_all2k": 2000.0, "cs_aro_all3k": 3000.0}
Q_S2 = {"Q1s2": ("cs_aro_hpany_lev2_s2", ("cs_aro_hpany_s2", "cs_aro_lev2_s2")),
        "Q4s2": ("cs_aro_all2k_s2", ("cs_aro_s2",)),
        "Q5s2": ("cs_aro_all3k_s2", ("cs_aro_all2k_s2",))}
SEED1 = {"Q1s2": "PASS (PROVISIONAL)", "Q4s2": "FAIL", "Q5s2": "PASS (PROVISIONAL)"}
SEED1_RETURNS = {"Q1s2": "PASS (PROVISIONAL)", "Q4s2": "PASS (PROVISIONAL)", "Q5s2": "PASS (PROVISIONAL)"}


def _patch_rec_path():
    from research.nnnc_switches import judge as J

    orig = J._rec_path

    def rec_path(tag: str, fid: str) -> Path:
        if tag.endswith("_s2"):
            return STRAT / "regime" / "arms" / f"{tag[:-3]}_{fid}_s2_{fid.upper()}.json"
        return orig(tag, fid)

    J._rec_path = rec_path


def _guard(pr: dict, tags: list[str]) -> list[str]:
    from research.control_sizer import judge as CJ
    from research.nnnc_switches import judge as J

    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        return [f"{rel} has uncommitted changes"]
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    if not ts:
        return [f"{rel} is not committed"]
    return [f"record older than the pre-registration commit: {p.name}" for t in tags for f in pr["folds"]
            if (p := J._rec_path(t, f)).is_file() and p.stat().st_mtime <= ts]


def _stake_model_share(tag: str, limit: float) -> float | None:
    from research.exit_coupling import feasibility as F

    ok = []
    for t in F.archive_trades(tag):
        o = min((o for o in t["orders"] if o.get("ft_is_entry")), key=lambda o: o["order_filled_timestamp"])
        model = min(limit * WEIGHTS[str(t["enter_tag"]).rsplit(":", 1)[-1]], _cap(t["pair"], pd.Timestamp(t["open_date"])))
        ok.append(abs(float(o["amount"]) * float(o["safe_price"]) / model - 1) <= 0.01)
    return round(float(np.mean(ok)), 4) if ok else None


def inst_seed2(pr: dict) -> dict:
    from evaluation.pool import pool
    from research.nnnc_switches import judge as J

    out = {}
    missing = {b: pool(J._plan(f"{b}_s2"))["folds_missing"] for b in BOOKS}
    out["folds_missing"] = {b: m for b, m in missing.items() if m}
    seeds = set()
    for b in BOOKS:
        for f in J._plan(f"{b}_s2")["folds"]:
            py = STRAT / "NNNC" / f"{f['strategy_class']}.py"
            m = re.search(r"MODEL_SEED = (\d+)", py.read_text()) if py.is_file() else None
            seeds.add(int(m.group(1)) if m else None)
    out["model_seeds"] = sorted(seeds, key=str)
    picks = json.loads(PICKS.read_text()) if PICKS.is_file() else {}
    out["verify_all"] = bool(picks) and all(v.get("verify") for v in picks.get("folds", {}).values()) \
        and len(picks.get("folds", {})) == len(pr["folds"])
    if out["folds_missing"]:
        out["pass"] = False
        return out
    _, share = _agree("cs_aro_s2")
    out["cs_aro_s2_agree_share"] = round(share, 4)
    r1, r2 = J._recs("cs_aro", pr["folds"]), J._recs("cs_aro_s2", pr["folds"])
    out["cs_aro_hash_diff_folds"] = sum(r1[f]["trades_hash"] != r2[f]["trades_hash"] for f in pr["folds"])
    out["stake_model_share"] = {b: _stake_model_share(f"{b}_s2", lim) for b, lim in LIMITS.items()}
    out["pass"] = bool(out["model_seeds"] == [2] and out["verify_all"] and 0.45 <= share <= 0.55
                       and out["cs_aro_hash_diff_folds"] >= 1
                       and all((v or 0) >= 0.80 for v in out["stake_model_share"].values()))
    return out


def inst_combo(pr: dict) -> dict:
    from research.exit_coupling.feasibility import STOP_REASONS

    tag = "cs_aro_all3k_hpany"
    a = pd.DataFrame(_adds(tag)[0])
    on_add = a[a["exit_on_add_candle"]]
    _, share = _agree(tag)
    _, ctl = _agree("cs_aro")
    out = {"adds": {h: int(a["fold"].isin(pr[h]).sum()) for h in ("early", "late")},
           "adds_not_in_profit": int((~a["in_profit"]).sum()),
           "in_profit_share": round(float(a["in_profit"].mean()), 4),
           "nonstop_exits_on_add_candle": int((~on_add["exit_reason"].isin(STOP_REASONS)).sum()),
           "stop_exits_on_add_candle": int(on_add["exit_reason"].isin(STOP_REASONS).sum()),
           "first_fill_model_share": _stake_model_share(tag, 3000.0), "agree_share": round(share, 4)}
    out["pass"] = bool(all(out["adds"].values()) and out["adds_not_in_profit"] > 0
                       and out["nonstop_exits_on_add_candle"] == 0 and (out["first_fill_model_share"] or 0) >= 0.80
                       and abs(share - ctl) <= 0.05)
    return out


def seed2(pr: dict) -> dict:
    from research.nnnc_switches.judge import judge_one

    inst = inst_seed2(pr)
    out = {"label": pr["label"], "instrument": inst, "decisions": {}, "returns_only": {}, "replicates": {}, "judged": {}}
    if not inst["pass"]:
        out["decisions"] = {q: {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"} for q in Q_S2}
        return out
    for q, (arm, ctls) in Q_S2.items():
        vs = [judge_one(pr, arm, c) for c in ctls]
        out["decisions"][q] = _decide(pr, vs, False)
        out["returns_only"][q] = _decide(pr, vs, True)
        out["replicates"][q] = {"standard": out["decisions"][q]["decision"] == SEED1[q],
                                "returns_only": out["returns_only"][q]["decision"] == SEED1_RETURNS[q]}
        for v in vs:
            out["judged"][f"{arm}|{v['control']}"] = v
    for b in BOOKS[1:]:
        out["judged"][f"{b}_s2|cs_aro_s2"] = judge_one(pr, f"{b}_s2", "cs_aro_s2")
    for b in BOOKS:
        out["judged"][f"{b}_s2|{b}"] = judge_one(pr, f"{b}_s2", b)
    return out


def combo(pr: dict) -> dict:
    from research.nnnc_switches.judge import judge_one

    inst = inst_combo(pr)
    out = {"label": pr["label"], "instrument": inst, "judged": {}}
    if not inst["pass"]:
        out["decision"] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
        return out
    v = judge_one(pr, "cs_aro_all3k_hpany", "cs_aro_all3k")
    out["decision"] = _decide(pr, [v], False)
    out["returns_only"] = _decide(pr, [v], True)
    out["judged"]["cs_aro_all3k_hpany|cs_aro_all3k"] = v
    for c in ("cs_aro_hpany_lev2", "cs_aro"):
        out["judged"][f"cs_aro_all3k_hpany|{c}"] = judge_one(pr, "cs_aro_all3k_hpany", c)
    return out


def main(argv=None) -> int:
    step = (argv or sys.argv[1:])[0]
    pr = json.loads(PREREG.read_text())
    _patch_rec_path()
    tags = [f"{b}_s2" for b in BOOKS] if step == "seed2" else ["cs_aro_all3k_hpany"]
    bad = _guard(pr, tags)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = seed2(pr) if step == "seed2" else combo(pr)
    (HERE / f"verdicts_{step}.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps(out["instrument"], default=str))
    for k in ("decisions", "returns_only", "replicates", "decision"):
        if k in out:
            print(k, out[k])
    for v in out["judged"].values():
        if v.get("verdict") != "MISSING":
            print(_line(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
