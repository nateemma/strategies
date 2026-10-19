"""Liquidity-rank sizing, seed-2 repeat (preregistration_liqsz_s2.json).

QZ1s2 judge_one(cs_aro_liqsz_s2, cs_aro_s4k_s2); QZ2s2 judge_one(cs_aro_all3k_liqsz_s2, cs_aro_all3k_s4k_s2);
then the pre-registered two-seed combination with the seed-1 gains in verdicts_liqsr2.json.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.liqsz_s2 <dump root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.cap import _decide, _line  # noqa: E402
from research.tb_model.liqnz import slippage  # noqa: E402
from research.tb_model.liqsr import _dumps  # noqa: E402
from research.tb_model.liqsr2 import FACTOR, _rows  # noqa: E402
from research.tb_model.seed2 import _patch_rec_path  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_liqsz_s2.json"
QUESTIONS = {"QZ1s2": ("cs_aro_liqsz_s2", "cs_aro_s4k_s2", "QZ1"), "QZ2s2": ("cs_aro_all3k_liqsz_s2", "cs_aro_all3k_s4k_s2", "QZ2")}
NULLS = (("cs_aro_s4k_s2", "cs_aro_s2"), ("cs_aro_all3k_s4k_s2", "cs_aro_all3k_s2"))


def _guard(pr: dict) -> list[str]:
    from research.control_sizer import judge as CJ
    from research.nnnc_switches import judge as J

    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        return [f"{rel} has uncommitted changes"]
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    if not ts:
        return [f"{rel} is not committed"]
    tags = [t for a, c, _ in QUESTIONS.values() for t in (a, c)]
    return [f"record missing or older than the pre-registration commit: {p.name}" for t in tags for f in pr["folds"]
            if not (p := J._rec_path(t, f)).is_file() or p.stat().st_mtime <= ts]


def main(argv=None) -> int:
    from research.nnnc_switches.judge import judge_one

    _patch_rec_path()
    root = Path((argv or sys.argv[1:])[0])
    pr = json.loads(PREREG.read_text())
    bad = _guard(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    s1 = json.loads((HERE / "verdicts_liqsr2.json").read_text())["judged"]
    out = {"label": pr["label"], "instrument": {}, "decisions": {}, "returns_only": {}, "combined": {}, "judged": {}}
    rows = {}
    for q, (arm, ctl, q1) in QUESTIONS.items():
        a = _rows(arm, _dumps(root / arm.removesuffix("_s2")))
        c = _rows(ctl, {})
        rows[arm] = a
        hi = a["size"] == ":liqhi"
        i = {"trades": len(a), "control_trades": len(c), "tagged_share": round(float(a["size"].notna().mean()), 4),
             "hi_share": round(float(hi.mean()), 4), "mean_factor": round(float(a["size"].map(FACTOR).mean()), 4),
             "tag_matches_dump": round(float((hi == a.keep).mean()), 4),
             "halves": {h: int(a.fold.isin(pr[h]).sum()) for h in ("early", "late")}, "adds": int(a.adds.clip(lower=0).sum())}
        i["pass"] = bool(i["tagged_share"] == 1.0 and 0.65 <= i["hi_share"] <= 0.85 and 0.95 <= i["mean_factor"] <= 1.05
                         and i["tag_matches_dump"] >= 0.99 and all(i["halves"].values()) and i["adds"] == 0
                         and abs(len(a) / len(c) - 1) <= 0.02)
        out["instrument"][q] = i
        if not i["pass"]:
            out["decisions"][q] = out["combined"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        v = judge_one(pr, arm, ctl)
        out["judged"][f"{arm}|{ctl}"] = v
        d = out["decisions"][q] = _decide(pr, [v], False)
        out["returns_only"][q] = _decide(pr, [v], True)
        v1 = s1[f"{arm.removesuffix('_s2')}|{ctl.removesuffix('_s2')}"]
        mean = {h: round((v[f"{h}_gain_pp"] + v1[f"{h}_gain_pp"]) / 2, 2) for h in ("early", "late")}
        comb = ("PASS (PROVISIONAL, 2 seeds)" if d["decision"] == "PASS (PROVISIONAL)" and min(mean.values()) >= 1.0
                else "FAIL" if d["decision"] == "FAIL" else "INCONCLUSIVE")
        out["combined"][q] = {"decision": comb, "two_seed_mean_gain": mean, "seed1": q1}
    for arm, ctl in NULLS:
        out["judged"][f"{arm}|{ctl}"] = judge_one(pr, arm, ctl)
    out["slippage"] = slippage(pr, rows)
    (HERE / "verdicts_liqsz_s2.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps(out["instrument"]))
    print("decisions", out["decisions"])
    print("returns_only", out["returns_only"])
    print("combined", out["combined"])
    for v in out["judged"].values():
        print(_line(v))
    for t, r in out["slippage"].items():
        print("slippage", t, json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
