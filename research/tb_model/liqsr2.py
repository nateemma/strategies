"""Run-level signal rank and liquidity-rank sizing (preregistration_liqsr2.json).

QR1-3 judge_one(<book>_liqrun, <book>_s4k); QZ1-2 judge_one(<book>_liqsz, <book>_s4k); RETURNS-ONLY
secondary; instrument per question first, from each arm's own FT_LIQ_GUARD_DUMP (no P&L read on a
failure). Reported: each arm vs its liqsr counterpart, slippage.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.liqsr2 <dump root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.cap import _decide, _line  # noqa: E402
from research.tb_model.liqnz import slippage  # noqa: E402
from research.tb_model.liqsr import _dumps, _sig  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_liqsr2.json"
QUESTIONS = {"QR1": ("st_g24_mom_liqrun", "st_g24_mom_s4k"), "QR2": ("cs_aro_liqrun", "cs_aro_s4k"),
             "QR3": ("cs_aro_all3k_liqrun", "cs_aro_all3k_s4k"), "QZ1": ("cs_aro_liqsz", "cs_aro_s4k"),
             "QZ2": ("cs_aro_all3k_liqsz", "cs_aro_all3k_s4k")}
FACTOR = {":liqhi": 1.0 / 0.875, ":liqlo": 0.5 / 0.875}


def _rows(tag: str, dumps: dict) -> pd.DataFrame:
    from research.exit_coupling import feasibility as F

    out = []
    for t in F.archive_trades(tag):
        s = _sig(dumps, t["pair"].replace("/", "_"), pd.Timestamp(t["open_date"]).tz_localize(None) - pd.Timedelta("15min"))
        et = str(t.get("enter_tag") or "")
        out.append({"pair": t["pair"].split("/")[0], "fold": t["fold"], "stake": float(t["stake_amount"]),
                    "keep": bool(s is not None and s["keep"]),
                    "fails": bool(s is not None and pd.notna(s["thr"]) and not s["qv"] > s["thr"]),
                    "size": next((k for k in FACTOR if et.endswith(k)), None),
                    "adds": sum(bool(o.get("ft_is_entry")) for o in t.get("orders") or []) - 1})
    return pd.DataFrame(out)


def instrument(pr: dict, root: Path) -> tuple[dict, dict]:
    out, rows = {}, {}
    for q, (arm, ctl) in QUESTIONS.items():
        dumps = _dumps(root / arm)
        a = _rows(arm, dumps)
        rows[arm] = a
        i = {"dump_files": sum(len(v) for v in dumps.values()), "trades": len(a),
             "halves": {h: int(a.fold.isin(pr[h]).sum()) for h in ("early", "late")},
             "adds": int(a.adds.clip(lower=0).sum())}
        ok = all(i["halves"].values()) and i["adds"] == 0
        if q.startswith("QR"):
            c = _rows(ctl, dumps)
            rows[ctl] = c
            i |= {"control_trades": len(c), "keep_share": round(float(a.keep.mean()), 4),
                  "control_fail_share": round(float(c.fails.mean()), 4)}
            ok = ok and i["keep_share"] >= 0.99 and i["control_fail_share"] >= 0.10
        else:
            hi = a["size"] == ":liqhi"
            i |= {"tagged_share": round(float(a["size"].notna().mean()), 4), "hi_share": round(float(hi.mean()), 4),
                  "mean_factor": round(float(a["size"].map(FACTOR).mean()), 4),
                  "tag_matches_dump": round(float((hi == a.keep).mean()), 4)}
            ok = (ok and i["tagged_share"] == 1.0 and 0.65 <= i["hi_share"] <= 0.85
                  and 0.95 <= i["mean_factor"] <= 1.05 and i["tag_matches_dump"] >= 0.99)
        i["pass"] = bool(ok)
        out[q] = i
    return out, rows


def main(argv=None) -> int:
    from research.control_sizer import judge as CJ
    from research.nnnc_switches.judge import judge_one

    root = Path((argv or sys.argv[1:])[0])
    pr = json.loads(PREREG.read_text())
    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        print(f"REFUSED: {rel} has uncommitted changes")
        return 2
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    bad = [x for a, _ in QUESTIONS.values() for x in CJ.refusals({**pr, "arm": a}, commit_ts=ts)] if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst, rows = instrument(pr, root)
    out = {"label": pr["label"], "instrument": inst, "decisions": {}, "returns_only": {}, "judged": {}}
    for q, (arm, ctl) in QUESTIONS.items():
        if not inst[q]["pass"]:
            out["decisions"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        v = judge_one(pr, arm, ctl)
        out["decisions"][q] = _decide(pr, [v], False)
        out["returns_only"][q] = _decide(pr, [v], True)
        out["judged"][f"{arm}|{ctl}"] = v
        sr = arm.rsplit("_", 1)[0] + "_liqsr"
        out["judged"][f"{arm}|{sr}"] = judge_one(pr, arm, sr)
    out["slippage"] = slippage(pr, {k: v for k, v in rows.items() if k in {a for a, _ in QUESTIONS.values()}})
    (HERE / "verdicts_liqsr2.json").write_text(json.dumps(out, indent=1, default=str))
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
