"""Signal-rank liquidity guard (preregistration_liqsr.json).

QS1 judge_one(st_g24_mom_liqsr, st_g24_mom_s4k); QS2 judge_one(cs_aro_liqsr, cs_aro_s4k);
QS3 judge_one(cs_aro_all3k_liqsr, cs_aro_all3k_s4k); RETURNS-ONLY secondary; instrument per question
first, from each arm's own FT_LIQ_GUARD_DUMP (no P&L read on a failure). Reported: vs the liqnz arms,
slippage, refusal share by pair.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.liqsr <dump root>/<arm>/*.csv
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

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_liqsr.json"
QUESTIONS = {"QS1": ("st_g24_mom_liqsr", "st_g24_mom_s4k"), "QS2": ("cs_aro_liqsr", "cs_aro_s4k"),
             "QS3": ("cs_aro_all3k_liqsr", "cs_aro_all3k_s4k")}
REPORTED = {"st_g24_mom_liqsr": "st_g24_mom_liqnz", "cs_aro_liqsr": "cs_aro_liqnz", "cs_aro_all3k_liqsr": "cs_aro_all3k_liqnz"}


def _dumps(d: Path) -> dict:
    out: dict = {}
    for p in sorted(d.glob("*.csv")):
        x = pd.read_csv(p)
        x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_localize(None)
        out.setdefault(p.name.rsplit("_", 1)[0], []).append(x)
    return out


def _sig(dumps: dict, pair: str, ts: pd.Timestamp):
    """The signal row in the trade's own fold dump (earliest-starting dump reaching ts)."""
    d = min((x for x in dumps.get(pair, []) if x.date.iloc[0] <= ts <= x.date.iloc[-1]),
            key=lambda x: x.date.iloc[0], default=None)
    if d is None:
        return None
    row = d[d.date == ts]
    return None if row.empty else row.iloc[0]


def _rows(tag: str, dumps: dict) -> pd.DataFrame:
    from research.exit_coupling import feasibility as F

    out = []
    for t in F.archive_trades(tag):
        s = _sig(dumps, t["pair"].replace("/", "_"), pd.Timestamp(t["open_date"]).tz_localize(None) - pd.Timedelta("15min"))
        out.append({"pair": t["pair"].split("/")[0], "fold": t["fold"], "stake": float(t["stake_amount"]),
                    "found": s is not None, "keep": bool(s is not None and s["keep"]),
                    "fails": bool(s is not None and pd.notna(s["thr"]) and not s["qv"] > s["thr"]),
                    "adds": sum(bool(o.get("ft_is_entry")) for o in t.get("orders") or []) - 1})
    return pd.DataFrame(out)


def instrument(pr: dict, root: Path) -> tuple[dict, dict]:
    out, rows = {}, {}
    for q, (arm, ctl) in QUESTIONS.items():
        dumps = _dumps(root / arm)
        a, c = _rows(arm, dumps), _rows(ctl, dumps)
        rows[arm], rows[ctl] = a, c
        i = {"dump_files": sum(len(v) for v in dumps.values()), "trades": len(a), "control_trades": len(c),
             "keep_share": round(float(a.keep.mean()), 4), "found_share": round(float(a.found.mean()), 4),
             "control_fail_share": round(float(c.fails.mean()), 4),
             "control_fail_by_pair": c.groupby("pair").fails.mean().round(2).to_dict(),
             "halves": {h: int(a.fold.isin(pr[h]).sum()) for h in ("early", "late")},
             "adds": int(a.adds.clip(lower=0).sum())}
        i["pass"] = bool(i["keep_share"] >= 0.99 and i["control_fail_share"] >= 0.10
                         and all(i["halves"].values()) and i["adds"] == 0)
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
        out["judged"][f"{arm}|{REPORTED[arm]}"] = judge_one(pr, arm, REPORTED[arm])
    out["slippage"] = slippage(pr, rows)
    (HERE / "verdicts_liqsr.json").write_text(json.dumps(out, indent=1, default=str))
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
