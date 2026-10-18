"""All-available entry (preregistration_all.json).

cs_aro_all2k / cs_aro_all3k stake min(L x w, fillable cap, available wallet). Instrument per arm (no
P&L read for a failed arm). Q4 judge_one(cs_aro_all2k, cs_aro); Q5 judge_one(cs_aro_all3k, cs_aro_all2k);
RETURNS-ONLY secondary ignores Sharpe / maxDD. Reported: vs cs_aro_lev2 / cs_aro_hpany, exposure.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.all_in
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
from research.tb_model.lev import _pooled  # noqa: E402
from research.tb_model.lev2 import WEIGHTS, _cap  # noqa: E402
from research.tb_model.stack import _agree  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_all.json"
ARMS = {"cs_aro_all2k": 2000.0, "cs_aro_all3k": 3000.0}
QUESTIONS = {"Q4": ("cs_aro_all2k", "cs_aro"), "Q5": ("cs_aro_all3k", "cs_aro_all2k")}
REPORTED = ("cs_aro_lev2", "cs_aro_hpany")
EXPOSURE = ("cs_aro_all2k", "cs_aro_all3k", "cs_aro_lev2", "cs_aro_hpany", "cs_aro")


def instrument(tag: str, limit: float) -> dict:
    from research.exit_coupling import feasibility as F

    ctl = {(t["pair"], str(pd.Timestamp(t["open_date"]))): float(t["stake_amount"]) for t in F.archive_trades("cs_aro")}
    trades = F.archive_trades(tag)
    ok, stakes, ctl_stakes = [], [], []
    for t in trades:
        k = (t["pair"], str(pd.Timestamp(t["open_date"])))
        if k not in ctl:
            continue
        model = min(limit * WEIGHTS[str(t["enter_tag"]).rsplit(":", 1)[-1]], _cap(t["pair"], pd.Timestamp(t["open_date"])))
        ok.append(abs(float(t["stake_amount"]) / model - 1) <= 0.01)
        stakes.append(float(t["stake_amount"]))
        ctl_stakes.append(ctl[k])
    adds = sum((t.get("n_entries") or 1) - 1 for t in _pooled(tag))
    _, share = _agree(tag)
    _, cshare = _agree("cs_aro")
    out = {"trades": len(trades), "cs_aro_trades": len(ctl), "shared": len(ok),
           "model_share": round(float(np.mean(ok)), 4) if ok else None,
           "wallet_limited_share": round(1 - float(np.mean(ok)), 4) if ok else None,
           "median_stake": round(float(np.median(stakes)), 1) if stakes else None,
           "cs_aro_median_stake": round(float(np.median(ctl_stakes)), 1) if ctl_stakes else None,
           "adds": int(adds), "agree_share": round(share, 4)}
    out["pass"] = bool(adds == 0 and abs(share - cshare) <= 0.05 and (out["model_share"] or 0) >= 0.80
                       and (out["median_stake"] or 0) > (out["cs_aro_median_stake"] or 0))
    return out


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
    inst = {a: instrument(a, lim) for a, lim in ARMS.items()}
    pooled = {t: _pooled(t) for t in EXPOSURE}
    exp = {t: {h: {k: round(v, 1) for k, v in X.exposure([x for x in pooled[t] if x["fold_id"] in pr[h]]).items()}
               for h in ("early", "late")} for t in EXPOSURE}
    out = {"label": pr["label"], "instrument": inst, "exposure": exp, "decisions": {}, "returns_only": {}, "judged": {}}
    for q, (arm, ctl) in QUESTIONS.items():
        if not inst[arm]["pass"] or (ctl in inst and not inst[ctl]["pass"]):
            out["decisions"][q] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
            continue
        v = judge_one(pr, arm, ctl)
        out["decisions"][q] = _decide(pr, [v], False)
        out["returns_only"][q] = _decide(pr, [v], True)
        out["judged"][f"{arm}|{ctl}"] = v
    for arm in ARMS:
        if inst[arm]["pass"]:
            for c in REPORTED:
                out["judged"][f"{arm}|{c}"] = judge_one(pr, arm, c)
    (HERE / "verdicts_all.json").write_text(json.dumps(out, indent=1, default=str))
    for a, i in inst.items():
        print(a, "instrument", json.dumps(i))
    print("decisions", out["decisions"])
    print("returns_only", out["returns_only"])
    for v in out["judged"].values():
        print(_line(v))
    print("avg / peak stake", {t: (e["early"]["avg_stake"], e["late"]["avg_stake"], max(e["early"]["peak_stake"], e["late"]["peak_stake"]))
                               for t, e in exp.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
