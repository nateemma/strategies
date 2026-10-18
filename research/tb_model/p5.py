"""All-available entry at 5% participation (preregistration_p5.json).

cs_aro_all3k_p5 = cs_aro_all3k with QUOTE_VOLUME_HEADROOM_MULT 20. Instrument first (no P&L read on a
failure); Q7 judge_one(cs_aro_all3k_p5, cs_aro), RETURNS-ONLY secondary; reported vs cs_aro_all3k.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.p5
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
from research.tb_model.lev2 import HEADROOM, WEIGHTS, _cap  # noqa: E402
from research.tb_model.stack import _agree  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_p5.json"
ARM, LIMIT, MULT = "cs_aro_all3k_p5", 3000.0, 20.0


def instrument() -> dict:
    from research.exit_coupling import feasibility as F

    key = lambda t: (t["pair"], str(pd.Timestamp(t["open_date"])))  # noqa: E731
    ctl = {key(t) for t in F.archive_trades("cs_aro")}
    a3k = {key(t): float(t["stake_amount"]) for t in F.archive_trades("cs_aro_all3k")}
    trades = F.archive_trades(ARM)
    ok, ratio, stakes = [], [], []
    for t in trades:
        k = key(t)
        stakes.append(float(t["stake_amount"]))
        if k in a3k:
            ratio.append(float(t["stake_amount"]) / a3k[k])
        if k not in ctl:
            continue
        cap = _cap(t["pair"], pd.Timestamp(t["open_date"])) * HEADROOM / MULT
        model = min(LIMIT * WEIGHTS[str(t["enter_tag"]).rsplit(":", 1)[-1]], cap)
        ok.append(abs(float(t["stake_amount"]) / model - 1) <= 0.01)
    adds = sum((t.get("n_entries") or 1) - 1 for t in _pooled(ARM))
    _, share = _agree(ARM)
    _, cshare = _agree("cs_aro")
    out = {"trades": len(trades), "all3k_trades": len(a3k), "cs_aro_trades": len(ctl), "shared_cs_aro": len(ok),
           "model_share": round(float(np.mean(ok)), 4) if ok else None,
           "median_ratio_vs_all3k": round(float(np.median(ratio)), 4) if ratio else None,
           "median_stake": round(float(np.median(stakes)), 1) if stakes else None,
           "all3k_median_stake": round(float(np.median(list(a3k.values()))), 1),
           "adds": int(adds), "agree_share": round(share, 4), "cs_aro_agree_share": round(cshare, 4)}
    out["pass"] = bool((out["model_share"] or 0) >= 0.80 and (out["median_ratio_vs_all3k"] or 1) < 1
                       and adds == 0 and abs(share - cshare) <= 0.05)
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
    bad = CJ.refusals({**pr, "arm": ARM}, commit_ts=ts) if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = instrument()
    out = {"label": pr["label"], "instrument": inst, "judged": {}}
    if not inst["pass"]:
        out["decision"] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
    else:
        v = judge_one(pr, ARM, "cs_aro")
        out["decision"] = _decide(pr, [v], False)
        out["returns_only"] = _decide(pr, [v], True)
        out["judged"][f"{ARM}|cs_aro"] = v
        out["judged"][f"{ARM}|cs_aro_all3k"] = judge_one(pr, ARM, "cs_aro_all3k")
    (HERE / "verdicts_p5.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps(inst))
    print("decision", out["decision"], "returns_only", out.get("returns_only"))
    for v in out["judged"].values():
        print(_line(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
