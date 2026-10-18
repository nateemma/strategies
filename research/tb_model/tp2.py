"""TP2 arm (preregistration_tp2.json): instrument, then the verdict vs the width book.

Instrument (no P&L read): each recorded w_aro_tp2 trade is replayed from its entry under the arm's
exit -- take-profit 2 x the label ATR(14)/close of the signal bar, the folds' own stop / grace /
other exits (replay.exit_setup_json) -- and the replay must reproduce >= 90% of recorded exits
(reason AND bar). Also reported: take-profit level quantiles, share below the fixed 2.2%.
Verdict: research/nnnc_switches/judge.judge_one(arm, w_aro); small-margin rule. Reported beside it:
judge_one vs wf_aro_g24, and the replay's prediction (+3.53 / +3.51 pp on w_aro's trades).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.tp2
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model import replay as R  # noqa: E402  (patches feasibility.exit_setup)

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_tp2.json"
ARM, K, FIDELITY_MIN = "w_aro_tp2", 2.0, 0.90


def instrument() -> dict:
    from research.exit_coupling import feasibility as F
    from research.exit_coupling import paths as P

    df, _ = F.trade_rows(ARM, "NNNC_W_ARO_TP2_f00", "reversion")
    cls, params, roi = R.exit_setup_json("NNNC_W_ARO_TP2_f00")
    params = F.family_params(cls, params, "reversion")
    params = dataclasses.replace(params, take_profit_atr_mult=None)
    match, tps = [], []
    for r in df.itertuples():
        b, _ = F.bars_for(r.pair)
        tp = K * float(R.atr14(r.pair)[r.e - 1])
        tps.append(tp)
        x = P.replay(b, r.e, r.open_rate, dataclasses.replace(params, take_profit=tp), roi=roi,
                     roi_suppressed=r.roi_suppressed, end=r.end, tick=r.tick)
        match.append(x.reason == r.rec_reason and x.idx == r.rec_idx)
    match, tps = np.asarray(match), np.asarray(tps)
    by = {reason: round(float(match[df.rec_reason.to_numpy() == reason].mean()), 4)
          for reason in sorted(set(df.rec_reason))}
    out = {"n_trades": int(len(df)), "replay_agreement": round(float(match.mean()), 4),
           "by_recorded_reason": by, "tp_quantiles": np.percentile(tps, [5, 25, 50, 75, 95]).round(4).tolist(),
           "tp_below_fixed_share": round(float((tps < 0.022).mean()), 3),
           "exit_reasons": df.rec_reason.value_counts().to_dict()}
    out["pass"] = out["replay_agreement"] >= FIDELITY_MIN
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
    out = {"label": pr["label"], "instrument": inst}
    if not inst["pass"]:
        out["decision"] = {"decision": "INCONCLUSIVE", "reason": "instrument failed; no P&L read"}
    else:
        v = judge_one(pr, ARM, "w_aro")
        small = v["verdict"] == "PASS" and min(v["early_gain_pp"], v["late_gain_pp"]) < pr["small_margin_pp"]
        out["decision"] = {"decision": "INCONCLUSIVE" if small else
                           ("PASS (PROVISIONAL)" if v["verdict"] == "PASS" else v["verdict"]),
                           "reason": "small margin" if small else f"judge_one vs w_aro {v['verdict']}"}
        out["vs_w_aro"] = v
        out["vs_wf_aro_g24"] = judge_one(pr, ARM, "wf_aro_g24")
    (HERE / "verdicts_tp2.json").write_text(json.dumps(out, indent=1, default=str))
    print("instrument", json.dumps({k: inst[k] for k in ("replay_agreement", "pass", "tp_quantiles", "tp_below_fixed_share")}))
    print("decision", out["decision"])
    for c in ("vs_w_aro", "vs_wf_aro_g24"):
        v = out.get(c)
        if v:
            print(c, v["verdict"], v["arm_pp"], v["ctl_pp"], round(v["early_gain_pp"], 2), round(v["late_gain_pp"], 2),
                  v["paired"]["verdict"], v["paired"]["p"], v["risk"]["verdict_sharpe"], v["risk"]["verdict_maxdd"], v["failed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
