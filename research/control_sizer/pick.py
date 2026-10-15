"""Control-sizer study: the per-fold agree threshold for `cs_aro`, ONE fold per process.

034's sizer picker (research/label_screen/pick.py, AGREE_RULE "p_buy") on `plan_cs_aro.json`:
first bar of each width-book candidate run in the fold's validation tail (training data the
weights never fit), threshold at the (1 - AGREE_SHARE) quantile of p_buy, 4 decimals, written to
the plan's `thresholds` / `threshold_picks`. `--verify` regenerates the fold class with the stored
threshold (the class the backtest loads) and checks every width-book entry tag equals
(p_buy > threshold) and the loaded threshold equals the stored one.

    PYTHONPATH=. ../../.venv/bin/python -m research.control_sizer.pick --fold f00 [--verify]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

PLAN = STRAT / "evaluation" / "folds" / "plan_cs_aro.json"


def _snapshot(plan: dict) -> dict:
    snap = Path(plan["params_snapshot"])
    snap = snap if snap.is_absolute() else STRAT.parents[1] / snap
    return json.loads(snap.read_text())


def _fold(plan: dict, fid: str) -> dict:
    return next(x for x in plan["folds"] if x["fold_id"] == fid)


def store(plan_path: Path, fid: str, threshold: float, extra: dict) -> float:
    t = round(float(threshold), 4)
    plan = json.loads(plan_path.read_text())
    plan.setdefault("thresholds", {})[fid] = t
    plan.setdefault("threshold_picks", {})[fid] = extra
    plan_path.write_text(json.dumps(plan, indent=1) + "\n")
    return t


def pick(fid: str) -> tuple[float, dict]:
    import run_folds as rf
    from research.label_screen.pick import p_buy
    from research.width_relative import pick as WP

    plan = json.loads(PLAN.read_text())
    f = _fold(plan, fid)
    if not (STRAT / "NNNC" / f"{f['strategy_class']}.py").is_file():
        rf.generate_fold(f, plan["base_strategy"], plan["family"], _snapshot(plan))
    frames, strat = WP._frames(plan, fid, 0.0)
    pb, effs = [], []
    for s in frames.values():
        rs = WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        probs = s[list(WP.PROB)].to_numpy()
        pb.append(p_buy(probs)[rs])
        effs.append(WP.effective(probs)[rs])
    score = np.concatenate(pb) if pb else np.empty(0)
    eff = np.concatenate(effs) if effs else np.empty(0)
    share = float(strat.AGREE_SHARE)
    t, achieved = WP.threshold_for_share(score, share)
    return t, {"target_share": share, "achieved_share": round(achieved, 4),
               "run_starts": int(len(score)), "buy_argmax_share": round(float(np.mean(eff > 0)), 4)}


def verify(fid: str) -> bool:
    import run_folds as rf
    from research.label_screen.pick import p_buy
    from research.width_relative import pick as WP

    plan = json.loads(PLAN.read_text())
    t = float(plan["thresholds"][fid])
    rf.generate_fold(_fold(plan, fid), plan["base_strategy"], plan["family"], _snapshot(plan),
                     threshold=t)
    frames, strat = WP._frames(plan, fid, None)
    loaded = float(strat.prediction_threshold.value)
    bad = n = 0
    for s in frames.values():
        enter = s["enter_long"].fillna(0).to_numpy() == 1
        score = p_buy(s[list(WP.PROB)].to_numpy())
        agree = s["enter_tag"].fillna("").astype(str).str.endswith(":agree").to_numpy()
        n += int(enter.sum())
        bad += int(np.sum(enter & (agree != (score > t))))
    print(f"cs_aro {fid}: verify threshold {t} loaded {loaded}: entry bars {n}, tag mismatches {bad}")
    return bad == 0 and abs(loaded - t) < 1e-9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="exactly one fold id")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return 0 if verify(a.fold) else 1
    t, extra = pick(a.fold)
    print(f"cs_aro {a.fold}: threshold {store(PLAN, a.fold, t, extra)} {json.dumps(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
