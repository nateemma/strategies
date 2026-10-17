"""HGB sizer study: the per-fold agree threshold for `cs_hgb`, ONE fold per process.

cs_aro's rule (research/control_sizer/pick.py) on the HGB score: the first bar of each width-book
run in the fold's validation tail, threshold at the (1 - AGREE_SHARE) quantile of the HGB p_buy
(the strategy's `sizer_score` column), 4 decimals, stored in plan_cs_hgb.json. Also records, per
pair, whether the HGB's fit cutoff (signals.py) lies before the strategy's own tail start
(`tail_leak_pairs` must be empty). `--verify` regenerates the fold class with the stored threshold
and checks every width-book entry tag equals (score > threshold).

    PYTHONPATH=. ../../.venv/bin/python -m research.hgb_sizer.pick --fold f00 [--verify]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.control_sizer import pick as CP  # noqa: E402
from research.hgb_sizer.signals import OUT  # noqa: E402

PLAN = CP.FOLDS / "plan_cs_hgb.json"


def _scores(frames: dict) -> tuple[np.ndarray, dict]:
    from research.width_relative import pick as WP

    sc, firsts = [], {}
    for pair, s in frames.items():
        rs = WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        sc.append(np.nan_to_num(s["sizer_score"].to_numpy(float), nan=0.0)[rs])
        d = pd.Timestamp(s["date"].iloc[0])
        firsts[pair] = d.tz_convert("UTC") if d.tzinfo else d.tz_localize("UTC")
    return (np.concatenate(sc) if sc else np.empty(0)), firsts


def pick(fid: str, plan_path: Path = PLAN) -> tuple[float, dict]:
    import run_folds as rf
    from research.width_relative import pick as WP

    plan = json.loads(plan_path.read_text())
    f = CP._fold(plan, fid)
    rf.generate_fold(f, plan["base_strategy"], plan["family"], CP._snapshot(plan))
    frames, strat = WP._frames(plan, fid, 0.0)
    score, firsts = _scores(frames)
    share = float(strat.AGREE_SHARE)
    t, achieved = WP.threshold_for_share(score, share)
    meta = json.loads((OUT / f"HGB_{fid}.json").read_text())
    mine = {p: pd.Timestamp(v) for p, v in meta["tail_starts"].items()}
    leak = sorted(p for p, d in firsts.items() if p in mine and mine[p] > d)
    missing = int(sum(np.isnan(s["sizer_score"].to_numpy(float))[WP.run_starts(
        s["enter_long"].fillna(0).to_numpy() == 1)].sum() for s in frames.values()))
    return t, {"target_share": share, "achieved_share": round(achieved, 4), "run_starts": len(score),
               "run_starts_without_hgb_p": missing, "tail_leak_pairs": leak,
               "hgb_sha256": meta["sha256"]}


def verify(fid: str, plan_path: Path = PLAN) -> bool:
    import run_folds as rf
    from research.width_relative import pick as WP

    plan = json.loads(plan_path.read_text())
    t = float(plan["thresholds"][fid])
    rf.generate_fold(CP._fold(plan, fid), plan["base_strategy"], plan["family"], CP._snapshot(plan),
                     threshold=t)
    frames, strat = WP._frames(plan, fid, None)
    loaded = float(strat.prediction_threshold.value)
    bad = n = 0
    for s in frames.values():
        enter = s["enter_long"].fillna(0).to_numpy() == 1
        score = np.nan_to_num(s["sizer_score"].to_numpy(float), nan=0.0)
        agree = s["enter_tag"].fillna("").astype(str).str.endswith(":agree").to_numpy()
        n += int(enter.sum())
        bad += int(np.sum(enter & (agree != (score > t))))
    print(f"{plan['run_id']} {fid}: verify threshold {t} loaded {loaded}: entry bars {n}, tag mismatches {bad}")
    return bad == 0 and abs(loaded - t) < 1e-9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="exactly one fold id")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return 0 if verify(a.fold) else 1
    t, extra = pick(a.fold)
    print(f"cs_hgb {a.fold}: threshold {CP.store(PLAN, a.fold, t, extra)} {json.dumps(extra)}")
    return 0 if not extra["tail_leak_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
