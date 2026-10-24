"""040 T034 sizer threshold per fold: cs_aro's procedure on the selected NNMT model's p_buy.

On the fold's TRAINING-window validation tail (research.width_relative.pick._frames: the last
1 - TRAIN_DATA_SPLIT of each pair's training data, which the weights never fit), take the width
book's entry run starts (the <M>_wb arm, the sizer's identity twin) and set the threshold at the
(1 - AGREE_SHARE) quantile of p_buy there (4 decimals), so ~half of the training-window width-book
entries agree. Writes research/nnmt/sizer_thresholds.json {train_timerange: pick}; --seed 2 picks on the
seed-2 weights (run_folds' namespacing: <class>_s2) into sizer_thresholds_s2.json.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.pick [--folds f00,f01] [--seed 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.nnmt.build_arms import SIZER_THRESHOLDS, selected  # noqa: E402
from research.nnpredict.pick import threshold  # noqa: E402


def pick_fold(plan: dict, fid: str, seed: int = 1) -> dict:
    import run_folds as rf
    from research.width_relative import pick as WP

    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    if seed != 1:
        f = dict(f, strategy_class=f"{f['strategy_class']}_s{seed}")
        plan = dict(plan, folds=[f])
    rf.generate_fold(f, plan["base_strategy"], plan["family"], None, seed=seed)
    frames, _ = WP._frames(plan, fid, None)
    scores = [s["%predict_prob_buy"].to_numpy(float)[WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)]
              for s in frames.values()]
    sc = np.concatenate(scores) if scores else np.empty(0)
    t, achieved = threshold(sc)
    return {"fold": fid, "train_timerange": f["train_timerange"], "threshold": t,
            "achieved_share": round(achieved, 4), "run_starts": int(len(sc))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    path = SIZER_THRESHOLDS if a.seed == 1 else SIZER_THRESHOLDS.with_name(f"sizer_thresholds_s{a.seed}.json")
    tag = selected()[0]
    plan = json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}_wb.json").read_text())
    out = json.loads(path.read_text()) if path.exists() else {}
    for f in plan["folds"]:
        if a.folds and f["fold_id"] not in a.folds.split(","):
            continue
        r = pick_fold(plan, f["fold_id"], a.seed)
        out[r["train_timerange"]] = r
        path.write_text(json.dumps(out, indent=1))
        print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
