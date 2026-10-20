"""039 sizer threshold per fold (research R8): cs_aro's procedure on NNPredict's predicted move.

On the fold's TRAINING-window validation tail (research.width_relative.pick._frames: the last
1 - TRAIN_DATA_SPLIT of each pair's training data, which the weights never fit), take the width
book's entry run starts (the <C>_WB arm, the sizer's own control class) and set the threshold at the
(1 - AGREE_SHARE) quantile of %predict_gain there (4 decimals), so ~half of the training-window
width-book entries agree. Writes research/nnpredict/sizer_thresholds.json {train_timerange: thr}.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.pick [--folds f00,f01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
OUT = HERE / "sizer_thresholds.json"
AGREE_SHARE = 0.5


def threshold(scores: np.ndarray, share: float = AGREE_SHARE) -> tuple[float, float]:
    from research.width_relative.pick import threshold_for_share

    s = np.asarray(scores, dtype=float)
    return threshold_for_share(s[np.isfinite(s)], share)


def pick_fold(plan: dict, fid: str) -> dict:
    import run_folds as rf
    from research.width_relative import pick as WP

    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    rf.generate_fold(f, plan["base_strategy"], plan["family"], None, seed=1)
    frames, _ = WP._frames(plan, fid, None)
    scores = [s["%predict_gain"].to_numpy(float)[WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)]
              for s in frames.values()]
    sc = np.concatenate(scores) if scores else np.empty(0)
    t, achieved = threshold(sc)
    return {"fold": fid, "train_timerange": f["train_timerange"], "threshold": t,
            "achieved_share": round(achieved, 4), "run_starts": int(len(sc))}


def main(argv=None) -> int:
    from research.nnpredict.build_arms import selected

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds")
    ap.add_argument("--model", help="arm tag (default: the selected model)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    tag = a.model or selected()[0]
    plan = json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}_wb.json").read_text())
    out_p = Path(a.out)
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for f in plan["folds"]:
        if a.folds and f["fold_id"] not in a.folds.split(","):
            continue
        r = pick_fold(plan, f["fold_id"])
        out[r["train_timerange"]] = r
        out_p.write_text(json.dumps(out, indent=1))
        print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
