"""032 follow-up 1: re-pick the CO threshold so the arm produces as many ENTRY RUNS as its control
(2026-10-14; EXPLORATORY: 032's early half was read in S2 and f10-f14 totals were printed).

032 matched the share of BUY bars, but the CO arms took ~40% fewer trades: their signals come in
longer runs, and only a run's first bar can open a trade on a pair. This picks, per fold, the
threshold at which the CO model's number of entry runs on the TRAINING window equals the control's
at the control's own fold threshold. Only training data is read (pick_threshold.TrainWindow), and
only its VALIDATION TAIL: the last (1 - TRAIN_DATA_SPLIT) of each pair's frame, which neither model
trained on (early stopping only). Matching on the whole window failed: the control fires ~13x more
in-sample than out of sample (ARO f00: 14,603 runs, ~9% of bars), beyond the CO model's maximum.

  control runs: the control fold strategy as generated (its plan threshold), enter_long per pair.
  CO runs(t):   the CO fold strategy scored ONCE at prediction_threshold 0, so enter_long marks every
                bar that could enter at any threshold; at threshold t a bar enters iff it is such a
                bar and its %predict_prob_buy > t (argmax_with_threshold is a strict '>').
  Thresholds live on a 0.01 grid: prediction_threshold is DecimalParameter(decimals=2), which
  rounds whatever the plan writes (032's 0.5342 traded as 0.53). --verify checks the equivalence.
  A run = a maximal block of consecutive entry bars on one pair.

    PYTHONPATH=user_data/strategies .venv/bin/python -m research.candidate_training.run_match \\
        --model ARO --fold f00 --out <path>          # one fold per process (memory)
    ... --verify --fold f00                          # check the threshold-0 equivalence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "evaluation"))

FOLDS_DIR = STRAT / "evaluation" / "folds"
PLANS = {"ARO": ("wf_aro_g24", "co_aro"), "GBR": ("fs_gbr_gr", "co_gbr")}
GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)  # the values a decimals=2 threshold can take


def runs(sig: np.ndarray) -> int:
    """Number of maximal blocks of True in a 1-D boolean array."""
    s = np.asarray(sig, dtype=bool)
    if s.size == 0:
        return 0
    return int(s[0]) + int(np.sum(s[1:] & ~s[:-1]))


def runs_at(eff: list[np.ndarray], t: float) -> int:
    return sum(runs(e > t) for e in eff)


def pick_for_runs(eff: list[np.ndarray], target: int) -> tuple[float, int]:
    """(threshold, runs) with runs closest to `target` over the 0.01 grid the strategy can actually
    run (prediction_threshold is a DecimalParameter(decimals=2): a picked 0.5342 trades as 0.53);
    ties go to the higher threshold."""
    if not any((e > 0).any() for e in eff):
        raise ValueError("no bar can enter at any threshold")
    best = None
    for t in GRID[::-1]:  # high -> low, so the first of equal distances is the higher threshold
        r = runs_at(eff, float(t))
        if best is None or abs(r - target) < abs(best[1] - target):
            best = (float(t), r)
    return best


def _plan(tag: str) -> dict:
    return json.loads((FOLDS_DIR / f"plan_{tag}.json").read_text())


def _fold(plan: dict, fid: str) -> dict:
    return next(f for f in plan["folds"] if f["fold_id"] == fid)


def _signals(plan: dict, fid: str, threshold: float | None):
    """Per pair (enter_long, p_buy) on the fold's TRAINING window; threshold None = as generated."""
    import pick_threshold as P
    from freqtrade.data.converter import trim_dataframe

    f = _fold(plan, fid)
    strat, cfg = P._load_fold_context(f["strategy_class"], plan["family"], plan["config"])
    if threshold is not None:
        strat.prediction_threshold.value = threshold
    data, timerange = P._load_train_data(strat, cfg, P.TrainWindow.from_fold(f))
    out = {}
    for pair, df in strat.advise_all_indicators(data).items():
        if df.empty:
            continue
        s = trim_dataframe(strat.advise_entry(df.copy(), {"pair": pair}), timerange,
                           startup_candles=strat.startup_candle_count)
        s = s.iloc[int(strat.TRAIN_DATA_SPLIT * len(s)):]  # validation tail (approx. split row)
        out[pair] = (s["enter_long"].fillna(0).to_numpy() == 1,
                     s["%predict_prob_buy"].fillna(0.0).to_numpy(float))
    return out


def effective(sig: dict) -> list[np.ndarray]:
    return [np.where(e, p, 0.0) for e, p in (sig[k] for k in sorted(sig))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PLANS), required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true",
                    help="compare threshold-0 scoring + p>t against scoring at the 032 threshold")
    a = ap.parse_args(argv)
    ctl_tag, co_tag = PLANS[a.model]
    co_plan = _plan(co_tag)
    t032 = round(float(co_plan["thresholds"][a.fold]), 2)  # as the DecimalParameter loads it
    low = _signals(co_plan, a.fold, 0.0)
    eff = effective(low)
    if a.verify:
        real = _signals(co_plan, a.fold, t032)
        diff = sum(int(np.sum(real[k][0] != (eff_k > t032)))
                   for k, eff_k in zip(sorted(low), eff))
        n = sum(int(real[k][0].sum()) for k in real)
        print(f"{a.fold}: verify threshold {t032}: entry bars {n}, mismatched bars {diff}")
        return 0 if diff == 0 else 1
    ctl = _signals(_plan(ctl_tag), a.fold, None)
    target = sum(runs(e) for e, _ in ctl.values())
    t, r = pick_for_runs(eff, target)
    res = {"fold": a.fold, "control_runs": target,
           "control_bars": int(sum(e.sum() for e, _ in ctl.values())),
           "co_runs_032": runs_at(eff, t032), "threshold_032": t032,
           "threshold": round(t, 6), "co_runs": r, "co_bars": int(sum((e > t).sum() for e in eff))}
    print(f"{a.fold}: " + json.dumps(res), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
