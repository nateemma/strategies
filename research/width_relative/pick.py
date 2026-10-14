"""033 T015: per-fold agree threshold for the width-relative sizer (research R7, analyze A2).

On the fold's TRAINING data only -- its validation tail (the last 1 - TRAIN_DATA_SPLIT of each
pair's frame, which the weights never fit) -- take the FIRST bar of each width-book candidate run
(enter_long rising 0 -> 1 per pair: the only bars that can open a trade on a free pair) and pick the
4-decimal threshold at the (1 - AGREE_SHARE) quantile of the model's effective BUY probability
(p_buy where BUY is the argmax, else 0). The fold class is loaded as a backtest loads it
(ft_bot_start). One fold per process (a 15-fold process was reaped for low memory in 032).

    PYTHONPATH=user_data/strategies .venv/bin/python -m research.width_relative.pick \\
        --model ARO --fold f00 --out <path> [--verify]
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
PROB = ("%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy")


def run_starts(enter: np.ndarray) -> np.ndarray:
    e = np.asarray(enter, dtype=bool)
    return e & ~np.r_[False, e[:-1]]


def effective(probs: np.ndarray) -> np.ndarray:
    """p_buy where BUY (index 2) is the argmax, else 0; NaN rows -> 0."""
    p = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0)
    return np.where(np.argmax(p, axis=1) == 2, p[:, 2], 0.0)


def threshold_for_share(eff: np.ndarray, share: float) -> tuple[float, float]:
    """(4-decimal threshold, achieved share of eff > threshold)."""
    if len(eff) == 0:
        raise ValueError("no candidate run starts to pick from")
    t = round(float(np.quantile(eff, 1.0 - share)), 4)
    return t, float(np.mean(eff > t))


def _frames(plan: dict, fid: str, threshold: float | None):
    import pick_threshold as P
    from freqtrade.data.converter import trim_dataframe

    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    strat, cfg = P._load_fold_context(f["strategy_class"], plan["family"], plan["config"])
    strat.ft_bot_start()  # the fold .json params, as backtesting loads them (032 amendment 2)
    if threshold is not None:
        strat.prediction_threshold.value = threshold
    data, timerange = P._load_train_data(strat, cfg, P.TrainWindow.from_fold(f))
    out = {}
    for pair, df in strat.advise_all_indicators(data).items():
        if df.empty:
            continue
        s = trim_dataframe(strat.advise_entry(df.copy(), {"pair": pair}), timerange,
                           startup_candles=strat.startup_candle_count)
        out[pair] = s.iloc[int(strat.TRAIN_DATA_SPLIT * len(s)):]  # validation tail
    return out, strat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("ARO", "GBR"), required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out")
    ap.add_argument("--plan", help="plan file name in evaluation/folds (default plan_wr_<m>_sz.json); "
                                   "the seed-2 repeat passes its own suffixed plan")
    ap.add_argument("--verify", action="store_true",
                    help="rescore at the plan's threshold: tags must equal eff > threshold")
    a = ap.parse_args(argv)
    plan = json.loads((FOLDS_DIR / (a.plan or f"plan_wr_{a.model.lower()}_sz.json")).read_text())
    if a.verify:
        t = float(plan["thresholds"][a.fold])
        frames, strat = _frames(plan, a.fold, None)  # the fold .json's own threshold
        loaded = float(strat.prediction_threshold.value)
        bad = n = 0
        for s in frames.values():
            enter = s["enter_long"].fillna(0).to_numpy() == 1
            eff = effective(s[list(PROB)].to_numpy())
            agree = s["enter_tag"].fillna("").astype(str).str.endswith(":agree").to_numpy()
            n += int(enter.sum())
            bad += int(np.sum(enter & (agree != (eff > t))))
        print(f"{a.fold}: verify plan threshold {t} loaded {loaded}: entry bars {n}, "
              f"tag mismatches {bad}")
        return 0 if (bad == 0 and abs(loaded - t) < 1e-9) else 1
    frames, strat = _frames(plan, a.fold, 0.0)
    effs = []
    for s in frames.values():
        rs = run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        effs.append(effective(s[list(PROB)].to_numpy())[rs])
    eff = np.concatenate(effs) if effs else np.empty(0)
    share = float(strat.AGREE_SHARE)
    t, achieved = threshold_for_share(eff, share)
    res = {"fold": a.fold, "threshold": t, "target_share": share,
           "achieved_share": round(achieved, 4), "run_starts": int(len(eff)),
           "buy_argmax_share": round(float(np.mean(eff > 0)), 4)}
    print(f"{a.fold}: " + json.dumps(res), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
