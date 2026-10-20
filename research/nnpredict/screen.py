"""039 S1 sizer screen (research R8; preregistration_sizer.json): does NNPredict's predicted move rank
the width book's early trades? No training, no backtest, early half only (the late half stays blind
for the sizer arm).

On the 412 early width-book trades (research.new_inputs.trades.trades(): reproduces the trade count
and cs_aro's 0.1004 control gap), the selected model's predict_gain at each trade's signal bar (the
NNP signal tables, research/nnnc_framework/signals.py --models NNP):
  coverage     share of trades with a finite score
  gap_share50  mean pp of the top half minus the bottom half by score, halves within fold
  partial      Spearman partial correlation of score with pp given bb_width (pooled; per fold)
  sign_folds   folds whose partial has the pooled sign
  rho_pctl     Spearman(score, the control's p_buy) -- orthogonality, reported

Both directions are screened (threshold amendment, option 3): "up" (score = predict_gain, as first
registered) and "flip" (score = -predict_gain; the decile check found Ridge's ranking inverted). The
screen passes if either direction passes; the passing direction (the larger gap_share50 if both) is
the one the sizer arm uses.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.screen \
        [--model NNP_RIDGE_EXC32 --prereg sizer_exc32 --out screen_exc32.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
SIG = STRAT / "research" / "nnnc_framework" / "signals"


def gap_share50(t: pd.DataFrame, col: str = "score") -> float:
    hi = t.groupby("fold")[col].transform("median") < t[col]
    return float(t.loc[hi, "pp"].mean() - t.loc[~hi, "pp"].mean())


def attach(t: pd.DataFrame, tables: dict) -> pd.DataFrame:
    parts = []
    for f, g in t.groupby("fold"):
        tb = tables[f][["pair", "date", "predict_gain"]].copy()
        tb["date"] = pd.to_datetime(tb["date"], utc=True)
        g = g.assign(signal_date=pd.to_datetime(g["signal_date"], utc=True))
        m = g.merge(tb, left_on=["pair", "signal_date"], right_on=["pair", "date"], how="left")
        m.index = g.index
        parts.append(m.drop(columns="date").rename(columns={"predict_gain": "score"}))
    return pd.concat(parts).loc[t.index]


def score(t: pd.DataFrame) -> dict:
    from research.new_inputs.screen import _sp, partial_spearman

    ok = np.isfinite(t["score"].to_numpy(float))
    s = t[ok]
    out = {"trades": int(len(t)), "covered": int(ok.sum()), "coverage": round(float(ok.mean()), 4)}
    if len(s) < 20:
        return out | {"pass": False, "reason": "fewer than 20 scored trades"}
    out["gap_share50"] = round(gap_share50(s), 5)
    out["partial"] = partial_spearman(s["score"], s["pp"], [s["bb_width"]])
    per = {f: partial_spearman(g["score"], g["pp"], [g["bb_width"]]) for f, g in s.groupby("fold")}
    out["per_fold"] = per
    sign = np.sign(out["partial"] or 0.0)
    out["sign_folds"] = int(sum(1 for v in per.values() if v is not None and np.sign(v) == sign and sign != 0))
    out["rho_pctl"] = _sp(s["score"], s["p_ctl"])
    return out


def decide(r: dict, pr: dict) -> dict:
    rule = pr["screen_rule"]
    ok = (r.get("gap_share50") is not None and r["gap_share50"] >= rule["min_gap_share50"]
          and (r.get("partial") or 0) > 0 and r.get("sign_folds", 0) >= rule["min_sign_folds"])
    return r | {"pass": bool(ok)}


def main(argv=None) -> int:
    import argparse

    from research.new_inputs import trades as TR
    from research.nnpredict import judge as J

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NNP", help="signal-table prefix (research/nnnc_framework/signals.py)")
    ap.add_argument("--prereg", default="sizer")
    ap.add_argument("--out", default="screen.json")
    args = ap.parse_args(argv)
    path, pr = J._prereg(args.prereg)
    bad = J._refused(path, {**pr, "arms": {}})
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    t = TR.trades()
    tables = {f: pd.read_parquet(SIG / f"{args.model}_{f}.parquet") for f in sorted(t["fold"].unique())}
    a = attach(t, tables)
    dirs = {"up": decide(score(a), pr), "flip": decide(score(a.assign(score=-a["score"])), pr)}
    passed = [d for d in dirs if dirs[d]["pass"]]
    best = max(passed, key=lambda d: dirs[d]["gap_share50"]) if passed else None
    r = {"pass": bool(passed), "direction": best, "model": args.model, "directions": dirs,
         "instrument": t.attrs.get("check")}
    (HERE / args.out).write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({"pass": r["pass"], "direction": best,
                      **{d: {k: v for k, v in x.items() if k != "per_fold"} for d, x in dirs.items()}}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
