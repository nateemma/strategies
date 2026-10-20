"""039 combined sizer (preregistration_mix.json): cs_aro's width book sized by the MEAN of ARO's p_buy
percentile and the excursion-H32 Ridge's predicted-move percentile -- cs_mix's design
(NNNC/_mix_sizer.MixSizerMixin) with nnp_ridge_exc32_op's score in place of the 037 composite.

ARO's p_buy is computed in the strategy (cs_aro's weights); the exc32 score is read from per-fold tables,
because one strategy carries one model. Steps, ONE fold per process (each loads a model):

  tables  user_data/data/nnp_exc32_sizer/HGB_fNN.parquet (date, pair, p_buy = %predict_gain): the
          fold's exc32 model on its validation tail (the WA arm's fold strategy: w_aro's width book) +
          its test window (research/nnnc_framework/signals NNP_RIDGE_EXC32_fNN)
  knots   user_data/data/mix_sizer_nnp_exc32/MIX_fNN.json: ARO p_buy and exc32 score at the validation-
          tail width-book run starts (cs_aro's fold strategy), sorted; HGB_fNN.json meta for the pick
  pick    the 0.5-share threshold of the mixed score at those run starts (research/hgb_sizer/pick.py,
          cs_mix's rule), stored in plan_cs_aro_nnp.json; --verify checks every tail tag

  check   instrument: the tables reproduce nnp_ridge_exc32_op_csa's in-backtest agree tags (>= 99%)

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.mix tables|knots|pick|verify --fold f00
    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnpredict.mix check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNNC"), str(STRAT / "NNPredict")]

DATA = STRAT.parents[1] / "user_data" / "data"
TABLES = DATA / "nnp_exc32_sizer"
KNOTS = DATA / "mix_sizer_nnp_exc32"
FOLDS = STRAT / "evaluation" / "folds"
SIG = STRAT / "research" / "nnnc_framework" / "signals"
SCORE_PLAN = FOLDS / "plan_nnp_ridge_exc32_op_wa.json"   # exc32 on w_aro's width book (identity-proven)
ARO_PLAN = FOLDS / "plan_cs_aro.json"                    # ARO's p_buy on the same width book
PLAN = FOLDS / "plan_cs_aro_nnp.json"


def _utc(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    return d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")


def tables(fid: str) -> dict:
    import run_folds as rf
    from research.width_relative import pick as WP

    plan = json.loads(SCORE_PLAN.read_text())
    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    rf.generate_fold(f, plan["base_strategy"], plan["family"], None, seed=1)
    frames, _ = WP._frames(plan, fid, None)
    tail = pd.concat([pd.DataFrame({"date": _utc(s["date"]), "pair": p, "p_buy": s["%predict_gain"].to_numpy(float)})
                      for p, s in frames.items()])
    test = pd.read_parquet(SIG / f"NNP_RIDGE_EXC32_{fid}.parquet")
    test = pd.DataFrame({"date": _utc(test["date"]), "pair": test["pair"], "p_buy": test["predict_gain"].astype(float)})
    tab = (pd.concat([tail, test]).drop_duplicates(["pair", "date"], keep="last")
           .sort_values(["pair", "date"]).reset_index(drop=True))
    TABLES.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(TABLES / f"HGB_{fid}.parquet", index=False)
    out = {"fold": fid, "rows": len(tab), "tail_rows": len(tail), "test_rows": len(test),
           "tail_first": str(tail["date"].min()), "test_last": str(test["date"].max()),
           "nan": int(tab["p_buy"].isna().sum())}
    print(json.dumps(out), flush=True)
    return out


def knots(fid: str) -> dict:
    import run_folds as rf
    from research.width_relative import pick as WP

    plan = json.loads(ARO_PLAN.read_text())
    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    from research.control_sizer import pick as CP

    rf.generate_fold(f, plan["base_strategy"], plan["family"], CP._snapshot(plan))
    frames, _ = WP._frames(plan, fid, 0.0)
    t = pd.read_parquet(TABLES / f"HGB_{fid}.parquet")
    t["date"] = _utc(t["date"])
    look = t.set_index(["pair", "date"])["p_buy"].sort_index()
    a, c = [], []
    for pair, s in frames.items():
        rs = WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        a.append(s["%predict_prob_buy"].to_numpy(float)[rs])
        ser = look.loc[pair] if pair in look.index.get_level_values(0) else pd.Series(dtype=float)
        c.append(_utc(s["date"]).map(ser).to_numpy(float)[rs])
    a, c = np.concatenate(a), np.concatenate(c)
    k = {"model": np.sort(a[np.isfinite(a)]).tolist(), "cmp": np.sort(c[np.isfinite(c)]).tolist()}
    KNOTS.mkdir(parents=True, exist_ok=True)
    (KNOTS / f"MIX_{fid}.json").write_text(json.dumps(k))
    sha = hashlib.sha256((TABLES / f"HGB_{fid}.parquet").read_bytes()).hexdigest()
    # fit_last empty, as for cs_aro's own score: the tail is the part of each pair's training data the
    # weights never fit (TRAIN_DATA_SPLIT), for both models
    meta = {"fold": fid, "run_starts": int(len(a)), "model_missing": int((~np.isfinite(a)).sum()),
            "cmp_missing": int((~np.isfinite(c)).sum()), "fit_last": {}, "sha256": sha,
            "rho_run_starts": float(pd.Series(a).corr(pd.Series(c), method="spearman"))}
    (KNOTS / f"HGB_{fid}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return meta


def check() -> dict:
    """Instrument: the tables carry what the backtest computes. Every nnp_ridge_exc32_op_csa trade (the
    exc32 sizer on the same width book, tagged from the IN-BACKTEST score) must get the same agree /
    disagree from the table value at its signal bar (open - 15 min) against that arm's threshold."""
    from evaluation.pool import pool
    from research.nnnc_switches.judge import _plan

    thr = json.loads((STRAT / "research" / "nnpredict" / "sizer_thresholds_exc32.json").read_text())
    by_fold = {f["fold_id"]: thr[f["train_timerange"]]["threshold"] for f in _plan("nnp_ridge_exc32_op_csa")["folds"]}
    t = pd.DataFrame(pool(_plan("nnp_ridge_exc32_op_csa"))["trades"])
    per, ok, n, miss = {}, 0, 0, 0
    for fid, g in t.groupby("fold_id"):
        tab = pd.read_parquet(TABLES / f"HGB_{fid}.parquet")
        tab["date"] = _utc(tab["date"])
        look = tab.set_index(["pair", "date"])["p_buy"]
        sig = _utc(g["open_date"]) - pd.Timedelta(minutes=15)
        v = look.reindex(pd.MultiIndex.from_arrays([g["pair"], sig])).to_numpy(float)
        agree_tab = v > by_fold[fid]
        agree_tag = g["enter_tag"].str.endswith(":agree").to_numpy()
        m = np.isfinite(v)
        per[fid] = {"trades": int(len(g)), "missing": int((~m).sum()), "match": int((agree_tab == agree_tag)[m].sum())}
        ok += per[fid]["match"]
        n += int(m.sum())
        miss += per[fid]["missing"]
    out = {"trades": int(len(t)), "missing": miss, "match_share": round(ok / n, 4) if n else None,
           "pass": bool(n) and ok / n >= 0.99 and miss / len(t) <= 0.01, "per_fold": per}
    (STRAT / "research" / "nnpredict" / "mix_table_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_fold"}), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("tables", "check", "knots", "pick", "verify"))
    ap.add_argument("--fold", help="exactly one fold id (not for check)")
    a = ap.parse_args(argv)
    if a.step == "check":
        return 0 if check()["pass"] else 1
    if a.step == "tables":
        tables(a.fold)
        return 0
    if a.step == "knots":
        knots(a.fold)
        return 0
    from research.hgb_sizer import pick as HP

    HP.OUT = KNOTS                     # read at call time inside pick()
    if a.step == "verify":
        return 0 if HP.verify(a.fold, PLAN) else 1
    t, extra = HP.pick(a.fold, PLAN)
    print(f"cs_aro_nnp {a.fold}: threshold {HP.CP.store(PLAN, a.fold, t, extra)} {json.dumps(extra)}", flush=True)
    return 0 if not extra["tail_leak_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
