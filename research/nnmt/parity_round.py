"""Start-date parity of the in-strategy transition stand-in (preregistration_transition_round.json "instrument").

The stand-in's fit predictions (pred.parquet beside each fitted fold) are computed on inputs built from the
FULL history (series start 2021-01-01). A backtest builds them from its own range plus startup only, and a
live bot from its ~920-candle buffer. For one fold and one host family this populates the fold class over
the backtest range exactly as freqtrade does (BTC/USDT from the same range, as the dataprovider gives it),
scores the fitted stand-in in-strategy and compares with pred.parquet on every test row.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.parity_round --family NNMT --fold f10 \
        --root transition_r9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNNC"), str(STRAT / "NNMT")]
ROOT = STRAT.parents[1]
PLANS = {"NNNC": "plan_cs_aro_tr_r9.json", "NNMT": "plan_tr_scale_mt_r9.json"}


def main(argv=None) -> int:
    import _transition_standin as TS
    import run_folds as RF
    from freqtrade.configuration import TimeRange
    from freqtrade.data import history
    from pick_threshold import _load_fold_context
    from utils.CloseHistoryStore import _grid

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(PLANS), required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--root", default="transition_r9")
    a = ap.parse_args(argv)
    plan = json.loads((STRAT / "evaluation" / "folds" / PLANS[a.family]).read_text())
    f = next(x for x in plan["folds"] if x["fold_id"] == a.fold)
    params = json.loads((ROOT / plan["params_snapshot"]).read_text()) if plan.get("params_snapshot") else None
    RF.generate_fold(f, plan["base_strategy"], plan["family"], params, seed=1)
    strat, cfg = _load_fold_context(f["strategy_class"], plan["family"], plan["config"])
    strat.ft_load_hyper_params(False)
    cutoff = f["train_timerange"].split("-")[1]
    st = TS.StandIn(cutoff, root=TS.ARTIFACTS.parent / a.root)
    ref = pd.read_parquet(TS.ARTIFACTS.parent / a.root / cutoff / "pred.parquet")
    ref = ref[ref["part"] == "test"].assign(date=lambda d: pd.to_datetime(d["date"], utc=True))
    tr = TimeRange.parse_timerange(f["test_timerange"])
    pairs = cfg["exchange"]["pair_whitelist"]
    data = history.load_data(datadir=cfg["datadir"], pairs=pairs + [TS.CONTEXT_PAIR], timeframe="15m", timerange=tr,
                             startup_candles=strat.startup_candle_count, fail_without_data=True)
    btc = data[TS.CONTEXT_PAIR]
    hb = _grid(history.load_pair_history(pair=TS.CONTEXT_PAIR, timeframe="15m", datadir=cfg["datadir"],
                                         fill_up_missing=False), 15)
    ts, te = pd.Timestamp(f["test_start"], tz="UTC"), pd.Timestamp(f["test_end"], tz="UTC")
    rows = []
    for pair in pairs:
        df = strat.advise_indicators(data[pair].copy(), {"pair": pair})
        hp = _grid(history.load_pair_history(pair=pair, timeframe="15m", datadir=cfg["datadir"],
                                             fill_up_missing=False), 15)
        X, tradeable = TS.features(df, btc, hp, hb, decimals=st.decimals)
        d = pd.to_datetime(df["date"], utc=True)
        win = ((d >= ts) & (d < te)).to_numpy()
        for cell in st.meta["cells"]:
            p = st.predict(cell, X, tradeable & win)
            m = win & tradeable
            rows.append(pd.DataFrame({"date": d[m].to_numpy(), "pair": pair, "cell": cell, "p": p[m]}))
    mine = pd.concat(rows, ignore_index=True).assign(date=lambda x: pd.to_datetime(x["date"], utc=True))
    out = {"family": a.family, "fold": a.fold, "root": a.root, "decimals": st.decimals, "cells": {}}
    for cell in st.meta["cells"]:
        m = ref[ref["cell"] == cell].merge(mine[mine["cell"] == cell], on=["date", "pair"], how="outer",
                                           suffixes=("_ref", "_new"), indicator=True)
        both = m[m["_merge"] == "both"]
        thr = st.threshold(cell, 80)
        out["cells"][cell] = {"rows_ref_only": int((m["_merge"] == "left_only").sum()),
                              "rows_new_only": int((m["_merge"] == "right_only").sum()),
                              "max_abs_p_diff": float((both["p_ref"] - both["p_new"]).abs().max()) if len(both) else None,
                              "fire_mismatch": int(((both["p_ref"] >= thr) != (both["p_new"] >= thr)).sum())}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
