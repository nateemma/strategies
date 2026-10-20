"""NNNC framework study: per-bar entry-signal tables on each fold's TEST window (no backtest, no training).

For each model's walk-forward fold class (`NNNC_ARO_fNN`, `NNNC_GBR_fNN`, `NNNC_GFM_fNN`,
`NNNC_WF_fNN`) this runs the strategy's own `populate_indicators` -> `populate_entry_trend` over
the fold's test window -- one forward pass through the already-trained weights, exactly as
`evaluation/pick_threshold.py` does over a train window -- and keeps, per pair and bar: the BUY
probability, whether BUY is the argmax, `enter_long`, `enter_tag` and the guard inputs.
Strict model-key mode is forced: a missing model halts, it never trains.

The tables let stage 1 price ENTRY levers (threshold, guards, a regime-conditional threshold)
offline: `entries.py` turns signal rows into trades with the null exit replay and
one-open-trade-per-pair, and must first reproduce the recorded entries.

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.signals [--models ARO,GBR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "evaluation"))
os.environ["FT_MODEL_EVAL_STRICT"] = "1"

HERE = Path(__file__).resolve().parent
OUT = HERE / "signals"
CONFIG = "user_data/strategies/config/config_wf.json"
PLANS = {"ARO": "wf_aro_b030", "GBR": "sw_gbr_ctl_b030", "GFM": "sw_gfm_ctl_b030",
         "MLX": "sw_mlx_ctl_b030"}
#: 032: candidate-only arms (research/candidate_training). Not in the default --models list.
CO_PLANS = {"CO_ARO": "co_aro", "CO_GBR": "co_gbr",
            "AROV": "cs_arov"}  # research/aro_volume: ARO retrained on volume > 0 rows
#: research/xmodel_sizer: the GBR / MLX width-book sizers (their own p_buy on the test windows)
CO_PLANS |= {"CS_GBR": "cs_gbr", "CS_MLX": "cs_mlx",
             "CS_TBA": "cs_tba", "CS_TBB": "cs_tbb", "CS_TBD": "cs_tbd"}  # research/tb_model
#: 033: width-relative sizer arms (research/width_relative); the exit-off plan (same predictions).
WR_PLANS = {"WR_ARO": "wr_aro_sz", "WR_GBR": "wr_gbr_sz"}
#: 034: label-screen candidates (research/label_screen; the training `es` plans).
from research.label_screen.candidates import table_plans as _ls_table_plans  # noqa: E402

LS_PLANS = _ls_table_plans()
#: 035: new-input survivors (research/new_inputs; the training plans). Empty until survivors exist.
from research.new_inputs.arms import table_plans as _ni_table_plans  # noqa: E402

NI_PLANS = _ni_table_plans()


def _nnp_plans() -> dict:
    """039: the NNPredict model chosen by R7 (research/nnpredict/verdicts_threshold.json, else verdicts_op.json),
    tabled with its predicted move (`predict_gain`). Empty until that selection exists."""
    d = STRAT / "research" / "nnpredict"
    v, th = d / "verdicts_op.json", d / "verdicts_threshold.json"
    sel = (json.loads(th.read_text()).get("selected_union") if th.exists()
           else json.loads(v.read_text()).get("selected") if v.exists() else None)
    # the decile check (research/nnpredict/deciles.py): Ridge and the threshold amendment's retrains
    out = {"NNP_RIDGE": "nnp_ridge_op", "NNP_RIDGE_EXC": "nnp_ridge_exc_op", "NNP_RIDGE_H24": "nnp_ridge_h24_op",
           "NNP_RIDGE_EXC32": "nnp_ridge_exc32_op"}
    return out | ({"NNP": sel} if sel else {})


NNP_PLANS = _nnp_plans()
#: plan name -> strategy family directory (default NNNC)
PLAN_FAMILY = {plan: "NNPredict" for plan in NNP_PLANS.values()}
KEEP = ["volume", "close", "rvol", "atr_pct_roll", "guard_metric", "close_norm", "adx",
        "bb_width"]
PROB_COLS = ["%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy"]


def fold_table(cls: str, test_timerange: str, family: str = "NNNC") -> pd.DataFrame:
    from freqtrade.configuration import TimeRange
    from freqtrade.data import history
    from freqtrade.data.converter import trim_dataframe
    from pick_threshold import _load_fold_context

    strat, cfg = _load_fold_context(cls, family, CONFIG)
    # pick_threshold never calls ft_bot_start, so the fold's .json (its per-fold threshold, its
    # guards) is not applied there -- harmless for a probability distribution, wrong for
    # enter_long. Apply it exactly as ft_bot_start does.
    strat.ft_load_hyper_params(False)
    tr = TimeRange.parse_timerange(test_timerange)
    data = history.load_data(
        datadir=cfg["datadir"], pairs=cfg["exchange"]["pair_whitelist"], timeframe=cfg["timeframe"],
        timerange=tr, startup_candles=strat.startup_candle_count, fail_without_data=True,
        data_format=cfg.get("dataformat_ohlcv", "feather"),
        candle_type=cfg.get("candle_type_def", "spot"))
    pre = strat.advise_all_indicators(data)
    rows = []
    for pair, df in pre.items():
        if df.empty:
            continue
        sig = strat.advise_entry(df.copy(), {"pair": pair})
        sig = trim_dataframe(sig, tr, startup_candles=strat.startup_candle_count)
        probs = (sig[PROB_COLS].to_numpy(float) if set(PROB_COLS) <= set(sig.columns)
                 else np.full((len(sig), 3), np.nan))  # a regressor (NNPredict) has no class probabilities
        out = pd.DataFrame({"date": sig["date"].values, "pair": pair})
        if "%predict_gain" in sig.columns:  # 039: NNPredict's predicted move, the sizer score
            out["predict_gain"] = sig["%predict_gain"].to_numpy(float)
        out["p_buy"] = probs[:, 2]
        out["p_sell"], out["p_hold"] = probs[:, 0], probs[:, 1]
        out["buy_argmax"] = np.argmax(np.nan_to_num(probs, nan=-1.0), axis=1) == 2
        out["enter_long"] = sig.get("enter_long", 0).fillna(0).astype(int).values
        out["enter_tag"] = sig.get("enter_tag", "").fillna("").astype(str).values
        for c in KEEP:
            out[c] = sig[c].values if c in sig.columns else np.nan
        rows.append(out)
    t = pd.concat(rows, ignore_index=True)
    t.attrs["threshold"] = float(getattr(getattr(strat, "prediction_threshold", None), "value", np.nan))
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PLANS))
    ap.add_argument("--folds", help="comma-separated fold ids (default all); 034 screens f00-f07")
    a = ap.parse_args(argv)
    only = set(a.folds.split(",")) if a.folds else None
    OUT.mkdir(exist_ok=True)
    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for m in a.models.split(","):
        plan = json.loads((STRAT / "evaluation/folds" /
                           f"plan_{(PLANS | CO_PLANS | WR_PLANS | LS_PLANS | NI_PLANS | NNP_PLANS)[m]}.json").read_text())
        family = PLAN_FAMILY.get((PLANS | CO_PLANS | WR_PLANS | LS_PLANS | NI_PLANS | NNP_PLANS)[m], "NNNC")
        for f in plan["folds"]:
            if only and f["fold_id"] not in only:
                continue
            dest = OUT / f"{m}_{f['fold_id']}.parquet"
            if dest.exists():
                continue
            t = fold_table(f["strategy_class"], f["test_timerange"], family)
            t.to_parquet(dest)
            meta[f"{m}_{f['fold_id']}"] = {"class": f["strategy_class"],
                                           "threshold": t.attrs["threshold"],
                                           "rows": len(t), "enter_long": int(t["enter_long"].sum())}
            meta_path.write_text(json.dumps(meta, indent=1))
            print(m, f["fold_id"], len(t), "rows", int(t["enter_long"].sum()), "entries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
