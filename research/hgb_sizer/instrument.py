"""HGB sizer study: instrument checks on `cs_hgb`, per half. No P&L is read: trade records are read
for pair, dates, enter_tag and stake_amount only.

cs_aro's checks (research/control_sizer/instrument.py) with the tag check against the HGB's p_buy
(user_data/data/hgb_sizer/signals_fNN.parquet = the control's ARO_fNN table with p_buy replaced),
plus: width-book entries without an HGB p_buy, at most 1% per half.

    PYTHONPATH=. ../../.venv/bin/python -m research.hgb_sizer.instrument
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC"), str(STRAT / "evaluation")]

from research.control_sizer.instrument import _plan  # noqa: E402
from research.hgb_sizer.signals import OUT  # noqa: E402

HERE = Path(__file__).resolve().parent
MAX_MISSING = 0.01


def fold_checks(f: dict, fw: dict, thr: dict, cuts: dict) -> dict:
    from evaluation.bootstrap import _trades
    from evaluation.pool import arm_record
    from research.exit_coupling import feasibility as F
    from research.width_relative import instrument as W

    fid = f["fold_id"]
    table = W._table_index(pd.read_parquet(OUT / f"signals_{fid}.parquet"))
    trades = _trades(arm_record(f["arm_tag"], fid.upper())) or []
    w_trades = _trades(arm_record(fw["arm_tag"], fid.upper())) or []
    jt = W.fold_json_threshold(f["strategy_class"])
    a = {"n_trades": len(trades),
         "threshold_loaded_ok": jt is not None and fid in thr
         and round(float(jt), 4) == round(float(thr[fid]), 4)}
    a |= W.entry_and_tag_checks(trades, table, float(cuts[f["train_timerange"]]),
                                float(thr[fid]), F.bars_for, agree_rule="p_buy")
    missing = 0
    for t in trades:
        _, dates = F.bars_for(t["pair"])
        e = int(dates.get_loc(pd.Timestamp(t["open_date"])))
        key = (t["pair"], W._utc(dates[e - 1]))
        missing += key not in table.index or pd.isna(table.loc[key]["p_buy"])
    a |= {"stakes": W.stake_checks(trades, w_trades), "entries_without_hgb_p": int(missing)}
    keys, wk = W.trade_keys(trades), W.trade_keys(w_trades)
    return a | {"trades_equal_width_book": keys == wk,
                "trades_only_arm": len(keys - wk), "trades_only_width": len(wk - keys)}


def main() -> int:
    from evaluation.fingerprint import load_strategy_class
    from research.label_screen.instrument import sz_arm_summary

    pr = json.loads((HERE / "preregistration.json").read_text())
    plan, wplan = _plan(pr["arm"]), _plan(pr["comparators"]["width"])
    cuts = load_strategy_class(plan["base_strategy"]).WIDTH_CUTS
    thr = plan.get("thresholds", {})
    folds = {f["fold_id"]: fold_checks(f, fw, thr, cuts)
             for f, fw in zip(plan["folds"], wplan["folds"])}
    out = {"label": "HGB-sizer instrument (no P&L read)", "folds": folds, "halves": {}}
    for h in ("early", "late"):
        s = sz_arm_summary({k: folds[k] for k in pr[h]})
        n = sum(folds[k]["n_trades"] for k in pr[h])
        miss = sum(folds[k]["entries_without_hgb_p"] for k in pr[h])
        s["missing_hgb_share"] = round(miss / n, 4) if n else None
        s["missing_hgb_ok"] = bool(n) and miss / n <= MAX_MISSING
        out["halves"][h] = s
        print(h, json.dumps(s), flush=True)
    out["pass"] = all(s["S0_pass"] and not s["S0_inconclusive"] and s["missing_hgb_ok"]
                      for s in out["halves"].values())
    print("instrument pass:", out["pass"])
    (HERE / "instrument.json").write_text(json.dumps(out, indent=1, default=str))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
