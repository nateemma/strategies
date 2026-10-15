"""Control-sizer study: instrument checks on `cs_aro`, per half. No P&L is read: trade records are
read for pair, dates, enter_tag and stake_amount only.

Per fold (034's sz checks, research/label_screen/instrument.py and research/width_relative/
instrument.py): trade set == the width book's (`w_aro`) on (pair, open_date, close_date); 0
entries outside the width-book rule; 0 tag mismatches (agree = p_buy > threshold, from the control's
signal table ARO_fNN); threshold loaded at 4 dp; capital vs the width book. Per half: at most 2
folds with a differing trade set, mean capital ratio within +-10%, no zero-trade fold.

    PYTHONPATH=. ../../.venv/bin/python -m research.control_sizer.instrument
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC"), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
FOLDS_DIR = STRAT / "evaluation" / "folds"
SIG = STRAT / "research" / "nnnc_framework" / "signals"


def _plan(tag: str) -> dict:
    return json.loads((FOLDS_DIR / f"plan_{tag}.json").read_text())


def fold_checks(f: dict, fw: dict, thr: dict, cuts: dict) -> dict:
    from evaluation.bootstrap import _trades
    from evaluation.pool import arm_record
    from research.exit_coupling import feasibility as F
    from research.width_relative import instrument as W

    fid = f["fold_id"]
    table = W._table_index(pd.read_parquet(SIG / f"ARO_{fid}.parquet"))
    trades = _trades(arm_record(f["arm_tag"], fid.upper())) or []
    w_trades = _trades(arm_record(fw["arm_tag"], fid.upper())) or []
    jt = W.fold_json_threshold(f["strategy_class"])
    a = {"n_trades": len(trades),
         "threshold_loaded_ok": jt is not None and fid in thr
         and round(float(jt), 4) == round(float(thr[fid]), 4)}
    a |= W.entry_and_tag_checks(trades, table, float(cuts[f["train_timerange"]]),
                                float(thr[fid]), F.bars_for, agree_rule="p_buy")
    a |= {"stakes": W.stake_checks(trades, w_trades)}
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
    out = {"label": "control-sizer instrument (no P&L read)", "folds": folds, "halves": {}}
    for h in ("early", "late"):
        s = sz_arm_summary({k: folds[k] for k in pr[h]})
        out["halves"][h] = s
        print(h, json.dumps(s), flush=True)
    out["pass"] = all(s["S0_pass"] and not s["S0_inconclusive"] for s in out["halves"].values())
    print("instrument pass:", out["pass"])
    (HERE / "instrument.json").write_text(json.dumps(out, indent=1, default=str))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
