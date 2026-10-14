"""034 instrument (T012): S0 per candidate, and the S2 arm checks once records exist. No P&L is
read: trade records are read for pair, dates, enter_tag and stake_amount only.

  --stage s0  per candidate on f00-f07 (CONTROL excluded; GFM = reused weights):
              weights present at the training plan's model key; model keys distinct from the
              control's and from every other candidate's; sizer thresholds present for every
              early fold, stored at <= 4 decimals, classes declaring 4 decimals; the S0a label
              figures (shares, Spearman with bb_width on rule-eligible bars) copied from audit.json.
  --stage s2  per survivor arm on f00-f07 (033's checks, research/width_relative/instrument.py):
              sz: trade set == the width book's (`w_aro`) on (pair, open_date, close_date), at
                  most 2 folds differing; 0 entries outside the width-book rule; 0 tag
                  mismatches; capital vs the width book within +-10%; realised agree share and
                  stake ratio; wallet shortfalls (reported);
              es: every entry's signal bar has the model's own decision (BUY argmax AND p_buy >
                  threshold) -- no trigger gate; threshold loaded at 4 dp; trades vs the control.
  Refuses if the pre-registration is uncommitted (s2 only reads records made after it).

    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.instrument --stage s0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC"), str(STRAT / "evaluation")]

from research.label_screen import candidates as C  # noqa: E402


HERE = Path(__file__).resolve().parent
FOLDS_DIR = STRAT / "evaluation" / "folds"
SIG = STRAT / "research" / "nnnc_framework" / "signals"
EARLY = tuple(f"f{i:02d}" for i in range(8))
HASH_DIFF_MAX, CAPITAL_BAND = 2, 0.10


def _plan(tag: str) -> dict:
    return json.loads((FOLDS_DIR / f"plan_{tag}.json").read_text())


def four_dp(x) -> bool:
    return x is not None and round(float(x), 4) == float(x)


def keys_distinct(keys: dict) -> dict:
    """keys: {tag: {fold: key}}; CONTROL included. Per tag: True if none of its fold keys is
    another tag's key for the same fold (GFM is expected to equal nothing but itself)."""
    out = {}
    for tag, ks in keys.items():
        out[tag] = all(ks[f] not in {o[f] for t, o in keys.items() if t != tag and f in o}
                       for f in ks)
    return out


def s0_pass(row: dict) -> bool:
    return bool(row["weights_missing"] == [] and row["key_distinct"]
                and row["sz_thresholds_missing"] == [] and row["sz_thresholds_4dp"]
                and row["classes_4dp"])


def sz_arm_summary(folds: dict) -> dict:
    F = folds.values()
    diff = [k for k, x in folds.items() if not x["trades_equal_width_book"]]
    caps = [x["stakes"].get("capital_ratio_vs_width") for x in F]
    s = {"entry_violations": sum(x["entry_violations"] for x in F),
         "tag_mismatches": sum(x["tag_mismatches"] for x in F),
         "thresholds_loaded_ok": all(x["threshold_loaded_ok"] for x in F),
         "zero_trade_folds": [k for k, x in folds.items() if x["n_trades"] == 0],
         "hash_diff_folds": diff,
         "capital_ratio_mean": round(float(np.nanmean([c for c in caps if c is not None])), 4)
         if any(c is not None for c in caps) else None}
    s["capital_ok"] = s["capital_ratio_mean"] is not None and \
        abs(s["capital_ratio_mean"] - 1) <= CAPITAL_BAND
    s["S0_pass"] = bool(s["entry_violations"] == 0 and s["tag_mismatches"] == 0
                        and s["thresholds_loaded_ok"] and len(diff) <= HASH_DIFF_MAX
                        and s["capital_ok"])
    s["S0_inconclusive"] = bool(s["zero_trade_folds"])
    return s


def es_arm_summary(folds: dict) -> dict:
    F = folds.values()
    s = {"non_model_entries": sum(x["non_model_entries"] for x in F),
         "thresholds_loaded_ok": all(x["threshold_loaded_ok"] for x in F),
         "zero_trade_folds": [k for k, x in folds.items() if x["n_trades"] == 0],
         "trades": sum(x["n_trades"] for x in F),
         "control_trades": sum(x["control_trades"] for x in F)}
    s["S0_pass"] = bool(s["non_model_entries"] == 0 and s["thresholds_loaded_ok"])
    s["S0_inconclusive"] = bool(s["zero_trade_folds"])
    return s


def _model_entry_check(trades, table, threshold, bars_for) -> int:
    from research.width_relative import instrument as W

    bad = 0
    for t in trades:
        _, dates = bars_for(t["pair"])
        e = int(dates.get_loc(pd.Timestamp(t["open_date"])))
        key = (t["pair"], W._utc(dates[e - 1]))
        if key not in table.index:
            bad += 1
            continue
        r = table.loc[key]
        bad += not (bool(r["buy_argmax"]) and float(r["p_buy"]) > threshold)
    return int(bad)


def stage_s0() -> dict:
    import run_folds as rf
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    audit = json.loads((HERE / "audit.json").read_text())
    cfg = load_config(STRAT / "config" / "config_wf.json")

    def key(plan, f):
        cls = load_strategy_class(plan["base_strategy"])
        pinned = type(f["strategy_class"], (cls,), {
            "MODEL_TRAIN_TIMERANGE": f["train_timerange"], "MODEL_SEED": 1,
            "__module__": cls.__module__})
        return MK.key_hash(MK.resolve(pinned, cfg))

    ctl = _plan("wf_aro_g24")
    keys = {"CONTROL": {f["fold_id"]: key(ctl, f) for f in ctl["folds"] if f["fold_id"] in EARLY}}
    out = {"label": "034 S0 (no P&L read)", "candidates": {}}
    for c in C.CANDIDATES:
        if c.tag == "CONTROL":
            continue
        es, sz = _plan(C.plan_tag(c, "es")), _plan(C.plan_tag(c, "sz"))
        early = [f for f in es["folds"] if f["fold_id"] in EARLY]
        keys[c.tag] = {f["fold_id"]: key(es, f) for f in early}
        thr = sz.get("thresholds", {})
        a = audit["candidates"][c.tag]
        out["candidates"][c.tag] = {
            "weights_missing": [f["fold_id"] for f in early
                                if not rf.fold_has_model(f, es["base_strategy"], es["config"], 1)],
            "sz_thresholds_missing": [f for f in EARLY if f not in thr],
            "sz_thresholds_4dp": all(four_dp(v) for v in thr.values()),
            "classes_4dp": all(load_strategy_class(n).prediction_threshold.decimals == 4
                               for n in (C.class_name(c), f"{C.class_name(c)}_SZ")),
            "s0a_status": a["status"],
            "label": {f: {"kept": r["kept"], "spearman_width_eligible": r["spearman_width_eligible"]}
                      for f, r in a["folds"].items()}}
    distinct = keys_distinct(keys)
    for tag, row in out["candidates"].items():
        # GFM is the reused GFM weights: distinct from the control and every trained candidate
        row["key_distinct"] = distinct[tag]
        row["S0_pass"] = s0_pass(row)
        print(tag, json.dumps({k: v for k, v in row.items() if k != "label"}), flush=True)
    return out


def stage_s2() -> dict:
    from evaluation.bootstrap import _trades
    from evaluation.fingerprint import load_strategy_class
    from evaluation.pool import arm_record
    from research.exit_coupling import feasibility as F
    from research.width_relative import instrument as W

    surv = json.loads((HERE / "survivors.json").read_text())["survivors"]
    wplan, ctl = _plan("w_aro"), _plan("wf_aro_g24")
    cuts = load_strategy_class("NNNC_W_ARO").WIDTH_CUTS
    out = {"label": "034 S2 instrument (no P&L read)", "arms": {}}
    for tag in surv:
        c = C.BY_TAG[tag]
        for role in c.roles:
            plan = _plan(C.plan_tag(c, role))
            thr = plan.get("thresholds", {})
            folds = {}
            for f, fw, fc in zip(plan["folds"], wplan["folds"], ctl["folds"]):
                fid = f["fold_id"]
                if fid not in EARLY:
                    continue
                table = W._table_index(pd.read_parquet(
                    SIG / (f"GFM_{fid}.parquet" if tag == "GFM" else f"LS_{tag}_{fid}.parquet")))
                trades = _trades(arm_record(f["arm_tag"], fid.upper())) or []
                jt = W.fold_json_threshold(f["strategy_class"])
                a = {"n_trades": len(trades),
                     "threshold_loaded_ok": jt is not None and fid in thr
                     and round(float(jt), 4) == round(float(thr[fid]), 4)}
                if role == "sz":
                    w_trades = _trades(arm_record(fw["arm_tag"], fid.upper())) or []
                    a |= W.entry_and_tag_checks(trades, table, float(cuts[f["train_timerange"]]),
                                                float(thr[fid]), F.bars_for)
                    a |= {"stakes": W.stake_checks(trades, w_trades)}
                    keys, wk = W.trade_keys(trades), W.trade_keys(w_trades)
                    a |= {"trades_equal_width_book": keys == wk,
                          "trades_only_arm": len(keys - wk), "trades_only_width": len(wk - keys)}
                else:
                    a["non_model_entries"] = _model_entry_check(trades, table, float(thr[fid]),
                                                                F.bars_for)
                    a["control_trades"] = len(_trades(arm_record(fc["arm_tag"], fid.upper())) or [])
                folds[fid] = a
            summ = sz_arm_summary(folds) if role == "sz" else es_arm_summary(folds)
            out["arms"][plan["run_id"]] = {"tag": tag, "role": role, "folds": folds, **summ}
            print(plan["run_id"], json.dumps(summ), flush=True)
    return out


def _prereg_committed() -> bool:
    rel = "research/label_screen/preregistration.json"
    st = subprocess.run(["git", "-C", str(STRAT), "status", "--porcelain", "--", rel],
                        capture_output=True, text=True).stdout.strip()
    log = subprocess.run(["git", "-C", str(STRAT), "log", "-1", "--format=%h", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return bool(log) and not st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("s0", "s2"))
    a = ap.parse_args(argv)
    if not _prereg_committed():
        print("REFUSED: research/label_screen/preregistration.json is uncommitted or dirty")
        return 2
    path = HERE / "instrument.json"
    cur = json.loads(path.read_text()) if path.is_file() else {}
    cur |= stage_s0() if a.stage == "s0" else stage_s2()
    path.write_text(json.dumps(cur, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
