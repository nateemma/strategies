"""033 S0 (T021): label and instrument checks on the width-relative arms. No P&L is read: trade
records are read for pair, dates, enter_tag, stake_amount and exit_reason only.

Per model (ARO, GBR):
  label      every fold's width_relative.json: each side's bucket shares in [0.40, 0.60] and
             |Spearman(label, bb_width)| <= 0.05 on the training rows; kept, BUY, SELL > 0;
             row_filter.json class counts;
  identity   exit-off arm (`wr_<m>_sz`): per-fold set of (pair, open_date, close_date) == the
             width book's (`w_<m>`); a differing fold is flagged; more than 2 of 15 fails S0
             (pre-registration amendment: trades_hash includes profit_ratio, which the stake's
             coin-amount rounding moves at ~1e-7);
  entries    both arms: recorded entries whose signal bar (entry bar - 1) fails the width-book rule
             (gbb rule AND bb_width >= the fold's cut) -- must be 0;
  tags       both arms: ':agree' tags must equal buy_argmax AND p_buy > threshold at the signal bar;
  threshold  each fold's generated .json threshold equals the plan's to 4 decimals;
  stakes     realised agree share among trades, mean stake vs the width book's 900, and trades whose
             stake is below the intended proposed x weight (flag above 5% of a fold's trades);
  exits      exit-on arm (`wr_<m>_szx`): model exits per fold and exit-gate-passing bars (rvol > 2.0
             AND guard_metric > -0.1) inside open trades; a fold with such bars and no model exit
             fails liveness;
  zero-trade folds are INCONCLUSIVE.

    PYTHONPATH=. ../../.venv/bin/python -m research.width_relative.instrument
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "NNNC"))

HERE = Path(__file__).resolve().parent
FOLDS_DIR = STRAT / "evaluation" / "folds"
SIG = STRAT / "research" / "nnnc_framework" / "signals"
PROPOSED_STAKE = 900.0
EXIT_GATE = (2.0, -0.1)  # rvol >, guard_metric > (the controls' exit gate)
SHARE_BOUNDS, SPEARMAN_MAX, HASH_DIFF_MAX, SHORT_FLAG = (0.40, 0.60), 0.05, 2, 0.05


def label_checks(rep: dict | None, rf: dict | None) -> dict:
    if rep is None:
        return {"present": False, "pass": False}
    lo, hi = SHARE_BOUNDS
    shares = [b["pos_share"] for b in rep["buckets"]]
    ok_share = all(lo <= s <= hi for s in shares)
    sp = [rep.get("spearman_buy"), rep.get("spearman_sell")]
    ok_sp = all(x is not None and abs(x) <= SPEARMAN_MAX for x in sp)
    counts = (rf or {}).get("train", {}).get("class_counts", [0, 0, 0])
    ok_n = rep["kept"] > 0 and rep.get("labelled_buy", 0) > 0 and rep.get("labelled_sell", 0) > 0
    return {"present": True, "share_range": [min(shares), max(shares)], "spearman": sp,
            "train_class_counts": counts, "kept": rep["kept"], "invalid": rep["invalid"],
            "pass": bool(ok_share and ok_sp and ok_n)}


def stake_weight(tag: str, share: float = 0.5, weights=(1.0, 0.5)) -> float:
    wa, wd = weights
    return (wa if str(tag).endswith(":agree") else wd) / (share * wa + (1 - share) * wd)


def stake_checks(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    tags = [t.get("enter_tag") or "" for t in trades]
    agree = np.array([s.endswith(":agree") for s in tags])
    stake = np.array([float(t["stake_amount"]) for t in trades])
    intended = np.array([PROPOSED_STAKE * stake_weight(s) for s in tags])
    short = stake < intended * (1 - 1e-6)
    return {"trades": len(trades), "agree_share": round(float(agree.mean()), 4),
            "mean_stake": round(float(stake.mean()), 2), "width_book_stake": PROPOSED_STAKE,
            "short_stakes": int(short.sum()), "short_share": round(float(short.mean()), 4),
            "flag": bool(short.mean() > SHORT_FLAG)}


def _utc(x):
    x = pd.Timestamp(x)
    return x.tz_localize("UTC") if x.tz is None else x.tz_convert("UTC")


def _table_index(t: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(t["date"])
    d = d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")
    return t.assign(date=d).set_index(["pair", "date"]).sort_index()


def entry_and_tag_checks(trades, table, cut, threshold, bars_for) -> dict:
    viol = mism = 0
    for t in trades:
        _, dates = bars_for(t["pair"])
        e = int(dates.get_loc(pd.Timestamp(t["open_date"])))
        key = (t["pair"], _utc(dates[e - 1]))
        if key not in table.index:
            viol += 1
            continue
        r = table.loc[key]
        ok = (r["guard_metric"] < 0.3) and (r["bb_width"] > 0.035) and (r["bb_width"] >= cut)
        viol += not ok
        agree = bool(r["buy_argmax"]) and float(r["p_buy"]) > threshold
        mism += agree != str(t.get("enter_tag") or "").endswith(":agree")
    return {"entry_violations": int(viol), "tag_mismatches": int(mism)}


def exit_checks(trades, table) -> dict:
    rvol_min, guard_min = EXIT_GATE
    model_exits = sum(str(t.get("exit_reason") or "").startswith("model_exit") for t in trades)
    gate_trades = 0
    for t in trades:
        try:
            w = table.loc[t["pair"]].loc[_utc(t["open_date"]):_utc(t["close_date"])]
        except KeyError:
            continue
        gate_trades += bool(((w["rvol"] > rvol_min) & (w["guard_metric"] > guard_min)).any())
    return {"model_exits": int(model_exits), "trades_with_gate_bars": int(gate_trades),
            "liveness_fail": bool(gate_trades > 0 and model_exits == 0)}


def trade_keys(trades: list[dict]) -> set:
    return {(t["pair"], str(t["open_date"]), str(t["close_date"])) for t in trades}


def fold_json_threshold(cls: str) -> float | None:
    p = STRAT / "NNNC" / f"{cls}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("params", {}).get("buy", {}).get("prediction_threshold")


def _location_dir(plan: dict, f: dict) -> Path:
    from research.candidate_training.train import _location

    return Path(_location(f, plan["base_strategy"], plan["config"], 1).path).parent


def main() -> int:
    from evaluation.fingerprint import load_strategy_class
    from evaluation.pool import arm_record
    from research.exit_coupling import feasibility as F

    out = {"label": "033 S0: label + instrument checks (no P&L read)", "models": {}}
    for m in ("ARO", "GBR"):
        sz = json.loads((FOLDS_DIR / f"plan_wr_{m.lower()}_sz.json").read_text())
        szx = json.loads((FOLDS_DIR / f"plan_wr_{m.lower()}_szx.json").read_text())
        wplan = json.loads((FOLDS_DIR / f"plan_w_{m.lower()}.json").read_text())
        cuts = load_strategy_class(f"NNNC_W_{m}").WIDTH_CUTS
        thr = sz["thresholds"]
        res = {"folds": {}, "arms": {"exit_off": {}, "exit_on": {}}}
        hash_diff = []
        for f, fx, fw in zip(sz["folds"], szx["folds"], wplan["folds"]):
            fid = f["fold_id"]
            d = _location_dir(sz, f)
            rep = json.loads((d / "width_relative.json").read_text()) \
                if (d / "width_relative.json").is_file() else None
            rf = json.loads((d / "row_filter.json").read_text()) \
                if (d / "row_filter.json").is_file() else None
            table = _table_index(pd.read_parquet(SIG / f"WR_{m}_{fid}.parquet"))
            cut = float(cuts[f["train_timerange"]])
            fold = {"label": label_checks(rep, rf), "threshold_plan": thr[fid]}
            from evaluation.bootstrap import _trades

            w_keys = trade_keys(_trades(arm_record(fw["arm_tag"], fid.upper())) or [])
            for arm, ff in (("exit_off", f), ("exit_on", fx)):
                r = arm_record(ff["arm_tag"], fid.upper())
                trades = _trades(r) or []
                jt = fold_json_threshold(ff["strategy_class"])
                a = {"n_trades": len(trades),
                     "threshold_loaded_ok": jt is not None
                     and round(float(jt), 4) == round(float(thr[fid]), 4)}
                a |= entry_and_tag_checks(trades, table, cut, float(thr[fid]), F.bars_for)
                a |= {"stakes": stake_checks(trades)}
                if arm == "exit_off":
                    keys = trade_keys(trades)
                    a["trades_equal_width_book"] = keys == w_keys
                    a["trades_only_arm"] = len(keys - w_keys)
                    a["trades_only_width"] = len(w_keys - keys)
                    if keys != w_keys:
                        hash_diff.append(fid)
                else:
                    a |= exit_checks(trades, table)
                fold[arm] = a
            res["folds"][fid] = fold
        F_ = res["folds"].values()
        common = {
            "label_pass": all(x["label"]["pass"] for x in F_),
            "zero_trade_folds": None}
        for arm in ("exit_off", "exit_on"):
            s = {
                "entry_violations": sum(x[arm]["entry_violations"] for x in F_),
                "tag_mismatches": sum(x[arm]["tag_mismatches"] for x in F_),
                "thresholds_loaded_ok": all(x[arm]["threshold_loaded_ok"] for x in F_),
                "zero_trade_folds": [k for k, x in res["folds"].items() if x[arm]["n_trades"] == 0],
                "stake_flag_folds": [k for k, x in res["folds"].items()
                                     if x[arm]["stakes"].get("flag")]}
            if arm == "exit_off":
                s["hash_diff_folds"] = hash_diff
                s["hash_ok"] = len(hash_diff) <= HASH_DIFF_MAX
            else:
                s["liveness_fail_folds"] = [k for k, x in res["folds"].items()
                                            if x[arm]["liveness_fail"]]
            s["S0_pass"] = bool(common["label_pass"] and s["entry_violations"] == 0
                                and s["tag_mismatches"] == 0 and s["thresholds_loaded_ok"]
                                and s.get("hash_ok", True) and not s.get("liveness_fail_folds"))
            s["S0_inconclusive"] = bool(s["zero_trade_folds"])
            res["arms"][arm] = s
        res["label_pass"] = common["label_pass"]
        out["models"][m] = res
        print(m, json.dumps({"label_pass": res["label_pass"], **res["arms"]}), flush=True)
    (HERE / "instrument.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
