"""031 T021: do the backtested adds match the screen's DCA adds on the same signals?

Matches `hp_aro_dca`'s backtest adds (trades with a second filled entry order: pair, add fill
date) against the screen's DCA:all:1h adds (`adds_<M>.json`, from `screen.py --dump-adds`) on
the same folds. The gate (FR-009, SC-002) is precision >= 95% (backtest adds that are screen
adds) with zero `row_misaligned`; recall, exits on the add candle, and the screen-vs-arm pp gap
are reported, not gated.

    PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.fidelity hp_aro_dca \\
        --model ARO --control wf_aro_g24 --folds f00,f01,f02,f03,f04,f05,f06,f07

Exit 0 pass, 2 below the gate (or misaligned rows), 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from evaluation import pool as _pool  # noqa: E402,F401 -- pins bootstrap.BT_RESULTS
from evaluation.bootstrap import _scale, _trades  # noqa: E402


HERE = Path(__file__).resolve().parent
ARMS = STRAT / "regime" / "arms"
GATE = 0.95


def _rec(tag: str, fid: str) -> dict:
    return json.loads((ARMS / f"{tag}_{fid}_{fid.upper()}.json").read_text())


def backtest_adds(rec: dict) -> list[dict]:
    rows = _trades(rec) or []
    scale = _scale(rec, rows)
    out = []
    for t in rows:
        fills = sorted(o["order_filled_timestamp"] for o in t.get("orders") or []
                       if o.get("ft_is_entry") and o.get("order_filled_timestamp"))
        for ts in fills[1:]:
            add = pd.Timestamp(ts, unit="ms", tz="UTC")
            close = pd.Timestamp(t["close_date"])
            close = close.tz_localize("UTC") if close.tz is None else close
            out.append({"pair": t["pair"], "add_date": add, "close_date": close,
                        "exit_on_add_candle": bool(close == add),
                        "pp": float(t["profit_abs"]) * scale})
    return out


def run(arm: str, model: str, control: str, folds: list[str], adds_cond: str = "all") -> dict:
    scr = json.loads((HERE / (f"adds_{model}.json" if adds_cond == "all"
                              else f"adds_{model}_{adds_cond}.json")).read_text())
    screen = {}
    for a in scr["adds"]:
        screen.setdefault(a["fold"], set()).add((a["pair"], pd.Timestamp(a["add_date"])))
    per, tot = {}, {"matched": 0, "backtest": 0, "screen": 0, "misaligned": 0,
                    "exit_on_add": 0, "exit_on_add_pp": 0.0}
    for f in folds:
        rec, ctl = _rec(arm, f), _rec(control, f)
        bt = backtest_adds(rec)
        keys = {(a["pair"], a["add_date"]) for a in bt}
        sc = screen.get(f, set())
        hp = rec.get("hp_counters") or {}
        row = {"backtest_adds": len(keys), "screen_adds": len(sc), "matched": len(keys & sc),
               "exit_on_add_candle": sum(a["exit_on_add_candle"] for a in bt),
               "exit_on_add_candle_pp": round(sum(a["pp"] for a in bt if a["exit_on_add_candle"]), 4),
               "arm_delta_pp": round((rec.get("wallet_profit_pct") or 0.0)
                                     - (ctl.get("wallet_profit_pct") or 0.0), 4),
               "screen_delta_pp": (scr.get("fold_delta_pp") or {}).get(f),
               "row_misaligned": int(hp.get("refused_row_misaligned", 0)),
               "hp_counters": hp,
               "unmatched_backtest": sorted(f"{p} {d}" for p, d in keys - sc)[:20],
               "unmatched_screen": sorted(f"{p} {d}" for p, d in sc - keys)[:20]}
        if row["screen_delta_pp"] is not None:
            row["gap_pp"] = round(row["arm_delta_pp"] - row["screen_delta_pp"], 4)
        per[f] = row
        tot["matched"] += row["matched"]
        tot["backtest"] += row["backtest_adds"]
        tot["screen"] += row["screen_adds"]
        tot["misaligned"] += row["row_misaligned"]
        tot["exit_on_add"] += row["exit_on_add_candle"]
        tot["exit_on_add_pp"] += row["exit_on_add_candle_pp"]
    precision = tot["matched"] / tot["backtest"] if tot["backtest"] else 0.0
    recall = tot["matched"] / tot["screen"] if tot["screen"] else 0.0
    passed = precision >= GATE and tot["misaligned"] == 0 and tot["backtest"] > 0
    return {"arm": arm, "model": model, "control": control, "folds": folds,
            "gate": {"precision_min": GATE, "row_misaligned_max": 0},
            "precision": round(precision, 4), "recall": round(recall, 4),
            "totals": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tot.items()},
            "arm_delta_pp": round(sum(r["arm_delta_pp"] for r in per.values()), 4),
            "screen_delta_pp": round(sum(r["screen_delta_pp"] or 0.0 for r in per.values()), 4),
            "verdict": "PASS" if passed else "FAIL", "per_fold": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arm")
    ap.add_argument("--model", default="ARO")
    ap.add_argument("--control", default="wf_aro_g24")
    ap.add_argument("--folds", default="f00,f01,f02,f03,f04,f05,f06,f07")
    ap.add_argument("--cond", default="all")
    a = ap.parse_args(argv)
    out = run(a.arm, a.model, a.control, a.folds.split(","), a.cond)
    (HERE / ("fidelity.json" if a.cond == "all" else f"fidelity_{a.cond}.json")).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per_fold"}, indent=1, default=str))
    for f, r in out["per_fold"].items():
        print(f"{f}: bt {r['backtest_adds']:3d} scr {r['screen_adds']:3d} match {r['matched']:3d} "
              f"exit@add {r['exit_on_add_candle']:2d} arm {r['arm_delta_pp']:+7.2f} "
              f"screen {r['screen_delta_pp'] if r['screen_delta_pp'] is not None else float('nan'):+7.2f}")
    return 0 if out["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
