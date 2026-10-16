"""036 T023: re-state an arm's backtest trades at estimated execution cost (research R6; FR-011).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.restate --arm wf_aro_g24 [--folds f00,f01]

For every trade in each fold's arm record, the cost model prices the entry (buy) and exit (sell)
at the trade's stake, using the pair's rolling 24 h quote volume at the entry and exit candles
(causal). Net = gross profit_abs − stake × (cost_in + cost_out) / 1e4. Trade identity is
unchanged, and compounding is ignored, since the stake is a fixed amount with a per-trade cap.

Writes research/live_pnl/restated_<arm>.json (data-model RestatedResult). The output is an
execution measurement on fold models that inherit the full-history scaler (FR-020).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
FOLDS = STRAT / "evaluation" / "folds"
ARMS = STRAT / "regime" / "arms"
RESULTS = STRAT.parent / "backtest_results"
WALLET = 10000.0
EARLY = tuple(f"f{i:02d}" for i in range(8))


def price_trades(trades: pd.DataFrame, cost_fn, qv_at) -> pd.DataFrame:
    """Add cost_in/cost_out (bps), cost_abs and net_abs. `cost_fn(pair, stake, qv, side)`,
    `qv_at(pair, ts)` -> rolling 24 h quote volume at that candle."""
    t = trades.copy()
    t["cost_in"] = [cost_fn(p, s, qv_at(p, o), "buy")
                    for p, s, o in zip(t.pair, t.stake_amount, t.open_date)]
    t["cost_out"] = [cost_fn(p, s, qv_at(p, c), "sell")
                     for p, s, c in zip(t.pair, t.stake_amount, t.close_date)]
    t["cost_abs"] = t.stake_amount * (t.cost_in + t.cost_out) / 1e4
    t["net_abs"] = t.profit_abs - t.cost_abs
    return t


def summarise(t: pd.DataFrame) -> dict:
    return {"n": int(len(t)), "gross_abs": round(float(t.profit_abs.sum()), 2),
            "cost_abs": round(float(t.cost_abs.sum()), 2), "net_abs": round(float(t.net_abs.sum()), 2),
            "gross_pct": round(100 * float(t.profit_abs.sum()) / WALLET, 3),
            "net_pct": round(100 * float(t.net_abs.sum()) / WALLET, 3)}


def fold_trades(arm_tag: str, fold_id: str) -> pd.DataFrame:
    rec = json.loads((ARMS / f"{arm_tag}_{fold_id.upper()}.json").read_text())
    with zipfile.ZipFile(RESULTS / rec["result_file"]) as z:
        name = [n for n in z.namelist() if n.endswith(".json") and "config" not in n
                and "meta" not in n][0]
        s = list(json.loads(z.read(name))["strategy"].values())[0]
    t = pd.DataFrame(s["trades"])
    t["open_date"] = pd.to_datetime(t.open_date, utc=True)
    t["close_date"] = pd.to_datetime(t.close_date, utc=True)
    return t


def run(arm: str, folds: list[str] | None = None) -> dict:
    from research.live_pnl import cost_model as CM

    model, sha = CM.load()
    plan = json.loads((FOLDS / f"plan_{arm}.json").read_text())
    vol = {}

    def qv_at(pair, ts):
        if pair not in vol:
            vol[pair] = CM.qv24(pair)
        s = vol[pair]
        return float(s.asof(pd.Timestamp(ts).floor("15min") - pd.Timedelta(minutes=15)))

    per_fold, frames = {}, []
    for f in plan["folds"]:
        if folds and f["fold_id"] not in folds:
            continue
        t = fold_trades(f["arm_tag"], f["fold_id"])
        t = t[~t.pair.isin(model["excluded_pairs"])]
        t = price_trades(t, lambda p, s, q, sd: CM.cost_bps(model, p, s, q, sd), qv_at)
        per_fold[f["fold_id"]] = summarise(t)
        frames.append(t.assign(fold=f["fold_id"]))
    allt = pd.concat(frames, ignore_index=True)
    early = allt[allt.fold.isin(EARLY)]
    late = allt[~allt.fold.isin(EARLY)]
    out = {"label": "036 re-statement (PROVISIONAL execution measurement; fold models inherit the "
                    "full-history scaler, FR-020)",
           "arm": arm, "cost_model_sha": sha, "extrapolated": bool(model["extrapolated_history"]),
           "excluded_pairs": model["excluded_pairs"], "per_fold": per_fold,
           "early": summarise(early), "late": summarise(late), "total": summarise(allt)}
    (HERE / f"restated_{arm}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("arm", "early", "late", "total")}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--folds")
    a = ap.parse_args(argv)
    run(a.arm, a.folds.split(",") if a.folds else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
