"""036 T025: the live record against the cost model, every 4 weeks (research R13; FR-016).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.model_check

For each recorded order and exit with a depth snapshot, it compares the cost the model predicts
(from participation at that candle) with the cost of walking that snapshot for the granted stake.
It appends a check with the median signed error per tier. The bias flag is set when |median|
exceeds the pre-registered tolerance in the same tier on two consecutive checks. While the flag is
open, capacity.py offers nothing for adoption. Writes research/live_pnl/model_check.json.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "model_check.json"


def tier_errors(rows: pd.DataFrame) -> dict:
    """rows: tier, predicted, walked -> {tier: median signed error (bps)}."""
    if rows.empty:
        return {}
    e = (rows.predicted - rows.walked).groupby(rows.tier.astype(str)).median()
    return {k: round(float(v), 3) for k, v in e.items()}


def entry_stake(orders: pd.DataFrame, pair: str, exit_time: str) -> float | None:
    """The granted stake of the pair's last non-rejected order before an exit (exits carry no
    stake; one trade per pair means the last entry is the trade's)."""
    if orders.empty:
        return None
    o = orders[(orders.pair == pair) & (orders.order_time <= exit_time)]
    if "rejected" in o:
        o = o[~o.rejected.astype(bool)]
    return None if o.empty else o.sort_values("order_time").granted_stake.iloc[-1]


def update(state: dict, check: dict, tol: float) -> dict:
    checks = state.get("checks", []) + [check]
    flag = False
    if len(checks) >= 2:
        a, b = checks[-2]["median_signed_error_bps"], checks[-1]["median_signed_error_bps"]
        flag = any(abs(a.get(t, 0)) > tol and abs(b.get(t, 0)) > tol
                   and (a[t] > 0) == (b[t] > 0) for t in b)
    return {"label": "036 model check (PROVISIONAL)", "tolerance_bps": tol, "checks": checks,
            "bias_flag": bool(flag or state.get("bias_flag", False) and not check.get("cleared"))}


def main() -> int:
    from research.live_pnl import cost_model as CM
    from research.live_pnl import depth as D
    from research.live_pnl.parity import REC_ROOT, read_jsonl

    model, sha = CM.load()
    tol = CM.TOL_BPS
    recs = [r for r in json.loads((HERE / "live_models.json").read_text()) if r.get("started")]
    depth = {}
    for r in recs:
        for f in sorted((REC_ROOT / r["run_id"]).glob("depth-*.jsonl.gz")):
            for x in gzip.open(f, "rt"):
                d = json.loads(x)
                depth[d["key"]] = d
    rows = []
    vol = {}
    for r in recs:
        orders = read_jsonl("orders", r["run_id"])
        for kind, side in (("orders", "buy"), ("exits", "sell")):
            df = orders if kind == "orders" else read_jsonl("exits", r["run_id"])
            for o in df.itertuples() if not df.empty else []:
                snap = depth.get(o.depth_ref)
                if not snap or snap["depth_missing"] or o.pair in model["excluded_pairs"]:
                    continue
                stake = entry_stake(orders, o.pair, o.time) if kind == "exits" else o.granted_stake
                if stake is None or not stake > 0:
                    continue
                if o.pair not in vol:
                    vol[o.pair] = CM.qv24(o.pair)
                t = pd.Timestamp(snap["time"]).floor("15min")
                qv = float(vol[o.pair].asof(t))
                lv = snap["asks"] if side == "buy" else snap["bids"]
                walked, ext = D.walk_cost_bps(lv, snap["mid"], stake, side)
                if ext:
                    continue
                rows.append({"tier": model["tiers"][o.pair], "walked": walked,
                             "predicted": CM.cost_bps(model, o.pair, stake, qv, side)})
    state = json.loads(OUT.read_text()) if OUT.is_file() else {}
    check = {"date": date.today().isoformat(), "n": len(rows), "cost_model_sha": sha,
             "median_signed_error_bps": tier_errors(pd.DataFrame(rows))}
    out = update(state, check, tol)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(check), "bias_flag", out["bias_flag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
