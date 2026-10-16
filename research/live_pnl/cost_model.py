"""036 T022: fit the execution cost model from recorded depth (research R5; preregistration_cost.json).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.cost_model --fit

The tool refuses while the recording gate is closed, or while preregistration_cost.json is
uncommitted or modified. It writes research/live_pnl/cost_model.json (data-model CostModel), and
`load()` and `cost_bps()` are what re-statement and the stake cap read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.live_pnl import depth as D  # noqa: E402


HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_cost.json"
OUT = HERE / "cost_model.json"
REC_ROOT = STRAT.parent / "data" / "live_nnnc"
OHLCV = STRAT.parent / "data" / "binanceus"
REF_STAKE = 900
HOLDOUT = 0.25
TOL_BPS = 5.0
COVERAGE_FLOOR = 0.80


def refusals(gate_open: bool, prereg_clean: bool) -> list[str]:
    out = []
    if not gate_open:
        out.append("recording gate is closed (research/live_pnl/gate.json)")
    if not prereg_clean:
        out.append("preregistration_cost.json is uncommitted or modified")
    return out


def _prereg_clean() -> bool:
    rel = PREREG.relative_to(STRAT)
    dirty = subprocess.run(["git", "-C", str(STRAT), "status", "--porcelain", "--", str(rel)],
                           capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "-C", str(STRAT), "ls-files", "--", str(rel)],
                             capture_output=True, text=True).stdout.strip()
    return bool(tracked) and not dirty


def qv24(pair: str) -> pd.Series:
    df = pd.read_feather(OHLCV / f"{pair.replace('/', '_')}-15m.feather",
                         columns=["date", "close", "volume"])
    s = (df["close"] * df["volume"]).rolling(96, min_periods=96).sum()
    return pd.Series(s.to_numpy(), index=pd.to_datetime(df["date"], utc=True))


def snapshot_rows(snaps: pd.DataFrame, vol: dict) -> pd.DataFrame:
    """One row per (snapshot, stake, side): cost, participation, depth."""
    rows = []
    for s in snaps.itertuples():
        if s.depth_missing or not s.bids or not s.asks:
            continue
        t = pd.Timestamp(s.time).floor("15min")
        v = vol[s.pair].get(t, np.nan)
        dq = (sum(p * q for p, q in s.bids) + sum(p * q for p, q in s.asks)) / 2
        for stake in D.STAKE_GRID:
            for side, lv in (("buy", s.asks), ("sell", s.bids)):
                c, ext = D.walk_cost_bps(lv, s.mid, stake, side)
                rows.append({"pair": s.pair, "day": str(s.time)[:10], "stake": stake, "side": side,
                             "cost": c, "extrapolated": ext, "p": D.participation(stake, v),
                             "depth_quote": dq})
    return pd.DataFrame(rows)


def fit(rows: pd.DataFrame, coverage: dict) -> dict:
    days = sorted(rows.day.unique())
    n_hold = max(1, math.ceil(len(days) * HOLDOUT))
    hold_days = set(days[-n_hold:])
    fit_r = rows[~rows.day.isin(hold_days) & ~rows.extrapolated]
    hold = rows[rows.day.isin(hold_days) & ~rows.extrapolated & (rows.stake == REF_STAKE)]
    tier = D.tiers(fit_r.groupby("pair").depth_quote.median().to_dict())
    part = {f"{t}|{sd}": D.MonotoneCurve.fit(g.p, g.cost)
            for (t, sd), g in fit_r.assign(tier=fit_r.pair.map(tier)).groupby(["tier", "side"])}
    perp = {f"{p}|{sd}": D.MonotoneCurve.fit(g.stake, g.cost)
            for (p, sd), g in fit_r.groupby(["pair", "side"])}
    hold = hold.assign(tier=hold.pair.map(tier))
    hold["pred_part"] = [float(part[f"{t}|{sd}"](p)) for t, sd, p in zip(hold.tier, hold.side, hold.p)]
    hold["pred_pair"] = [float(perp[f"{p}|{sd}"](s)) for p, sd, s in zip(hold.pair, hold.side, hold.stake)]
    err_part = float((hold.pred_part - hold.cost).abs().median())
    err_pair = float((hold.pred_pair - hold.cost).abs().median())
    choice = "participation" if err_part <= err_pair else "per_pair"
    col = "pred_part" if choice == "participation" else "pred_pair"
    per_pair_err = (hold[col] - hold.cost).abs().groupby(hold.pair).median().round(3).to_dict()
    excluded = sorted({p for p, e in per_pair_err.items() if e > TOL_BPS}
                      | {p for p, c in coverage.items() if c < COVERAGE_FLOOR})
    return {"label": "036 cost model (PROVISIONAL execution measurement)",
            "fitted_on": [days[0], days[-1]], "holdout_days": sorted(hold_days),
            "tiers": tier, "choice": choice, "extrapolated_history": choice == "per_pair",
            "holdout_error_pooled": {"participation": round(err_part, 3), "per_pair": round(err_pair, 3)},
            "holdout_error_by_pair": per_pair_err, "coverage": coverage, "excluded_pairs": excluded,
            "curves": {k: v.to_json() for k, v in part.items()},
            "per_pair": {k: v.to_json() for k, v in perp.items()}}


def load(path: Path = OUT) -> tuple[dict, str]:
    blob = path.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()[:16]


def cost_bps(model: dict, pair: str, stake: float, qv24_quote: float, side: str) -> float:
    if pair in model["excluded_pairs"]:
        raise KeyError(f"{pair} is excluded from the cost model")
    if model["choice"] == "participation":
        t = model["tiers"][pair]
        return float(D.MonotoneCurve.from_json(model["curves"][f"{t}|{side}"])(
            D.participation(stake, qv24_quote)))
    return float(D.MonotoneCurve.from_json(model["per_pair"][f"{pair}|{side}"])(stake))


def main(argv=None) -> int:
    from research.live_pnl import gate
    from research.live_pnl.parity import read_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", required=True)
    ap.parse_args(argv)
    bad = refusals(gate.is_open(), _prereg_clean())
    if bad:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(bad))
    recs = [r for r in json.loads((HERE / "live_models.json").read_text()) if r.get("started")]
    snaps = []
    for r in recs:
        for f in sorted((REC_ROOT / r["run_id"]).glob("depth-*.jsonl.gz")):
            snaps += [json.loads(x) for x in gzip.open(f, "rt")]
    snaps = pd.DataFrame(snaps)
    sched = snaps[snaps.trigger == "schedule"]
    coverage = (1 - sched.groupby("pair").depth_missing.mean()).round(4).to_dict()
    vol = {p: qv24(p) for p in snaps.pair.unique()}
    model = fit(snapshot_rows(snaps, vol), coverage)
    OUT.write_text(json.dumps(model, indent=1) + "\n")
    print(json.dumps({k: model[k] for k in ("choice", "holdout_error_pooled", "excluded_pairs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
