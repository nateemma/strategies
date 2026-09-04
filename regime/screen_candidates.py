"""Screen labelling candidates WITHOUT training anything (FR-001..FR-003).

Emits one ScreeningRecord per candidate: density, median labelled gain,
stop-clearing share, reward:risk, time-to-peak and stopped-before-peak. Sets
excluded_reason below the density floor and requires_hold_change where the
payoff peaks beyond the forced-exit window.

Contract: specs/003-breakout-tail-labelling/contracts/study-cli.md section 1.
Loads no model. Exit code 0 even when candidates are excluded -- exclusion is a
result, not an error.

    PYTHONPATH=user_data/strategies .venv/bin/python \
        user_data/strategies/regime/screen_candidates.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime.tail.candidates import (CANDIDATES, DENSITY_FLOOR_PCT,
                                    FORCED_EXIT_HOURS, BASIS_BB, validate_all)

DATA = Path("user_data/data/binanceus")
DEFAULT_PAIRS = ("XRP_USDT", "SOL_USDT", "LINK_USDT")
# The operative stop is the ATR band [-6%, -4%]; -5% is its midpoint and the
# value the economics were measured against. NOT self.stoploss, which is inert
# for NN strategies.
STOP_PCT = 5.0
BARS_PER_HOUR = 4


def _populated(pair: str) -> pd.DataFrame:
    from utils.DataframePopulator import DataframePopulator, DatasetType
    df = pd.read_feather(DATA / f"{pair}-15m.feather")
    return DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)


def _condition(df: pd.DataFrame, cand) -> np.ndarray:
    """The candidate's indicator condition, before any forward-looking filter."""
    bw = np.nan_to_num(np.asarray(df.get("bb_width", 0.0), dtype=float))
    if cand.basis == BASIS_BB:
        # bb_position is rescaled to [-1, 1]: +1 = at/above the upper band.
        bp = np.nan_to_num(np.asarray(df.get("bb_position", 0.0), dtype=float))
        return (bp >= 0.60) & (bw > 0.035)
    gm = np.nan_to_num(np.asarray(df.get("guard_metric", 0.0), dtype=float))
    return (gm > 0.2) & (bw > 0.035)


def screen_one(cand, pairs) -> dict:
    H = cand.horizon
    gains, dds, peaks, n_bars, n_lab = [], [], [], 0, 0
    for pair in pairs:
        df = _populated(pair)
        close = np.asarray(df["close"], dtype=float)
        cond = _condition(df, cand)
        n_bars += len(close)
        for t in np.flatnonzero(cond):
            if t + H + 1 >= len(close):
                continue
            fut = close[t + 1: t + H + 1]
            if fut.size == 0:
                continue
            g = (np.nanmax(fut) - close[t]) / close[t] * 100
            if g < cand.min_gain * 100:
                continue
            j = int(np.nanargmax(fut))
            worst = (np.nanmin(fut[: j + 1]) - close[t]) / close[t] * 100
            if cand.mae_bound is not None and worst < -cand.mae_bound * 100:
                continue          # path-aware constraint (FR-006)
            gains.append(g); dds.append(worst); peaks.append(j + 1); n_lab += 1
    if not gains:
        return {"candidate_id": cand.id, "density_pct": 0.0, "error": "no labels"}
    g = np.array(gains); d = np.array(dds); p = np.array(peaks)
    med = float(np.median(g))
    return {
        "candidate_id": cand.id,
        "density_pct": round(n_lab / n_bars * 100, 2),
        "median_gain_pct": round(med, 2),
        "clears_stop_pct": round(float((g >= STOP_PCT).mean() * 100), 1),
        "reward_risk": round(med / STOP_PCT, 2),
        "time_to_peak_h": round(float(np.median(p)) / BARS_PER_HOUR, 1),
        "stopped_first_pct": round(float((d <= -4.0).mean() * 100), 1),
        "n_pairs": len(pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--only-unscreened", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pairs = tuple(args.pairs.split(","))

    validate_all()
    todo = [c for c in CANDIDATES
            if not (args.only_unscreened and c.screened)]

    records = []
    for c in todo:
        if c.screened and not args.only_unscreened:
            rec = {"candidate_id": c.id, "density_pct": c.density_pct,
                   "median_gain_pct": c.median_gain_pct,
                   "clears_stop_pct": c.clears_stop_pct,
                   "reward_risk": c.reward_risk,
                   "time_to_peak_h": c.time_to_peak_h,
                   "stopped_first_pct": c.stopped_first_pct,
                   "n_pairs": 3, "source": "research.md R3/R5 (pre-recorded)"}
        else:
            rec = screen_one(c, pairs)
            rec["source"] = "measured now"
        ttp = rec.get("time_to_peak_h")
        rec["requires_hold_change"] = None if ttp is None else ttp > FORCED_EXIT_HOURS
        rec["excluded_reason"] = c.excluded_reason or (
            f"density {rec['density_pct']}% is below the {DENSITY_FLOOR_PCT}% "
            f"all-Hold collapse floor (FR-002)"
            if rec.get("density_pct", 0) < DENSITY_FLOOR_PCT else "")
        records.append(rec)

    hdr = (f"{'candidate':<22}{'%bars':>8}{'medGain':>9}{'clears':>8}"
           f"{'r:r':>6}{'peak(h)':>9}{'stopped':>9}  status")
    print("SCREENING RECORDS -- no model loaded, nothing trained (FR-003)\n")
    print(hdr)
    for r in records:
        if r.get("error"):
            print(f"{r['candidate_id']:<22}  {r['error']}")
            continue
        status = "EXCLUDED" if r["excluded_reason"] else (
            "admitted (needs hold change)" if r["requires_hold_change"]
            else "admitted")
        # path metrics are None for candidates excluded on the density floor --
        # tail_path is not run for a candidate that cannot be trained anyway
        def _f(v, w, dp=1, suf=""):
            return (f"{v:>{w}.{dp}f}{suf}" if v is not None else f"{'-':>{w}}{' ' * len(suf)}")
        print(f"{r['candidate_id']:<22}{r['density_pct']:>7.2f}%{r['median_gain_pct']:>9.2f}"
              f"{r['clears_stop_pct']:>7.1f}%{r['reward_risk']:>6.2f}"
              f"{_f(r['time_to_peak_h'], 9)}{_f(r['stopped_first_pct'], 8, 1, '%')}  {status}")

    adm = [r for r in records if not r["excluded_reason"] and not r.get("error")]
    best = max((r["reward_risk"] for r in adm), default=0.0)
    print(f"\n  admitted: {len(adm)} of {len(records)}")
    print(f"  best reward:risk among ADMITTED candidates: {best:.2f}")
    if best > 1.0:
        print("  SC-002 SATISFIED — an admitted candidate exceeds reward:risk 1.0")
    else:
        print("  *** SC-002 NOT SATISFIED — no admitted candidate exceeds 1.0.")
        print("      The feature's premise fails at screening; that is the result.")

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
