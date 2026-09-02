"""Phase A GATE - end-to-end lookahead validation (T011).

Contract: specs/001-market-phase-entry-gate/contracts/study-cli.md

Recomputes every instrument over history TRUNCATED at a cut point and asserts
the labels are identical to the full-history labels over the overlap. Exit code
2 if any instrument differs: it MUST NOT proceed to Phase B (FR-007, SC-001).

This is the measurement Principle V requires. Reasoning line-by-line about
causality is not evidence - centred windows, ffill reaching backwards, and
cross-sectional operations over a panel containing future rows all pass casual
inspection and all leak. freqtrade's own lookahead-analysis is not a substitute
either: it does not inspect study code that runs outside a strategy.

THE UNIVERSE IS HELD FIXED across both sides. liquid_pairs() selects pairs
using volume over all three windows, so recomputing it under truncation would
change the panel's COLUMNS and the test would measure universe selection rather
than per-bar causality. Note that this a-priori universe choice is itself a
survivorship decision - see the note in README.md - but it is a study design
parameter, not a per-bar leak, and it is what this test deliberately holds
constant.

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_a_lookahead.py
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from regime import instruments as I
from regime.regimedata import (TRADED_PAIRS, close_panel, coverage_ok,
                               liquid_pairs, load_ohlcv)

OUT_CSV = Path(__file__).parent / "lookahead_validation.csv"

# Cut points spread across the data, one inside each evaluation window.
DEFAULT_CUTS = ["20230630", "20241231", "20260331"]


def _btc_labels(insts, cut=None):
    d = load_ohlcv("BTC")
    if cut is not None:
        d = d.loc[d.index <= cut]
    return {i.name: i.fn(d) for i in insts}


def _market_labels(insts, pairs, cut=None):
    panel = close_panel(pairs)
    if cut is not None:
        panel = panel.loc[panel.index <= cut]
    ok = coverage_ok(panel)
    return {i.name: i.fn(panel, ok) for i in insts}


def _asset_labels(insts, pair, cut=None):
    d = load_ohlcv(pair)
    if cut is not None:
        d = d.loc[d.index <= cut]
    return {i.name: i.fn(d) for i in insts}


def compare(full: pd.Series, trunc: pd.Series, cut: pd.Timestamp) -> tuple:
    """Compare labels over the overlapping region [start, cut]."""
    f = full.loc[full.index <= cut]
    t = trunc.loc[trunc.index <= cut]
    idx = f.index.intersection(t.index)
    f, t = f.loc[idx].astype(str), t.loc[idx].astype(str)
    diff = f != t
    n_diff = int(diff.sum())
    first = str(f.index[diff.argmax()]) if n_diff else ""
    return len(idx), n_diff, first


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuts", default=",".join(DEFAULT_CUTS),
                    help="comma-separated YYYYMMDD truncation points")
    ap.add_argument("--instruments", default="all")
    args = ap.parse_args()

    I.validate_catalogue()
    wanted = None if args.instruments == "all" else set(args.instruments.split(","))
    cat = [i for i in I.CATALOGUE if wanted is None or i.name in wanted]
    cuts = [pd.Timestamp(c, tz="UTC") for c in args.cuts.split(",")]

    # Fixed universe for both sides of every comparison.
    pairs_liquid = liquid_pairs()
    rows = []

    for cut in cuts:
        t0 = time.time()

        for level in ("btc", "market", "asset"):
            insts = [i for i in cat if i.level == level]
            if not insts:
                continue

            if level == "btc":
                full = _btc_labels(insts)
                trunc = _btc_labels(insts, cut)
                for i in insts:
                    n, d, first = compare(full[i.name], trunc[i.name], cut)
                    rows.append({"instrument": i.name, "level": level, "target": "BTC",
                                 "cut_date": str(cut.date()), "n_bars_compared": n,
                                 "n_differing": d, "first_diff_date": first})

            elif level == "market":
                full = _market_labels(insts, pairs_liquid)
                trunc = _market_labels(insts, pairs_liquid, cut)
                for i in insts:
                    n, d, first = compare(full[i.name], trunc[i.name], cut)
                    rows.append({"instrument": i.name, "level": level, "target": "MARKET",
                                 "cut_date": str(cut.date()), "n_bars_compared": n,
                                 "n_differing": d, "first_diff_date": first})

            else:
                for p in TRADED_PAIRS:
                    full = _asset_labels(insts, p)
                    trunc = _asset_labels(insts, p, cut)
                    for i in insts:
                        n, d, first = compare(full[i.name], trunc[i.name], cut)
                        rows.append({"instrument": i.name, "level": level, "target": p,
                                     "cut_date": str(cut.date()), "n_bars_compared": n,
                                     "n_differing": d, "first_diff_date": first})

        print(f"  cut {cut.date()} checked in {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} comparisons)")

    # Per-instrument roll-up: an instrument passes only if EVERY target and
    # EVERY cut point matched.
    roll = df.groupby("instrument").agg(
        comparisons=("n_differing", "size"),
        total_bars=("n_bars_compared", "sum"),
        n_differing=("n_differing", "sum"),
    ).reset_index()
    roll["verdict"] = roll.n_differing.map(lambda v: "PASS" if v == 0 else "*** LEAK ***")
    print("\nper-instrument roll-up:")
    print(roll.to_string(index=False))

    failed = roll[roll.n_differing > 0]
    if len(failed):
        print(f"\n*** {len(failed)} instrument(s) LEAK - they must NOT proceed to Phase B ***")
        bad = df[df.n_differing > 0]
        print(bad[["instrument", "target", "cut_date", "n_differing",
                   "first_diff_date"]].head(20).to_string(index=False))
        print("\nlocalising hints (quickstart.md Scenario 2):")
        print("  divergence at the END of the overlap      -> centred rolling window")
        print("  divergence propagating BACKWARDS from cut -> ffill/bfill across the boundary")
        print("  market level only                         -> cross-sectional op over future rows")
        return 2

    print(f"\nOK - {int(roll.total_bars.sum()):,} label-bars compared, zero differences.")
    print("All instruments are causal end-to-end and may proceed to Phase B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
