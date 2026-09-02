"""Phase A - emit market-phase labels at all three levels (T010).

Contract: specs/001-market-phase-entry-gate/contracts/study-cli.md

Writes, under regime/labels/:
    asset_<PAIR>.feather   one column per asset-level instrument
    market.feather         one column per market-level instrument
    btc.feather            one column per BTC-level instrument
and regime/instrument_catalogue.csv.

DEVIATION FROM CONTRACT, deliberate: the contract said "one label artifact per
instrument per pair", which would be 6 x 77 = 462 files carrying the same
information. Grouping by level and pair gives 79 files, the same content, and
matches how Phase B and Phase C actually read it (per pair, all instruments at
once). Recorded here rather than silently.

Labels are stored as pandas categoricals so the files stay small; the label
files are regenerable and are gitignored.

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_a_classify.py --level all
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from regime import instruments as I
from regime.regimedata import (UNKNOWN, TRADED_PAIRS, available_pairs, close_panel,
                               coverage, coverage_ok, load_ohlcv, liquid_pairs,
                               window_slice, WINDOWS)

OUT_DEFAULT = Path(__file__).parent / "labels"
CATALOGUE_CSV = Path(__file__).parent / "instrument_catalogue.csv"


def classify_btc(instruments) -> pd.DataFrame:
    btc = load_ohlcv("BTC")
    out = {}
    for inst in instruments:
        out[inst.name] = inst.fn(btc).astype("category")
    return pd.DataFrame(out, index=btc.index)


def classify_market(instruments) -> pd.DataFrame:
    panel = close_panel()
    ok = coverage_ok(panel)
    out = {"coverage_n": coverage(panel).astype("int16")}
    for inst in instruments:
        out[inst.name] = inst.fn(panel, ok).astype("category")
    return pd.DataFrame(out, index=panel.index)


def classify_asset(instruments, pair: str) -> pd.DataFrame:
    d = load_ohlcv(pair)
    out = {}
    for inst in instruments:
        out[inst.name] = inst.fn(d).astype("category")
    return pd.DataFrame(out, index=d.index)


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_feather(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="all", choices=["asset", "market", "btc", "all"])
    ap.add_argument("--instruments", default="all",
                    help="comma-separated instrument names, or 'all'")
    ap.add_argument("--pairs", default="all",
                    help="comma-separated pairs (asset level), or 'all' for the TRADED universe")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    I.validate_catalogue()
    wanted = None if args.instruments == "all" else set(args.instruments.split(","))
    cat = [i for i in I.CATALOGUE if wanted is None or i.name in wanted]
    if not cat:
        raise SystemExit(f"no instruments matched {args.instruments!r}")

    levels = ["btc", "market", "asset"] if args.level == "all" else [args.level]
    summary = []

    for level in levels:
        insts = [i for i in cat if i.level == level]
        if not insts:
            continue
        t0 = time.time()

        if level == "btc":
            df = classify_btc(insts)
            write(df, args.out / "btc.feather")
            targets = ["BTC"]
        elif level == "market":
            df = classify_market(insts)
            write(df, args.out / "market.feather")
            targets = ["MARKET"]
        else:
            pairs = TRADED_PAIRS if args.pairs == "all" else args.pairs.split(",")
            # Asset level is summarised across EVERY pair, not just the first.
            # A pair listed part-way through a window is legitimately UNKNOWN
            # before it had data and during its estimator warmup - that is not a
            # defect. "Eligible" therefore means AFTER the instrument produced
            # its first valid label for that pair. That definition is exact and
            # needs no per-instrument warmup arithmetic: counting from the series
            # start was too permissive, since VR needs window+q+1 bars, not
            # window (it flagged a handful of SUI bars in W3 as defects).
            acc = {i.name: {"unknown": 0, "eligible": 0, "states": set(),
                            **{f"w_{w}": 0 for w in WINDOWS},
                            **{f"e_{w}": 0 for w in WINDOWS}} for i in insts}
            for n, p in enumerate(pairs, 1):
                dfp = classify_asset(insts, p)
                write(dfp, args.out / f"asset_{p}.feather")
                for inst in insts:
                    s = dfp[inst.name]
                    valid = s[s != UNKNOWN]
                    first_valid = valid.index.min() if len(valid) else None
                    eligible = (pd.Series(s.index, index=s.index) > first_valid
                                if first_valid is not None
                                else pd.Series(False, index=s.index))
                    a = acc[inst.name]
                    a["states"].update(map(str, s.cat.categories))
                    a["unknown"] += int(((s == UNKNOWN) & eligible).sum())
                    a["eligible"] += int(eligible.sum())
                    for w in WINDOWS:
                        sw, ew = window_slice(s, w), window_slice(eligible, w)
                        a[f"w_{w}"] += int(((sw == UNKNOWN) & ew).sum())
                        a[f"e_{w}"] += int(ew.sum())
                if n % 20 == 0 or n == len(pairs):
                    print(f"    asset: {n}/{len(pairs)} pairs", flush=True)
            targets = pairs

        print(f"  {level:<7} {len(insts)} instruments x {len(targets)} target(s) "
              f"in {time.time()-t0:.1f}s")

        for inst in insts:
            if level == "asset":
                a = acc[inst.name]
                pct = lambda num, den: float(num / den * 100) if den else 0.0
                row = {"instrument": inst.name, "level": level,
                       "n_states": len(a["states"]),
                       "unknown_pct": pct(a["unknown"], a["eligible"])}
                for w in WINDOWS:
                    row[f"unknown_{w}_pct"] = pct(a[f"w_{w}"], a[f"e_{w}"])
            else:
                s = df[inst.name]
                row = {"instrument": inst.name, "level": level,
                       "n_states": int(s.nunique()),
                       "unknown_pct": float((s == UNKNOWN).mean() * 100)}
                for w in WINDOWS:
                    row[f"unknown_{w}_pct"] = float((window_slice(s, w) == UNKNOWN).mean() * 100)
            summary.append(row)

    I.to_frame(cat).to_csv(CATALOGUE_CSV, index=False)
    print(f"\nwrote {CATALOGUE_CSV}")

    sm = pd.DataFrame(summary)
    print("\nUNKNOWN share per instrument (inside a window it is a DEFECT, not a state):")
    print(sm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    win_cols = [f"unknown_{w}_pct" for w in WINDOWS]
    defects = sm[(sm[win_cols] > 0).any(axis=1)]
    if len(defects):
        print(f"\n*** {len(defects)} instrument(s) emit UNKNOWN inside an evaluation window ***")
        print(defects[["instrument"] + win_cols].to_string(index=False))
        return 1
    print("\nOK - no instrument emits UNKNOWN inside W1/W2/W3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
