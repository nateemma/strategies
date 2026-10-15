"""035 T008: end-to-end truncation test of every input (FR-005; research R12).

At each of N shared cut times, every source of every pair is truncated at the cut's close and the
whole cache is rebuilt once (market state needs every pair). Every input of every pair at the cut
bar must equal the full-history build (atol 1e-12, NaN = NaN). A planted centred variant of the
taker share must FAIL, proving the harness can see lookahead.

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.causality [--cuts 200]
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from research.new_inputs import build as B
from research.new_inputs import inputs as I


HERE = Path(__file__).resolve().parent
ATOL = 1e-12
START = pd.Timestamp("2021-03-01", tz="UTC")
END = pd.Timestamp("2026-09-07", tz="UTC")


def cut_times(bundle: dict, n: int, start=None, end=None) -> list[pd.Timestamp]:
    every = set().union(*[set(f["date"]) for f in bundle["ohlcv"].values()])
    dates = pd.DatetimeIndex(sorted(every))
    lo = start if start is not None else dates[0]
    hi = end if end is not None else dates[-1]
    d = dates[(dates >= lo) & (dates <= hi)]
    idx = np.unique(np.linspace(0, len(d) - 1, n).round().astype(int))
    return list(d[idx])


def compare(full: dict, part: dict, cut: pd.Timestamp) -> list[dict]:
    bad = []
    for p, df in part.items():
        row = df[df["date"] == cut]
        if row.empty:
            continue
        ref = full[p][full[p]["date"] == cut]
        for col in I.all_columns():
            a, b = float(row[col].iloc[0]), float(ref[col].iloc[0])
            if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) <= ATOL):
                bad.append({"cut": str(cut), "pair": p, "column": col, "truncated": a, "full": b})
    return bad


def run(bundle: dict, n: int, start=None, end=None) -> dict:
    full = B.build_all(bundle)
    out = {"cuts": 0, "checked": 0, "mismatches": []}
    for cut in cut_times(bundle, n, start, end):
        part = B.build_all(bundle, cut)
        out["cuts"] += 1
        out["checked"] += sum(int((df["date"] == cut).any()) for df in part.values())
        out["mismatches"] += compare(full, part, cut)
    out["pass"] = not out["mismatches"]
    return out


@contextlib.contextmanager
def planted_centred_taker_share():
    """Lookahead on purpose: a centred 4-bar window reads the next bars."""
    orig = B.taker_share

    def centred(k, n=4):
        v = k["volume"].rolling(n, min_periods=n, center=True).sum()
        tb = k["taker_buy_base"].rolling(n, min_periods=n, center=True).sum()
        return (tb / v.where(v > 0)).clip(0, 1)

    B.taker_share = centred
    try:
        yield
    finally:
        B.taker_share = orig


def per_input(res: dict) -> dict:
    bad = {}
    for m in res["mismatches"]:
        bad.setdefault(m["column"], 0)
        bad[m["column"]] += 1
    return {i.tag: {"pass": not any(bad.get(c) for c in I.columns(i.tag)),
                    "mismatches": sum(bad.get(c, 0) for c in I.columns(i.tag))} for i in I.TABLE}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, default=200)
    a = ap.parse_args(argv)
    bundle = B.load_bundle()
    res = run(bundle, a.cuts, START, END)
    with planted_centred_taker_share():
        planted = run(bundle, 10, START, END)
    out = {"label": "035 S0 causality (truncation, shared cuts)", "table_sha": I.TABLE_SHA,
           "code_sha": B.code_sha(), "cuts": res["cuts"], "pair_cut_checks": res["checked"],
           "inputs": per_input(res), "pass": res["pass"],
           "planted_control": {"cuts": planted["cuts"], "fails": not planted["pass"],
                               "mismatches": len(planted["mismatches"])},
           "mismatch_sample": res["mismatches"][:50]}
    (HERE / "causality.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("cuts", "pair_cut_checks", "pass", "planted_control")}))
    return 0 if res["pass"] and out["planted_control"]["fails"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
