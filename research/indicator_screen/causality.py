"""037 column lookahead test (research R12, FR-004): every harvested column, truncated.

Reuses research.outcomes.causality's era slices (SOL/LTC/AAVE x 2021-10, 2023-10, 2025-07) and cut
scheme (16 cuts, 200-bar warm-up). For each slice the full harvest is compared bar by bar with the
harvest of every prefix ending at a cut: a causal column cannot change when later bars are removed.
NaN == NaN counts as equal; values must match within 1e-9 relative. A mismatch is a FAIL with its
first mismatch recorded; the harvest then drops the column (reason `lookahead`).

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.causality
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H

RTOL = 1e-9


def _mismatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=RTOL, atol=0.0)
    return ~(both_nan | close)


def check_columns(harvest, slices, cuts: int, warmup: int) -> tuple[dict, dict]:
    """`harvest(ohlcv) -> frame`; `slices` = [(label, ohlcv with a date column)]. Returns
    ({column: verdict row}, {"checked": [...]})."""
    res: dict[str, dict] = {}
    checked = []
    for label, ohlcv in slices:
        checked.append(label)
        full = harvest(ohlcv)
        n = len(full)
        cols = [c for c in full.columns if c not in ("date", *H.OHLCV)]
        for c in cols:
            res.setdefault(c, {"verdict": "PASS", "first_mismatch": None, "comparisons": 0})
        for cut in np.linspace(warmup + 50, n - 1, cuts).astype(int):
            prefix = harvest(ohlcv.iloc[: cut + 1].reset_index(drop=True))
            for c in cols:
                r = res[c]
                if r["verdict"] != "PASS":
                    continue
                if c not in prefix.columns:      # the class raised on a short prefix: nothing to compare
                    r.setdefault("absent_cuts", 0)
                    r["absent_cuts"] += 1
                    continue
                a = full[c].to_numpy()[warmup: cut + 1]
                b = prefix[c].to_numpy()[warmup: cut + 1]
                r["comparisons"] += 1
                bad = _mismatch(a, b)
                if bad.any():
                    j = int(np.flatnonzero(bad)[0]) + warmup
                    r["verdict"] = "FAIL"
                    r["first_mismatch"] = {
                        "slice": label, "cut_bar": int(cut), "bar": j, "bars_before_cut": int(cut - j),
                        "date": str(ohlcv["date"].iloc[j]),
                        "full": float(full[c].iloc[j]), "truncated": float(prefix[c].iloc[j])}
    for r in res.values():
        if r["verdict"] == "PASS" and r["comparisons"] == 0:
            r["verdict"] = "UNSCOREABLE"
    return res, {"checked": checked}


def real_slices(pairs, eras):
    from research.outcomes import causality as OC

    out, skipped = [], []
    for p in pairs:
        for e in eras:
            sl = OC._load_slice(p, e)
            if sl is None:
                skipped.append(f"{p}@{e}")
                continue
            out.append((f"{p}@{e}", sl.reset_index()[["date", *H.OHLCV]]))
    return out, skipped


def main(argv=None) -> int:
    from research.outcomes import causality as OC

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=",".join(OC.DEFAULT_PAIRS))
    a = ap.parse_args(argv)
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    H._quiet()
    cfg = H.base_config()
    scope, strats = H.load_scope(cfg)
    aro = H.load_aro()
    slices, skipped = real_slices(a.pairs.split(","), OC.DEFAULT_ERAS)
    t0 = time.perf_counter()

    def harvest(ohlcv):
        return H.harvest_frame(ohlcv, strats, aro, "SOL/USDT", keep_defects=True)[0]

    res, meta = check_columns(harvest, slices, OC.DEFAULT_CUTS, OC.DEFAULT_WARMUP)
    fails = sorted(c for c, r in res.items() if r["verdict"] == "FAIL")
    H.CAUSALITY.write_text(json.dumps({
        "test": "037 truncation test (research R12): full harvest vs prefix harvest at each cut",
        "slices": {"checked": meta["checked"], "skipped": skipped},
        "cuts": OC.DEFAULT_CUTS, "warmup": OC.DEFAULT_WARMUP, "rtol": RTOL,
        "n_columns": len(res), "n_fail": len(fails), "fails": fails,
        "columns": res}, indent=1, default=str))
    print(f"{len(res)} columns, {len(fails)} FAIL, {time.perf_counter() - t0:.0f}s")
    for c in fails:
        print(f"  FAIL {c}: {res[c]['first_mismatch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
