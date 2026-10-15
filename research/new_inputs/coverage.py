"""035 T017: S0 coverage and the live-feasibility record (FR-006, FR-007). No P&L is read.

Per input: the share of defined values on every pair's bars inside each fold's test window, and
on the 412 early width-book trades' signal bars (the coverage S1's survival rule reads); plus its
live source (informational, clarification Q2).

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.coverage
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.new_inputs import inputs as I  # noqa: E402


HERE = Path(__file__).resolve().parent
CACHE = STRAT.parent / "data" / "new_inputs"
PLAN = STRAT / "evaluation" / "folds" / "plan_wf_aro_g24.json"


def fold_windows() -> dict:
    out = {}
    for f in json.loads(PLAN.read_text())["folds"]:
        a, b = f["test_timerange"].split("-")
        out[f["fold_id"]] = (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
    return out


def bar_coverage(cache_dir: Path = CACHE) -> dict:
    win = fold_windows()
    out = {i.tag: {} for i in I.TABLE}
    for pair in I.PAIRS:
        c = pd.read_parquet(cache_dir / f"{I.file_stem(pair)}.parquet")
        c["date"] = pd.to_datetime(c["date"], utc=True)
        for f, (a, b) in win.items():
            w = c[(c["date"] >= a) & (c["date"] < b)]
            for i in I.TABLE:
                out[i.tag].setdefault(pair, {})[f] = \
                    None if w.empty else round(float(w[I.column(i.tag)].notna().mean()), 4)
    return out


def main() -> int:
    from research.new_inputs import screen as SC
    from research.new_inputs import trades as TR

    meta = json.loads((CACHE / "build_meta.json").read_text())
    if meta["table_sha"] != I.TABLE_SHA:
        print(f"REFUSED: cache built from table {meta['table_sha']}, current {I.TABLE_SHA}")
        return 2
    bars = bar_coverage()
    t = SC.attach(TR.trades())
    out = {"label": "035 S0 coverage (no P&L read)", "table_sha": I.TABLE_SHA,
           "cache": meta, "inputs": {}}
    for i in I.TABLE:
        col = I.column(i.tag)
        tc = float(t[col].notna().mean())
        out["inputs"][i.tag] = {"family": i.family, "source": i.source,
                                "live_source": i.live_source,
                                "early_trade_coverage": round(tc, 4), "bars": bars[i.tag]}
        print(f"{i.tag:7s} trades {tc:6.1%}  live: {i.live_source}", flush=True)
    (HERE / "coverage.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
