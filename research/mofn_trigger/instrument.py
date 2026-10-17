"""Instrument check for the mofn_4of8 trigger: each member computed by the trigger vs the 037
harvest's raw_buy column, per pair, full history (warm-up WARMUP bars excluded).

    PYTHONPATH=. ../../.venv/bin/python -m research.mofn_trigger.instrument
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.outcomes import audit_triggers as A

HERE = Path(__file__).resolve().parent
WARMUP = 200


def main() -> int:
    H._quiet()
    cfg = H.base_config()
    out = {}
    for pair in H.pairs(cfg):
        ohlcv = H.load_ohlcv(cfg, pair)
        got = A.mofn_votes(ohlcv)
        ref = pd.read_feather(H.CACHE / f"{pair.replace('/', '_')}.feather")
        assert len(ref) == len(ohlcv) and (pd.to_datetime(ref["date"]).values == ohlcv["date"].values).all()
        row = {}
        for (cls, var), col in zip(A.MOFN_MEMBERS, got.columns):
            name = f"raw_buy:{cls}" + (f":{var}" if var else "")
            r = np.nan_to_num(ref[name].to_numpy(float)) > 0.5
            g = got[col].to_numpy(bool)
            row[col] = {"mismatch": int((r[WARMUP:] != g[WARMUP:]).sum()), "fires": int(g[WARMUP:].sum())}
        out[pair] = row
        print(pair, {k: v["mismatch"] for k, v in row.items()}, flush=True)
    total = sum(v["mismatch"] for r in out.values() for v in r.values())
    (HERE / "instrument.json").write_text(json.dumps({"total_mismatch": total, "pairs": out}, indent=1))
    print("TOTAL mismatches", total)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
