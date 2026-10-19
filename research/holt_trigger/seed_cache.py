"""Seed the holt_triggers forecast cache from 038's saved Holt forecasts (pred/holt*.feather, f_48), so
backtests reuse the study's fits instead of refitting. Values are identical by construction (same
library call on the same closes); only candles on the trigger's date grid are written.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.holt_trigger.seed_cache
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.forecast import common as C
from research.forecast import frames as F
from research.outcomes import holt_triggers as H

STUDY = {32: "holt_w32", 64: "holt", 128: "holt_w128", 256: "holt_w256"}


def main() -> int:
    rows = []
    for w, m in STUDY.items():
        pred = pd.read_feather(C.PRED / f"{m}.feather", columns=["pair", "date", f"f_{H.HORIZON}"])
        for pair in C.pairs():
            fr = F.read(pair, ["date", "lc"])
            pos = pd.Series(fr.index, index=fr["date"])
            p = pred[(pred["pair"] == pair)].dropna()
            t = pos.reindex(p["date"]).to_numpy()
            ok = ~np.isnan(t) & (t >= w - 1)
            t = t[ok].astype(int)
            d = pd.DatetimeIndex(p["date"])[ok]
            on_grid = (((d - H.EPOCH) // H.BAR) % H.STRIDE == 0)
            lc = fr["lc"].to_numpy(float)
            for ti, di, fc in zip(t[on_grid], d[on_grid].asi8, p[f"f_{H.HORIZON}"].to_numpy()[ok][on_grid]):
                k = H._key(w, int(di), lc[ti], lc[ti - w + 1])
                rows.append((k[0], k[1], k[2], k[3], float(fc)))
    H.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=["w", "date_ns", "a", "b", "fc"])
    out.to_feather(H.CACHE_DIR / "seed_038.feather")
    print(f"seeded {len(out)} forecasts ({out.groupby('w').size().to_dict()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
