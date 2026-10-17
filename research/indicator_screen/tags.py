"""037 column tags (added after the lock, 2026-10-16, at the user's request): which harvested columns
are not comparable across time and so cannot be regime or label candidates as-is.

- calendar: day-of-week / day-of-year / hour / minute-of-day sin/cos -- with < 6 years of data a
  calendar "effect" is calendar fitting;
- price_level: the column is in price units (value / close within [0.5, 2] on >= 90% of rows,
  every pair): bands, moving averages, SAR, Ichimoku lines, Donchian/Keltner levels. Inside a
  90-day window such a column mostly says where price sits in that window's range;
- price_scaled: in absolute price units without being a price level (ATR or rolling std in
  dollars, MACD lines): across the 11 pairs, log median |value| rises with log median close
  (slope >= 0.7, R^2 >= 0.8), so values are not comparable across pairs or across years;
- non_stationary: a cumulative series that trends with time (|Spearman(value, time)| >= 0.6 on at
  least half the pairs), e.g. finta's WOBV running total (the next column is at 0.38).

Tagging does not change any score; the report lists tagged columns apart from the candidates.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.tags
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.indicator_screen import harvest as H
from research.indicator_screen.panel import scored_columns, screen_pairs

OUT = H.HERE / "tags.json"
CALENDAR = re.compile(r"(^|:)(dow|doy|hour|mod|dom|month)_(sin|cos)$")
PRICE_BAND = (0.5, 2.0)
PRICE_SHARE = 0.9
TREND_RHO = 0.6
SCALE_SLOPE, SCALE_R2 = 0.7, 0.8
STEP = 16                     # every 16th scored row (4 h) per pair


def scale_fit(med_abs: dict[str, float], med_close: dict[str, float]) -> tuple[float, float]:
    """(slope, R^2) of log median |value| on log median close across pairs."""
    ps = [p for p in med_abs if med_abs[p] > 0 and med_close.get(p, 0) > 0]
    if len(ps) < 6:
        return 0.0, 0.0
    x = np.log([med_close[p] for p in ps])
    y = np.log([med_abs[p] for p in ps])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0, 0.0
    slope, icept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + icept)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(r2)


def classify(name: str, ratio_share: dict[str, float], trend_rho: dict[str, float],
             scale: tuple[float, float] = (0.0, 0.0)) -> str:
    if CALENDAR.search(name):
        return "calendar"
    if ratio_share and min(ratio_share.values()) >= PRICE_SHARE:
        return "price_level"
    if scale[0] >= SCALE_SLOPE and scale[1] >= SCALE_R2:
        return "price_scaled"
    if trend_rho and np.mean([abs(r) >= TREND_RHO for r in trend_rho.values()]) >= 0.5:
        return "non_stationary"
    return "indicator"


def compute() -> dict:
    cols = [c for c in scored_columns() if not c.startswith("raw_")]
    share = {c: {} for c in cols}
    rho = {c: {} for c in cols}
    med = {c: {} for c in cols}
    med_close = {}
    for p in screen_pairs() + [H.BTC]:
        f = H.read_cache(p, ["meta:data_ok", "close", *cols])
        f = f[f["meta:data_ok"] > 0].iloc[::STEP]
        close = f["close"].to_numpy(float)
        med_close[p] = float(np.median(close))
        t = np.arange(len(f))
        for c in cols:
            if c not in f:
                continue
            v = f[c].to_numpy(float)
            ok = np.isfinite(v) & (close > 0)
            if ok.sum() < 100:
                continue
            r = v[ok] / close[ok]
            med[c][p] = float(np.median(np.abs(v[ok])))
            share[c][p] = float(np.mean((r >= PRICE_BAND[0]) & (r <= PRICE_BAND[1])))
            rho[c][p] = float(spearmanr(v[ok], t[ok]).statistic) if np.ptp(v[ok]) > 0 else 0.0
    tags = {c: classify(c, share[c], rho[c], scale_fit(med[c], med_close)) for c in cols}
    return {"rule": __doc__.split("\n\n")[1].strip(), "tags": tags,
            "counts": pd.Series(tags).value_counts().to_dict()}


def load() -> dict[str, str]:
    """{column: tag}; btc: columns take their own column's tag; raw signals are 'indicator'."""
    if not OUT.exists():
        return {}
    return json.loads(OUT.read_text())["tags"]


def tag_of(col: str, tags: dict[str, str]) -> str:
    return tags.get(col[4:] if col.startswith("btc:") else col, "indicator")


def main(argv=None) -> int:
    res = compute()
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(res["counts"])
    for k in ("price_scaled", "non_stationary", "calendar"):
        print(k, sorted(c for c, t in res["tags"].items() if t == k)[:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
