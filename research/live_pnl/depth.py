"""036 T020: execution cost from recorded order-book depth (research R5).

* `walk_cost_bps(levels, mid, stake, side)` walks the book against the mid. For a buy it consumes
  asks upward; for a sell, bids downward. Cost = (VWAP / mid − 1) × 1e4 for a buy, and
  (1 − VWAP / mid) × 1e4 for a sell. A stake beyond the recorded depth assumes the remainder
  fills at the last level, and is marked extrapolated.
* `participation(stake, qv24)` = stake ÷ rolling 24 h quote volume, which is causal.
* `MonotoneCurve` is an isotonic (non-decreasing) cost-vs-participation curve with linear
  interpolation, and `max_participation(budget)` is its inverse.
* `tiers(depth_by_pair)` assigns terciles of a structural measure: a pair's median top-20 quote
  depth. These are not per-pair tuned values (constitution, Principle X).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STAKE_GRID = (100, 200, 300, 500, 900, 1500, 2500, 5000)


def walk_cost_bps(levels, mid: float, stake: float, side: str) -> tuple[float, bool]:
    """(cost in bps, extrapolated). `levels` = [[price, base_qty], ...] best first."""
    if not levels or not mid or stake <= 0:
        raise ValueError("need levels, a mid and a positive stake")
    remaining, base, quote = float(stake), 0.0, 0.0
    for price, qty in levels:
        take = min(remaining, price * qty)
        base += take / price
        quote += take
        remaining -= take
        if remaining <= 1e-12:
            break
    extrapolated = remaining > 1e-12
    if extrapolated:
        last = levels[-1][0]
        base += remaining / last
        quote += remaining
    vwap = quote / base
    cost = (vwap / mid - 1.0) if side == "buy" else (1.0 - vwap / mid)
    return float(cost * 1e4), extrapolated


def participation(stake: float, qv24: float) -> float:
    return float("inf") if not qv24 or qv24 <= 0 else float(stake) / float(qv24)


@dataclass
class MonotoneCurve:
    p: np.ndarray       # increasing participation knots
    cost: np.ndarray    # non-decreasing cost (bps)

    @classmethod
    def fit(cls, p, cost, n_bins: int = 20) -> "MonotoneCurve":
        from sklearn.isotonic import IsotonicRegression

        p, cost = np.asarray(p, float), np.asarray(cost, float)
        ok = np.isfinite(p) & np.isfinite(cost)
        p, cost = p[ok], cost[ok]
        if len(p) < 2:
            raise ValueError("need at least 2 points")
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        kp = np.array([np.median(p[idx == i]) for i in range(len(edges) - 1) if (idx == i).any()])
        kc = np.array([np.median(cost[idx == i]) for i in range(len(edges) - 1) if (idx == i).any()])
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(kp, kc)
        return cls(kp, iso.predict(kp))

    def __call__(self, p) -> np.ndarray:
        return np.interp(np.asarray(p, float), self.p, self.cost)

    def max_participation(self, budget_bps: float) -> float:
        """The largest p with cost(p) <= budget (0 if even the smallest knot exceeds it)."""
        if self.cost[0] > budget_bps:
            return 0.0
        if self.cost[-1] <= budget_bps:
            return float(self.p[-1])
        i = int(np.searchsorted(self.cost, budget_bps, side="right"))
        c0, c1, p0, p1 = self.cost[i - 1], self.cost[i], self.p[i - 1], self.p[i]
        return float(p0 if c1 == c0 else p0 + (budget_bps - c0) * (p1 - p0) / (c1 - c0))

    def to_json(self) -> dict:
        return {"p": self.p.tolist(), "cost_bps": self.cost.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "MonotoneCurve":
        return cls(np.asarray(d["p"], float), np.asarray(d["cost_bps"], float))


def tiers(depth_by_pair: dict) -> dict:
    """Pair -> tier 0 (thinnest) .. 2 (deepest), by terciles of the given measure."""
    pairs = sorted(depth_by_pair, key=lambda k: depth_by_pair[k])
    n = len(pairs)
    return {p: min(2, i * 3 // n) for i, p in enumerate(pairs)}
