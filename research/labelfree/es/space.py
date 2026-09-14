"""The ten-dimension exit/guard search space (T004, T004a, T005, T006).

Bounds and incumbent values are specs/013-labelfree-es-policy-search/data-model.md §1. The
incumbent is READ from the strategy's own files rather than hard-coded, so it cannot drift
away from what production actually runs.

WHY TEN AND NOT THE BRIEF'S EIGHT. Two dimensions were added during planning:

  cexit_take_profit   51.6% of exits and +179.7% of P&L across the 378 archived trades. It is
                      the mechanism that OPPOSES the stop stack (-120.2%), and trading one
                      against the other is precisely what a joint search can do and a
                      one-at-a-time sweep cannot. The brief's list omits it.

  hold_window_hours   minimum holding time, which the spec names as a lever (spec.md:14). An
                      earlier draft called `unclog_12h` a portfolio-level rule and concluded
                      the lever was unsearchable. That was wrong -- BaseStrategy.py:1211-1218
                      shows it is purely per-trade -- so it is searched.

INADMISSIBILITY, NOT CLIPPING. A candidate violating floor < cap < 0 is rejected. Clipping
would map distinct candidates onto one point, and the deflation counts attempts: two clipped
candidates that became identical would be scored once and counted twice, which is exactly
backwards.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STRAT_PARAMS = ROOT / "NNNC" / "NNNC_MLX.json"
BASE_STRATEGY = ROOT / "Framework" / "BaseStrategy.py"


@dataclass(frozen=True)
class Dim:
    name: str
    lo: float
    hi: float
    incumbent: float
    integer: bool = False
    # True when the incumbent sits on a bound because the MECHANISM cannot go further, not
    # because the space was drawn too narrowly. FR-019 must not fire on these.
    one_sided: bool = False
    note: str = ""

    @property
    def width(self) -> float:
        return self.hi - self.lo


def _params() -> dict:
    """Incumbent values from the strategy's own hyperopt file."""
    d = json.loads(STRAT_PARAMS.read_text())["params"]
    return {**d.get("buy", {}), **d.get("sell", {})}


def _base_attr(name: str) -> float:
    """A plain class attribute from BaseStrategy.py, read from source.

    Importing BaseStrategy drags in freqtrade's strategy machinery for three constants, so
    the values are parsed instead. The regex is anchored to a 4-space-indented assignment so
    it cannot match a commented-out example (BaseStrategy.py:262-263 has exactly that).
    """
    import re
    m = re.search(rf"^    {name} = (-?[\d.]+)", BASE_STRATEGY.read_text(), re.M)
    if not m:
        raise RuntimeError(f"{name} not found as a class attribute in {BASE_STRATEGY}")
    return float(m.group(1))


def build() -> list[Dim]:
    p = _params()
    return [
        Dim("atr_stoploss_multiplier", 1.0, 5.0, _base_attr("atr_stoploss_multiplier"),
            note="scales the ATR stop; the stack it feeds is 25.9% of exits"),
        Dim("atr_stoploss_floor", -0.10, -0.02, _base_attr("atr_stoploss_floor"),
            note="loosest stop allowed (most negative)"),
        Dim("atr_stoploss_cap", -0.04, -0.005, _base_attr("atr_stoploss_cap"),
            note="tightest stop allowed (closest to zero)"),
        Dim("stoploss_grace_hours", 0.0, 12.0, float(p["stoploss_grace_hours"]),
            note="how long the stop is held wide; gates when the ATR stop becomes reachable"),
        Dim("stoploss_grace_level", -0.40, -0.05, float(p["stoploss_grace_level"]),
            note="how wide it is held"),
        Dim("cexit_take_profit", 0.010, 0.060, float(p["cexit_take_profit"]),
            note="51.6% of exits, +179.7% of P&L — added in planning, see module docstring"),
        Dim("cexit_max_days", 2, 28, float(p["cexit_max_days"]), integer=True,
            note="the holding-time upper bound"),
        Dim("entry_rvol_threshold", 0.25, 4.0, float(p["entry_rvol_threshold"]),
            note="the dominant entry guard"),
        Dim("entry_guard_threshold", 0.0, 1.0, float(p["entry_guard_threshold"]),
            note="the second entry guard"),
        Dim("hold_window_hours", 12.0, 48.0, 12.0, one_sided=True,
            note="minimum holding time. HoldWindowMixin can only WIDEN the unclog_12h "
                 "window, so 12.0 is both the incumbent and the floor — a property of the "
                 "mechanism, not a narrow space. Exempt from FR-019."),
    ]


DIMS = build()
NAMES = [d.name for d in DIMS]
INCUMBENT = {d.name: (round(d.incumbent) if d.integer else d.incumbent) for d in DIMS}


# --------------------------------------------------------------------------- encode / decode

def encode(params: dict) -> np.ndarray:
    """Decoded parameters -> unbounded z-space, so CMA-ES never sees a constrained problem."""
    z = np.empty(len(DIMS))
    for i, d in enumerate(DIMS):
        u = (float(params[d.name]) - d.lo) / d.width          # -> [0, 1]
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        z[i] = math.log(u / (1.0 - u))                        # logit
    return z


def decode(z) -> dict:
    """z-space -> parameters. Inverse of `encode` to within float tolerance."""
    out = {}
    for i, d in enumerate(DIMS):
        u = 1.0 / (1.0 + math.exp(-float(z[i])))
        v = d.lo + u * d.width
        out[d.name] = int(round(v)) if d.integer else v
    return out


class Inadmissible(ValueError):
    """A candidate that is not a point in the space at all."""


def validate(params: dict) -> None:
    """Raise `Inadmissible` rather than clip — see the module docstring."""
    if not (params["atr_stoploss_floor"] < params["atr_stoploss_cap"] < 0):
        raise Inadmissible(
            f"floor {params['atr_stoploss_floor']:.4f} < cap "
            f"{params['atr_stoploss_cap']:.4f} < 0 violated")


def on_bound(params: dict, rtol: float = 1e-6) -> list[str]:
    """Dimensions sitting on a bound (FR-019), EXCLUDING known one-sided mechanisms.

    `hold_window_hours` is at its floor by construction; reporting it as "a space drawn too
    narrowly" would be a false finding every single run.
    """
    hits = []
    for d in DIMS:
        if d.one_sided:
            continue
        v = float(params[d.name])
        tol = rtol * max(abs(d.width), 1.0)
        if abs(v - d.lo) <= tol or abs(v - d.hi) <= tol:
            hits.append(d.name)
    return hits


def candidate_id(params: dict) -> str:
    """Deterministic hash of the DECODED values.

    Decoded, so two z-vectors that round to the same `cexit_max_days` collide by
    construction: one evaluation for scoring, two for the trial count.
    """
    payload = json.dumps({k: (int(params[k]) if any(d.name == k and d.integer for d in DIMS)
                              else round(float(params[k]), 10))
                          for k in NAMES}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def distance_from_incumbent(params: dict) -> float:
    """Mean normalised |x - incumbent| over the ten dimensions, in [0, 1] (data-model §1a)."""
    return float(np.mean([abs(float(params[d.name]) - d.incumbent) / d.width for d in DIMS]))


def _show() -> None:
    print(f"{'#':>3}  {'parameter':<26}{'incumbent':>11}  {'bounds':<20}source")
    for i, d in enumerate(DIMS, 1):
        src = "NNNC_MLX.json" if d.name in _params() else (
            "BaseStrategy.py" if d.name.startswith("atr_") else "mechanism (see note)")
        b = f"[{d.lo:g}, {d.hi:g}]" + ("  int" if d.integer else "")
        print(f"{i:>3}  {d.name:<26}{d.incumbent:>11.4g}  {b:<20}{src}")
        if d.one_sided:
            print(f"     └─ one-sided: {d.note.splitlines()[0]}")


def _check(n: int = 10_000, seed: int = 7) -> int:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        p = {d.name: (int(rng.integers(d.lo, d.hi + 1)) if d.integer
                      else float(rng.uniform(d.lo, d.hi))) for d in DIMS}
        back = decode(encode(p))
        for d in DIMS:
            if d.integer:
                if int(back[d.name]) != int(p[d.name]):
                    print(f"FAIL {d.name}: {p[d.name]} -> {back[d.name]}")
                    return 1
            else:
                worst = max(worst, abs(back[d.name] - p[d.name]) / d.width)
    print(f"round-trip OK over {n} points; worst relative error {worst:.2e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        raise SystemExit(_check())
    _show()
