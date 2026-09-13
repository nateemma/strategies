"""THE CALIBRATION GATE (T006-T008, research R1, amended R1a).

The env's numpy step is a restatement of `sim.net_returns`, which is calibrated to the
arbiter at 0.00000000 bps and is a shared dependency of briefs 2, 5 and 6. An unverified
restatement is a different simulator wearing the same name.

TWO-PART GATE, because "bit-for-bit against MLX" was over-specified (R1a, measured
2026-09-12). MLX's `log()` and numpy's `log()` disagree by ~5 ulps at float32 -- the
discrepancy is already present in `bar_returns` before any of this module's arithmetic runs,
and NEITHER IS WRONG. Requiring rtol=0 across that boundary tests the libraries, not the
economics, and fails on something no change here can fix.

  PART 1 -- LOGIC (must be EXACT, rtol=0). Given the simulator's OWN bar returns, does the
           per-bar accumulation match `net_returns`? This is what isolates the four cases a
           naive step gets wrong:
              1. the w_{-1} = 0 opening boundary
              2. a path still in position when the window ends
              3. the liquidity cap binding
              4. a zero-volume bar
           Measured: 0.000e+00 on all four plus a random control.

  PART 2 -- PRECISION (bounded, not zero). The residual when the env computes its own bar
           returns in numpy. Reported with a ceiling of a few float32 ulps. A number ABOVE
           the ceiling is a real defect; a number at it is two correct log implementations.

Part 1 failing halts the feature. Part 2 exceeding its ceiling halts the feature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import sim                            # noqa: E402
from research.labelfree.env import step_economics, bar_returns_from_open   # noqa: E402
from research.labelfree import rl as RL                       # noqa: E402


def step_path(w: np.ndarray, open_px: np.ndarray, cost: np.ndarray,
              vol: np.ndarray) -> np.ndarray:
    """Accumulate the env's per-bar rewards over a whole path."""
    r = bar_returns_from_open(open_px)
    n = min(len(w), len(r))
    out = np.empty(n, dtype=float)
    prev = 0.0                                   # w_{-1} = 0, applied ONCE (case 1)
    for i in range(n):
        out[i], _, _ = step_economics(float(w[i]), prev, float(r[i]),
                                      float(cost[i]), float(vol[i]))
        prev = float(w[i])
    return out


def vectorised(w: np.ndarray, open_px: np.ndarray, cost: np.ndarray,
               vol: np.ndarray) -> np.ndarray:
    """The simulator's own answer, hard cap (evaluation mode)."""
    import mlx.core as mx
    out = sim.net_returns(mx.array(w.astype(np.float32)),
                          mx.array(open_px.astype(np.float32)),
                          mx.array(cost.astype(np.float32)),
                          volume_quote=mx.array(vol.astype(np.float32)), soft_cap=False)
    mx.eval(out)
    return np.asarray(out, dtype=float)


def make_case(kind: str, n: int, rng) -> tuple:
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n + RL.EXEC_LAG_BARS)))
    cost = np.full(n, 1e-4)
    vol = np.full(n, 1e9)
    w = rng.random(n)
    if kind == "ends_in_position":
        w[-5:] = 0.8                              # case 2
    elif kind == "cap_binds":
        vol = np.full(n, 0.5)                     # case 3: cap << requested
    elif kind == "zero_volume":
        vol = np.full(n, 1e9); vol[n // 2] = 0.0  # case 4
    elif kind == "opens_long":
        w[0] = 1.0                                # case 1: max requested at the boundary
    return w, px, cost, vol


def logic_only(w, open_px, cost, vol):
    """Part 1: step with the SIMULATOR'S bar returns, so only this module's economics vary."""
    import mlx.core as mx
    w32, px32, c32, v32 = (np.asarray(x, dtype=np.float32) for x in (w, open_px, cost, vol))
    r = np.asarray(sim.bar_returns(mx.array(px32)), dtype=np.float32)
    n = min(len(w32), len(r))
    out = np.empty(n, dtype=np.float32)
    prev = np.float32(0)                      # w_{-1} = 0, applied ONCE (case 1)
    for i in range(n):
        out[i], _, _ = step_economics(w32[i], prev, r[i], c32[i], v32[i])
        prev = w32[i]
    return out


PRECISION_CEILING = 5e-06        # a few float32 ulps at these magnitudes; above this is a bug


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--n-bars", type=int, default=400)
    ap.add_argument("--folds", default="f07", help="recorded for provenance only")
    a = ap.parse_args()

    rng = np.random.default_rng(RL.BOOTSTRAP_SEED)
    kinds = ["random", "opens_long", "ends_in_position", "cap_binds", "zero_volume"]
    logic = {k: 0.0 for k in kinds}
    prec = {k: 0.0 for k in kinds}
    logic_fail, checked = [], 0

    for p in range(a.n_paths):
        kind = kinds[p % len(kinds)]
        w, px, cost, vol = make_case(kind, a.n_bars, rng)
        theirs = vectorised(w, px, cost, vol)

        mine_logic = logic_only(w, px, cost, vol)
        n = min(len(mine_logic), len(theirs))
        t32 = theirs[:n].astype(np.float32)
        dl = float(np.max(np.abs(mine_logic[:n] - t32))) if n else 0.0
        logic[kind] = max(logic[kind], dl)
        if not np.array_equal(mine_logic[:n], t32):
            logic_fail.append((p, kind, dl))

        mine_full = step_path(w, px, cost, vol)
        m = min(len(mine_full), len(theirs))
        prec[kind] = max(prec[kind], float(np.max(np.abs(mine_full[:m] - theirs[:m]))))
        checked += 1

    print(f"EQUIVALENCE GATE — {checked} paths x {a.n_bars} bars\n")
    print("  PART 1 — LOGIC (simulator's own bar returns; must be EXACT)")
    print(f"  {'case':<20}{'max|diff|':>12}")
    for k in kinds:
        print(f"  {k:<20}{logic[k]:>12.2e}")
    print(f"  -> {'EXACT' if not logic_fail else '*** MISMATCH ***'}\n")

    print("  PART 2 — PRECISION (env computes its own log; bounded, not zero)")
    print(f"  {'case':<20}{'max|diff|':>12}")
    for k in kinds:
        flag = "  OVER CEILING" if prec[k] > PRECISION_CEILING else ""
        print(f"  {k:<20}{prec[k]:>12.2e}{flag}")
    worst = max(prec.values())
    print(f"  ceiling {PRECISION_CEILING:.0e} — MLX and numpy log() differ by ~5 ulps at")
    print(f"  float32 and neither is wrong; a value ABOVE the ceiling is a defect.\n")

    if logic_fail:
        print(f"  *** {len(logic_fail)} LOGIC MISMATCHES ***")
        for p, k, d in logic_fail[:5]:
            print(f"    path {p} ({k}): max|diff| {d:.3e}")
        print("\n  The env's economics are not the simulator's. Nothing downstream is evidence.")
        return 1
    if worst > PRECISION_CEILING:
        print(f"  *** PRECISION {worst:.2e} EXCEEDS CEILING {PRECISION_CEILING:.0e} ***")
        return 1
    print("  -> EQUIVALENT (logic exact, precision within ceiling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
