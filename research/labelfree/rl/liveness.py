"""Liveness probes for the swept parameters (T029-T031, FR-019, SC-015).

Four parameters in this study were passed, logged, believed active and reaching nothing.
Only a check on the quantity a parameter CLAIMS TO CONTROL catches that class.

EACH PROBE SITS IN ITS ACTIVE REGIME. `participation_weight` in a sibling module reported
DEAD on correct code because it was probed where its floor term was inactive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import rl as RL                        # noqa: E402
from research.labelfree.env import PooledTradingEnv            # noqa: E402
from research.labelfree.rl.rewards import SPECS                # noqa: E402

REL_TOL = 1e-9


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n + RL.EXEC_LAG_BARS)))
    X = rng.standard_normal((n, 4)).astype(np.float32)
    return X, px, np.full(n, 1e-4), np.full(n, 1e9)


def _exposure_path(band: float, targets: np.ndarray) -> np.ndarray:
    """Drive the env with a FIXED target sequence so only `band` varies."""
    X, px, c, v = _synthetic()
    env = PooledTradingEnv(X, px, c, v, continuous=True, band=band)
    env.reset(seed=0)
    out = []
    for t in targets[:env.n]:
        _, _, term, trunc, info = env.step(np.array([t], dtype=np.float32))
        out.append(info["w"])
        if term or trunc:
            break
    return np.array(out)


def _turnover(band: float) -> float:
    rng = np.random.default_rng(7)
    return float(np.abs(np.diff(_exposure_path(band, rng.random(600)))).sum())


def _reward_under(spec_name: str) -> float:
    X, px, c, v = _synthetic()
    env = PooledTradingEnv(X, px, c, v, reward_spec=SPECS[spec_name]())
    env.reset(seed=0)
    tot = 0.0
    for i in range(env.n):
        _, r, term, trunc, _ = env.step(1 if i % 7 else 0)   # forced churn, both specs alike
        tot += r
        if term or trunc:
            break
    return tot


def _drawdown_len(limit: float) -> int:
    X, px, c, v = _synthetic(seed=3)
    env = PooledTradingEnv(X, px, c, v, max_drawdown=limit)
    env.reset(seed=0)
    for i in range(env.n):
        _, _, term, trunc, _ = env.step(1)
        if term or trunc:
            return i + 1
    return env.n


PROBES = {
    "band_width": dict(
        # THE ONE TO WATCH (research R4). A FIXED band should dial turnover monotonically;
        # brief 02's LEARNED band instead found one of two attractors and made turnover a
        # coin flip across seeds. Probed across the target distribution so it binds.
        controls="no-trade width, hence turnover — R4's prediction is tested here",
        fn=lambda: (np.array([_turnover(0.0)]), np.array([_turnover(0.5)]))),
    "reward_spec": dict(
        controls="which economics the env pays",
        fn=lambda: (np.array([_reward_under("realised")]),
                    np.array([_reward_under("freqai_shaped")]))),
    "max_drawdown": dict(
        # ACTIVE REGIME: a limit above the episode's worst drawdown never binds and would
        # report DEAD on correct code. 1e-4 binds; 1e9 cannot.
        controls="episode termination (probed where it binds)",
        fn=lambda: (np.array([float(_drawdown_len(1e-4))]),
                    np.array([float(_drawdown_len(1e9))]))),
    "continuous_action": dict(
        controls="which action parameterisation runs at all",
        fn=lambda: (_exposure_path(0.0, np.full(600, 0.37)),
                    np.array([1.0] * 598))),
}


def check(name: str) -> dict:
    p = PROBES[name]
    a, b = p["fn"]()
    n = min(len(a), len(b))
    differs = not np.allclose(a[:n], b[:n], rtol=REL_TOL, atol=0.0)
    return {"parameter": name, "controls": p["controls"],
            "a": float(np.mean(a)), "b": float(np.mean(b)), "differs": differs}


def check_all(names=None, raise_on_dead: bool = True):
    out = []
    for n in (names or list(PROBES)):
        r = check(n)
        print(f"  {n:<20} {'LIVE' if r['differs'] else '*** DEAD ***':<14}"
              f"{r['a']:+.6g} vs {r['b']:+.6g}   ({r['controls']})", flush=True)
        out.append(r)
        if raise_on_dead and not r["differs"]:
            raise ValueError(
                f"{n} is DEAD: it controls {r['controls']}, but changing it left that "
                f"quantity at {r['a']!r}. Any sweep over it would be vacuous.")
    return out


if __name__ == "__main__":
    check_all()
