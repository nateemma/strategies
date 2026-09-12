"""Liveness probes: prove a swept parameter ACTUALLY changes what it claims to control.

Four parameters in this study were present, configured, believed active, and doing nothing
or half-doing it:

    ranking range      ranked on TRAIN instead of held-out          60 runs void
    label_threshold    one panel served every threshold             15 runs void
    purge / embargo    apply_leakage_control never called           arms B/C void
    turnover_control   arms.py had no loss-side penalty branch      the match lever failed

Every one produced plausible numbers and raised nothing. Wiring checks did not catch them --
the parameter WAS passed, it just reached nothing that mattered. Only a check on the value
ACTUALLY APPLIED, or on the quantity supposedly controlled, catches this class.

These probes run the REAL training path on a tiny synthetic panel, not a reimplementation of
it. That is deliberate: a duplicated loss here would drift from `arms.train_arm` and start
certifying code that is no longer the code that runs. The only thing altered for speed is
the chunk size, which changes the window length, not the loss.

    from research.labelfree.liveness import check_all
    check_all()      # raises on the first dead parameter
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROBE_CHUNK = 200          # a real chunk is 2880; 200 keeps a probe to a few seconds
PROBE_T = 400
PROBE_PAIRS = 2
REL_TOL = 1e-9             # "differs" means differs, not "differs appreciably"


@contextmanager
def _small_chunks():
    """Shrink the training chunk for the duration of a probe. Loss logic is untouched."""
    from research.labelfree import arms
    old = arms.CHUNK
    arms.CHUNK = PROBE_CHUNK
    try:
        yield
    finally:
        arms.CHUNK = old


def mini_panel(seed: int = 0, n_pairs: int = PROBE_PAIRS, T: int = PROBE_T,
               n_feat: int = 24) -> tuple:
    """A synthetic panel in load_panel's 9-tuple shape. Trending prices, non-trivial labels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_pairs, T, n_feat)).astype(np.float32)
    drift = np.linspace(0, 0.5, T)[None, :]
    P = (100 * np.exp(drift + 0.02 * rng.standard_normal((n_pairs, T)))).astype(np.float32)
    C = np.full((n_pairs, T), 1e-4, dtype=np.float32)
    V = np.full((n_pairs, T), 1e9, dtype=np.float32)
    M = np.ones((n_pairs, T), dtype=np.float32)
    Y = rng.integers(0, 3, size=(n_pairs, T)).astype(int)
    Vol = np.abs(rng.standard_normal((n_pairs, T))).astype(np.float32) * 0.01
    idx = None
    leak = {"purge_bars": 48, "embargo_bars": 48, "side": "synthetic",
            "n_bars": T, "usable_share": 1.0}
    return (X, P, C, V, M, Y, Vol, idx, leak)


def _train_loss(spec: dict, seed: int = 1) -> float:
    """Final training loss for one configuration, through the real train_arm."""
    from research.labelfree.arms import ArmConfig, train_arm
    tr, te = mini_panel(0), mini_panel(1)
    cfg = ArmConfig(seed=seed, fold_id="probe", epochs=1, **spec)
    with _small_chunks():
        return float(train_arm(cfg, tr, te)["final_loss"])


def _policy_output(spec: dict, seed: int = 1) -> np.ndarray:
    """Realised exposure path for one configuration, through the real policy."""
    import mlx.core as mx
    from research.labelfree.arms import ArmConfig, build_policy, windows
    X = mini_panel(0)[0]
    cfg = ArmConfig(seed=seed, fold_id="probe", **spec)
    pol = build_policy(cfg, X.shape[-1])
    w = pol(mx.array(windows(X[0, :PROBE_CHUNK])))
    mx.eval(w)
    return np.asarray(w)


# ---------------------------------------------------------------------------
# The registry. Each entry: two configurations differing in ONE parameter, and the
# quantity that parameter claims to control. If the two agree, the parameter is dead.
# ---------------------------------------------------------------------------

PROBES = {
    "gamma": dict(
        controls="perceived cost inside the training loss",
        a={"arm": "pnl_control", "gamma": 1.0},
        b={"arm": "pnl_control", "gamma": 20.0},
        measure=_train_loss),
    "lam_risk": dict(
        controls="the variance term of the utility objective",
        a={"arm": "pnl_control", "lam_risk": 0.5},
        b={"arm": "pnl_control", "lam_risk": 8.0},
        measure=_train_loss),
    # A PROBE MUST SIT IN ITS PARAMETER'S ACTIVE REGIME. participation_weight scales a
    # FLOOR penalty that contributes nothing while exposure is already above
    # min_exposure -- train.py records "identical results at 0 and 100 for the penalty
    # arms" for exactly this reason. Probing it at the default floor therefore reports
    # DEAD on correct code. min_exposure is pinned high here so the term is actually on.
    "participation_weight": dict(
        controls="the degeneracy guard's strength (probed with the floor ACTIVE)",
        a={"arm": "pnl_control", "min_exposure": 0.9, "participation_weight": 1.0},
        b={"arm": "pnl_control", "min_exposure": 0.9, "participation_weight": 500.0},
        measure=_train_loss),
    "min_exposure": dict(
        controls="where the participation floor sits",
        a={"arm": "pnl_control", "min_exposure": 0.01},
        b={"arm": "pnl_control", "min_exposure": 0.60},
        measure=_train_loss),
    "blend_alpha": dict(
        controls="the label cross-entropy term's weight (Arm B)",
        a={"arm": "blended", "blend_alpha": 0.0, "label_threshold": 0.007},
        b={"arm": "blended", "blend_alpha": 0.9, "label_threshold": 0.007},
        measure=_train_loss),
    "lam_turnover": dict(
        controls="the explicit |dw| penalty -- the branch that was MISSING from arms.py",
        a={"arm": "pnl_control", "turnover_control": "penalty", "lam_turnover": 0.0},
        b={"arm": "pnl_control", "turnover_control": "penalty", "lam_turnover": 5.0},
        measure=_train_loss),
    "band_width": dict(
        controls="the learned no-trade zone, hence turnover, at FIXED architecture",
        a={"arm": "pnl_control", "band_width": 0.1},
        b={"arm": "pnl_control", "band_width": 10.0},
        measure=_policy_output),
    "turnover_control": dict(
        controls="which turnover mechanism runs at all",
        a={"arm": "pnl_control", "turnover_control": "band"},
        b={"arm": "pnl_control", "turnover_control": "penalty", "lam_turnover": 1.0},
        measure=_policy_output),
}


def check(name: str) -> dict:
    """Probe one parameter. Returns the two measurements and whether they differ."""
    p = PROBES[name]
    va, vb = p["measure"](p["a"]), p["measure"](p["b"])
    if isinstance(va, np.ndarray):
        differs = not np.allclose(va, vb, rtol=REL_TOL, atol=0.0)
        sa, sb = float(np.mean(va)), float(np.mean(vb))
    else:
        differs = abs(va - vb) > REL_TOL * max(1.0, abs(va))
        sa, sb = va, vb
    return {"parameter": name, "controls": p["controls"], "a": sa, "b": sb,
            "differs": differs}


def check_all(names: list[str] | None = None, raise_on_dead: bool = True) -> list[dict]:
    out = []
    for n in (names or list(PROBES)):
        r = check(n)
        out.append(r)
        status = "LIVE" if r["differs"] else "*** DEAD ***"
        print(f"  {n:<22} {status:<12} {r['a']:+.6g} vs {r['b']:+.6g}   ({r['controls']})",
              flush=True)
        if raise_on_dead and not r["differs"]:
            raise ValueError(
                f"{n} is DEAD: it controls {r['controls']}, but changing it left that "
                f"quantity at {r['a']!r}. Any sweep over it would be vacuous.")
    return out


if __name__ == "__main__":
    check_all()
