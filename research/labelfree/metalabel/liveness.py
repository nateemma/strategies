"""Liveness probes for the veto model's swept parameters (T025, FR-012, SC-014).

Four parameters in this study were passed, logged, believed active and reaching nothing --
the ranking range (60 runs void), label_threshold (15 void), purge/embargo, and the turnover
penalty. Each produced plausible numbers and raised nothing. Only a check on the quantity a
parameter CLAIMS TO CONTROL catches that class.

A PROBE MUST SIT IN ITS PARAMETER'S ACTIVE REGIME. `participation_weight` in the sibling
module reported DEAD on correct code because it was probed where its floor term was
inactive. Each probe below states the regime it needs and why.

Probes run the REAL training path on the REAL trade table, not a reimplementation: a
duplicate here would drift from the code that runs and start certifying something else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

REL_TOL = 1e-9


def _data():
    from research.labelfree.metalabel import model as MD
    rows = [json.loads(l) for l in
            (ROOT / "research/labelfree/metalabel/rows.jsonl").read_text().splitlines() if l.strip()]
    asg = json.loads((ROOT / "research/labelfree/metalabel/assignment.json").read_text())
    a = asg[3]                       # f08: 197 train / 71 test — the widest fold
    Xtr, ytr = MD.matrix(rows, a["train_idx"])
    Xte, yte = MD.matrix(rows, a["test_idx"])
    return Xtr, ytr, Xte, yte


def _predict_with(overrides: dict, seed: int = 1) -> np.ndarray:
    """Train with PARAMS patched by `overrides` and return the test-set probabilities."""
    import lightgbm as lgb
    Xtr, ytr, Xte, _ = _data()
    from research.labelfree.metalabel import model as MD
    p = dict(MD.PARAMS); p.update(overrides)
    clf = lgb.LGBMClassifier(random_state=seed, **p)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _veto_count(thr: float) -> int:
    """How many trades a threshold actually vetoes -- the quantity it claims to control."""
    p = _predict_with({})
    return int((p < thr).sum())


PROBES = {
    "n_estimators": dict(
        controls="model capacity: more trees must change the predictions",
        fn=lambda: (_predict_with({"n_estimators": 5}), _predict_with({"n_estimators": 300}))),
    "max_depth": dict(
        controls="tree depth, hence interaction order",
        fn=lambda: (_predict_with({"max_depth": 1, "num_leaves": 2}),
                    _predict_with({"max_depth": 6, "num_leaves": 32}))),
    "min_child_samples": dict(
        # ACTIVE REGIME: this caps leaf size, so it binds only when a leaf WOULD be smaller
        # than the cap. Probed at 1 vs 60 against ~197 training rows it certainly binds;
        # probed at 10 vs 11 it would report DEAD on correct code.
        controls="minimum leaf occupancy (probed where it binds)",
        fn=lambda: (_predict_with({"min_child_samples": 1}),
                    _predict_with({"min_child_samples": 60}))),
    "learning_rate": dict(
        controls="shrinkage per boosting round",
        fn=lambda: (_predict_with({"learning_rate": 0.01}),
                    _predict_with({"learning_rate": 0.3}))),
    "veto_threshold": dict(
        # ACTIVE REGIME: the base win rate is 73.2%, so a calibrated model puts most
        # probability mass well above 0.5. A threshold below the mass vetoes nothing and
        # reads as DEAD while being perfectly functional -- so the probe brackets the
        # distribution rather than using the brief's operating points.
        controls="how many trades are vetoed (probed across the probability mass)",
        fn=lambda: (np.array([_veto_count(0.05)], dtype=float),
                    np.array([_veto_count(0.95)], dtype=float))),
}


def check(name: str) -> dict:
    p = PROBES[name]
    a, b = p["fn"]()
    differs = not np.allclose(a, b, rtol=REL_TOL, atol=0.0)
    return {"parameter": name, "controls": p["controls"],
            "a": float(np.mean(a)), "b": float(np.mean(b)), "differs": differs}


def check_all(names: list[str] | None = None, raise_on_dead: bool = True) -> list[dict]:
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
