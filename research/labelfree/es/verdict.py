"""Protocol §5 gates — the verdict the study is judged on (T049a-T049f).

ADDED BY THE /speckit-analyze PASS, which found the feature computed PBO and audited leakage
but had NO task producing this. Without it, 48,000 evaluations would run and no acceptance
gate would ever be evaluated. Principle II is NON-NEGOTIABLE and plan.md claimed it while
nothing implemented it.

    SC-001  pooled annualised return >= control + 3.35 pp/yr, bootstrap CI excluding zero
    SC-002  positive in >= 10 of 15 folds AND in each of W1/W2/W3      (Principle II)
    SC-003  beats all four mandatory baselines — the bar is the production control at
            +7.86 pp/yr measured over f05-f14, NOT zero
    SC-008  no configuration whose in-sample advantage exceeds its held-out advantage by
            more than the 3.35 pp/yr floor
"""
from __future__ import annotations

import numpy as np

from research.labelfree import admissibility as AD

FLOOR_PP = 3.35              # the acceptance floor, AGENT_GUIDE.md
SWEEP_NOISE_PP = 4.93        # one rvol sweep's entire span
BASELINE_BAR_PP = 7.86       # the production control over f05-f14 (baselines.py span mode)
MIN_POSITIVE_FOLDS = 10
N_FOLDS = 15
BOOTSTRAP_SEED = 20260913


def pooled(per_fold: list[dict], control_pp: float, n_boot: int = 10_000) -> dict:
    """SC-001. `per_fold` is one dict per fold: {fold_id, annualised_pp, test_days}."""
    if not per_fold:
        return {"gate": "SC-001", "passed": False, "reason": "no folds evaluated"}
    days = np.array([f.get("test_days", 90) for f in per_fold], dtype=float)
    vals = np.array([f["annualised_pp"] for f in per_fold], dtype=float)
    w = days / days.sum()
    arm_pp = float((vals * w).sum())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot = (vals[idx] * w[idx] / w[idx].sum(axis=1, keepdims=True)).sum(axis=1) - control_pp
    lo, hi = np.percentile(boot, [2.5, 97.5])
    delta = arm_pp - control_pp
    return {"gate": "SC-001", "arm_pp": round(arm_pp, 4), "control_pp": round(control_pp, 4),
            "delta_pp": round(delta, 4), "floor_pp": FLOOR_PP,
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "passed": bool(delta >= FLOOR_PP and (lo > 0 or hi < 0))}


def per_fold_positivity(per_fold: list[dict]) -> dict:
    """SC-002, first half. Principle II: a fold-concentrated win must fail."""
    pos = [f["fold_id"] for f in per_fold if f["annualised_pp"] > 0]
    return {"gate": "SC-002a", "positive_folds": len(pos), "of": len(per_fold),
            "required": MIN_POSITIVE_FOLDS, "which": pos,
            "passed": len(pos) >= MIN_POSITIVE_FOLDS and len(per_fold) == N_FOLDS}


def single_window(window_pp: dict) -> dict:
    """SC-002, second half. Principle II, NON-NEGOTIABLE.

    Without this a win concentrated in one era passes undetected -- the study has a recorded
    case of exactly that ("2 adjacent windows share a regime").
    """
    need = ("W1", "W2", "W3")
    missing = [w for w in need if w not in window_pp]
    return {"gate": "SC-002b", "windows": {w: round(window_pp.get(w, float("nan")), 4)
                                           for w in need},
            "missing": missing,
            "passed": not missing and all(window_pp[w] > 0 for w in need)}


def overfit_gap(rows: list[dict]) -> dict:
    """SC-008. Train advantage minus test advantage, per configuration."""
    bad = []
    for r in rows:
        gap = r["train_pp"] - r["test_pp"]
        if gap > FLOOR_PP:
            bad.append({"fold_id": r.get("fold_id"), "candidate_id": r.get("candidate_id"),
                        "train_pp": round(r["train_pp"], 3),
                        "test_pp": round(r["test_pp"], 3), "gap_pp": round(gap, 3)})
    return {"gate": "SC-008", "floor_pp": FLOOR_PP, "n_configurations": len(rows),
            "n_overfit": len(bad), "worst": sorted(bad, key=lambda x: -x["gap_pp"])[:5],
            "passed": not bad}


def baselines(arm_pp: float, arm_exposure: float, arm_tpy: float,
              baseline_pp: dict | None = None) -> dict:
    """SC-003. The bar is the PRODUCTION CONTROL at +7.86 pp/yr, not zero."""
    b = dict(baseline_pp or {"do_nothing": 0.00, "buy_and_hold": -14.60,
                             "ema_trend_20_60": -77.43, "production_control": 7.86})
    bar = max(b.values())
    gate = AD.compare(
        AD.Arm("candidate", arm_pp, arm_exposure, arm_tpy),
        AD.Arm("production_control", b["production_control"], 0.0611, 150.0),
        purpose="acceptance")
    return {"gate": "SC-003", "baselines_pp": b, "bar_pp": bar, "arm_pp": round(arm_pp, 4),
            "admissibility": gate,
            "passed": bool(arm_pp > bar and gate.get("admissible"))}


def status(gates: dict, pbo: float | None, delta_vs_incumbent_pp: float | None,
           beats_random: bool | None) -> dict:
    """The DERIVED verdict (data-model.md §7). Never typed by hand."""
    reasons = []
    if pbo is not None and pbo > 0.2:
        reasons.append(f"KILL: PBO {pbo:.3f} > 0.2 — the measured value IS the finding")
    if beats_random is False:
        reasons.append("KILL: the random control matched the search — the optimiser fit noise")
    g3 = gates.get("SC-003", {})
    if g3 and not g3.get("admissibility", {}).get("admissible", True):
        reasons.append("KILL: inadmissible — " + str(g3["admissibility"].get("verdict")))
    g8 = gates.get("SC-008", {})
    if g8 and not g8.get("passed", True):
        reasons.append(f"KILL: {g8['n_overfit']} configuration(s) overfit beyond the floor")

    unresolved = []
    if (delta_vs_incumbent_pp is not None
            and abs(delta_vs_incumbent_pp) < SWEEP_NOISE_PP):
        unresolved.append(
            f"NOT RESOLVED: |delta| {abs(delta_vs_incumbent_pp):.2f} pp is inside the "
            f"{SWEEP_NOISE_PP} pp sweep-noise band — an entire parameter sweep spans this, "
            f"so the move is indistinguishable from the sweep's own noise")
    if not gates.get("SC-001", {}).get("passed"):
        unresolved.append("NOT RESOLVED: nothing cleared SC-001 within the budget (SC-007) — "
                          "never 'no effect'")
    for g in ("SC-002a", "SC-002b"):
        if g in gates and not gates[g].get("passed"):
            unresolved.append(f"NOT RESOLVED: {g} — persistence failed (Principle II)")

    if reasons:
        return {"status": "KILL", "reasons": reasons + unresolved}
    if unresolved:
        return {"status": "NOT RESOLVED", "reasons": unresolved}
    return {"status": "ACCEPT", "reasons": ["every protocol §5 gate cleared"]}
