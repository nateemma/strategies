"""State -> entry permission, and the Principle IV liveness record (T027, T028, T030).

A hard ENTRY gate: while the filtered state is the high-volatility state, no new entries. Open
positions and every exit are untouched -- the exit stack stays the incumbent's, so the
comparison changes exactly one thing.

WHY `activity()` IS A PRECONDITION AND NOT A DIAGNOSTIC. More than half of `regime/VERDICT.md`'s
evidence -- 18 of 32 rows, measured 2026-09-15 -- scored edge_pct exactly 0 because n_taken ==
n_marginal: the gate admitted every marginal trade. Those are not failures of regime detection,
they are gates that never fired, and they are indistinguishable from a clean null unless
something refuses to report performance for them. That is this module's job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import MIN_ACTIVE_FOLDS


class InertGate(RuntimeError):
    """Raised when a performance statistic is requested from a gate that never fired."""


def decide(states: pd.DataFrame, high_vol_state: int) -> pd.DataFrame:
    if (states["mode"] != "filtered").any():
        raise ValueError("gate.decide refuses smoothed states -- that path is the leak "
                         "reference and must never reach a deployed run (FR-004)")
    out = states.copy()
    out["blocked"] = out["state"].to_numpy() == high_vol_state
    return out


def activity(decisions: pd.DataFrame, fold_id: str) -> dict:
    n = int(len(decisions))
    nb = int(decisions["blocked"].sum())
    if nb == 0:
        v = "inert"
    elif nb == n:
        v = "total"
    else:
        v = "active"
    return {"fold_id": fold_id, "n_candidates": n, "n_blocked": nb,
            "block_rate": (nb / n) if n else 0.0, "verdict": v}


def require_active(act: dict) -> None:
    """Callers MUST pass through here before computing any performance statistic."""
    if act["verdict"] == "inert":
        raise InertGate(
            f"{act['fold_id']}: the gate blocked 0 of {act['n_candidates']} candidates. "
            "This is a DEFECT under Principle IV -- report it as 'the gate never fired', "
            "never as evidence about regime gating (SC-012).")
    if act["verdict"] == "total":
        raise InertGate(
            f"{act['fold_id']}: the gate blocked ALL {act['n_candidates']} candidates. "
            "Abstention is not risk reduction; refused by admissibility (FR-003).")


def fold_summary(acts: list[dict]) -> dict:
    active = [a for a in acts if a["verdict"] == "active"]
    rates = np.array([a["block_rate"] for a in acts], dtype=float)
    return {
        "n_folds": len(acts),
        "n_active": len(active),
        "meets_min_active": len(active) >= MIN_ACTIVE_FOLDS,
        "block_rate_mean": float(rates.mean()) if len(rates) else 0.0,
        "block_rate_min": float(rates.min()) if len(rates) else 0.0,
        "block_rate_max": float(rates.max()) if len(rates) else 0.0,
    }
