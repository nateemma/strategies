"""Candidate registry for feature 003 — tail-targeted breakout labelling.

Implements the Candidate entity from data-model.md. Every candidate MUST cite the
measurement that motivates it (FR-008); a candidate below the density floor MUST
carry an `excluded_reason` and is never trained (FR-002).

The five guard_metric cells were screened in research.md R3 and are pre-recorded
here so they are not re-derived. The two remaining candidates carry empty
screening fields until T011/T012 populate them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Below this share of bars the classifier stops predicting the signal class
# (aug_risk all-Hold collapse). research.md R3.
DENSITY_FLOOR_PCT = 5.5

# The forced-exit rule fires at 12h whenever profit exceeds 0.5%
# (Framework/BaseStrategy.py:1040). A candidate whose payoff peaks beyond this
# cannot be tested as a label change alone. research.md R4.
FORCED_EXIT_HOURS = 12.0

BASIS_GUARD = "guard_metric"    # labels_breakout_gbb, method 24
BASIS_BB = "bb_position"        # labels_breakout_bb, method 26 -- scale [-1, 1]


@dataclass
class Candidate:
    id: str
    horizon: int
    min_gain: float
    basis: str
    motivating_measurement: str
    mae_bound: Optional[float] = None
    excluded_reason: str = ""
    # Screening fields -- None until a ScreeningRecord is attached
    density_pct: Optional[float] = None
    median_gain_pct: Optional[float] = None
    clears_stop_pct: Optional[float] = None
    reward_risk: Optional[float] = None
    time_to_peak_h: Optional[float] = None
    stopped_first_pct: Optional[float] = None

    @property
    def screened(self) -> bool:
        return self.density_pct is not None

    @property
    def requires_hold_change(self) -> Optional[bool]:
        """Derived: does the payoff peak beyond the forced-exit window?"""
        if self.time_to_peak_h is None:
            return None
        return self.time_to_peak_h > FORCED_EXIT_HOURS

    @property
    def admitted(self) -> bool:
        return not self.excluded_reason

    def validate(self) -> None:
        if not self.motivating_measurement.strip():
            raise ValueError(f"{self.id}: a candidate MUST cite a measurement (FR-008)")
        if self.basis not in (BASIS_GUARD, BASIS_BB):
            raise ValueError(f"{self.id}: unknown basis {self.basis!r}")
        if self.screened and self.density_pct < DENSITY_FLOOR_PCT and self.admitted:
            raise ValueError(
                f"{self.id}: density {self.density_pct}% is below the "
                f"{DENSITY_FLOOR_PCT}% floor but no excluded_reason is set (FR-002)"
            )


# ---------------------------------------------------------------------------
# Pre-recorded from research.md R3 / R5 -- measured, not re-derived.
# ---------------------------------------------------------------------------

CANDIDATES = [
    Candidate(
        id="C0_h48_t005_CONTROL",
        horizon=48, min_gain=0.005, basis=BASIS_GUARD,
        motivating_measurement=(
            "The parked configuration. Median labelled gain 2.83% against a -4%..-6% "
            "operative stop; only 26.0% of labelled bars ever clear it; reward:risk "
            "0.57. This is the defect the feature exists to test."),
        density_pct=9.07, median_gain_pct=2.83, clears_stop_pct=26.0,
        reward_risk=0.57, time_to_peak_h=6.8, stopped_first_pct=4.5,
    ),
    Candidate(
        id="C1_h48_t02",
        horizon=48, min_gain=0.02, basis=BASIS_GUARD,
        motivating_measurement=(
            "The best cell requiring NO exit change (FR-004): density 5.81% stays "
            "above the floor and time-to-peak 8.2h stays inside the 12h forced exit, "
            "while reward:risk improves 0.57 -> 0.86 and stop-clearing 26.0% -> 40.6%. "
            "Isolates the label effect from the hold window."),
        density_pct=5.81, median_gain_pct=4.29, clears_stop_pct=40.6,
        reward_risk=0.86, time_to_peak_h=8.2, stopped_first_pct=5.5,
    ),
    Candidate(
        id="C2_h48_t03",
        horizon=48, min_gain=0.03, basis=BASIS_GUARD,
        motivating_measurement=(
            "Better economics than C1 (reward:risk 1.07, 54.9% clearing the stop) but "
            "density 4.30% falls below the floor."),
        density_pct=4.30, median_gain_pct=5.34, clears_stop_pct=54.9,
        reward_risk=1.07,
        excluded_reason=(
            "density 4.30% is below the 5.5% all-Hold collapse floor (FR-002)"),
    ),
    Candidate(
        id="C3_h96_t03_TAIL",
        horizon=96, min_gain=0.03, basis=BASIS_GUARD,
        motivating_measurement=(
            "The best-economics cell above the density floor (FR-005): reward:risk "
            "1.28 and 66.8% of labelled bars clearing the stop, at 5.92% density. "
            "REQUIRES the hold-window change -- time-to-peak 17.0h exceeds the 12h "
            "forced exit (research.md R4)."),
        density_pct=5.92, median_gain_pct=6.41, clears_stop_pct=66.8,
        reward_risk=1.28, time_to_peak_h=17.0, stopped_first_pct=11.2,
    ),
    Candidate(
        id="C4_h96_t05",
        horizon=96, min_gain=0.05, basis=BASIS_GUARD,
        motivating_measurement=(
            "Best raw economics of any cell (reward:risk 1.69, 100% clearing the "
            "stop) but density 3.95% is well below the floor."),
        density_pct=3.95, median_gain_pct=8.43, clears_stop_pct=100.0,
        reward_risk=1.69, time_to_peak_h=18.2, stopped_first_pct=12.7,
        excluded_reason=(
            "density 3.95% is below the 5.5% all-Hold collapse floor (FR-002)"),
    ),

    # ----- UNSCREENED (T011, T012) -----------------------------------------
    Candidate(
        id="C5_h96_t03_MAE",
        horizon=96, min_gain=0.03, basis=BASIS_GUARD, mae_bound=0.04,
        motivating_measurement=(
            "Path-aware variant of C3 (FR-006). At CURRENT settings an MAE bound is "
            "immaterial -- only 4.5% of breakout and 5.4% of gbb labels need a >4% "
            "drawdown before their peak -- but at tail settings the stopped-before-"
            "peak share rises 2.5x to 11.2%, so the constraint may earn its place "
            "there. Bound set at the tight edge of the ATR band."),
    ),
    Candidate(
        id="C6_bb_tail",
        horizon=96, min_gain=0.03, basis=BASIS_BB,
        motivating_measurement=(
            "Alternative trigger basis (FR-007) so the result is not specific to one "
            "label form: a band POSITION rather than an oscillator level. Screened "
            "lower on MCC (0.721 vs 0.789) with comparable EV and half the signal "
            "volume. NOTE bb_position runs on [-1, 1], NOT [0, 1]."),
    ),
]


def validate_all() -> None:
    ids = [c.id for c in CANDIDATES]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    for c in CANDIDATES:
        c.validate()


def admitted() -> list:
    return [c for c in CANDIDATES if c.admitted]


def unscreened() -> list:
    return [c for c in CANDIDATES if not c.screened]


if __name__ == "__main__":
    validate_all()
    print(f"{len(CANDIDATES)} candidates, {len(admitted())} admitted, "
          f"{len(unscreened())} awaiting screening\n")
    for c in CANDIDATES:
        state = "EXCLUDED" if c.excluded_reason else ("screened" if c.screened else "PENDING")
        hold = "" if c.requires_hold_change is None else (
            "  needs hold-window change" if c.requires_hold_change else "")
        print(f"  {c.id:<22} {state:<9} basis={c.basis}"
              f"{f'  mae<={c.mae_bound}' if c.mae_bound else ''}{hold}")
        if c.excluded_reason:
            print(f"      excluded: {c.excluded_reason}")
