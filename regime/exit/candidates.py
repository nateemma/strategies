"""Candidate exit interventions (T003, T009-T011).

Every candidate carries the MEASUREMENT that motivates it (FR-009). A candidate
with no supporting measurement is not a candidate - that is the rule that keeps
this list from becoming a list of plausible ideas.

Candidates research already settled are PRE-POPULATED as excluded with their
reasons (FR-011), so none is quietly re-tested. Four of nine exit-side settings
do nothing at all, so a candidate touching one would produce a guaranteed null
that reads like a finding (FR-027).
"""
from __future__ import annotations

from dataclasses import dataclass

from regime.exit.baseline import BASELINE

ADJUST = "adjust_existing"
DORMANT = "enable_dormant"          # a change of kind, not degree - higher risk


@dataclass
class Candidate:
    name: str
    motivating_measurement: str      # REQUIRED, non-empty (FR-009)
    kind: str                        # ADJUST | DORMANT (FR-012)
    requires_retrain: bool = False   # MUST be False (FR-010, FR-025)
    excluded_reason: str = ""        # non-empty => never tested
    needs_framework_edit: bool = False   # edit-measure-revert only (FR-029)

    def __post_init__(self):
        if not self.motivating_measurement.strip():
            raise ValueError(f"{self.name}: a candidate MUST cite a measurement")
        if self.requires_retrain:
            raise ValueError(f"{self.name}: retraining is out of scope")
        if self.kind not in (ADJUST, DORMANT):
            raise ValueError(f"{self.name}: bad kind {self.kind!r}")

    @property
    def excluded(self) -> bool:
        return bool(self.excluded_reason)


# ---------------------------------------------------------------------------
# EXCLUDED - settled by research, never re-tested (FR-011)
# ---------------------------------------------------------------------------

EXCLUDED = [
    Candidate(
        "grace_level",
        "initial_stop_loss_abs is -5.00% on all 167 baseline trades, never the "
        "-0.27 grace level",
        ADJUST,
        excluded_reason=(
            "INERT, confirmed by test rather than inference: setting "
            "_ft_stop_uses_after_fill = True - the one-line enablement - gives "
            "BYTE-IDENTICAL results across all three windows (51/96/20 trades, "
            "identical hashes, initial stop still -5.00%). Inert more deeply "
            "than the flag."),
    ),
    Candidate(
        "ratchet_to_none",
        "the grace branch always returns 1.0, which freqtrade applies as a "
        "stoploss and ratchets upward; None is the documented sentinel",
        ADJUST,
        needs_framework_edit=True,
        excluded_reason=(
            "REFUTED at -9.24pp (summed 11.43% -> 2.19%), worse in all three "
            "windows, W3 negative. Under the change ROI exits collapse 36 -> 1 "
            "while take-profit rises 42 -> 97 at a LOWER mean. NOTE this refutes "
            "DELETING the mechanism only - see ratchet_explicit below."),
    ),
    Candidate(
        "trailing_params_current_arch",
        "trailing_stop_positive 0.01 vs 0.03 gives byte-identical backtests",
        ADJUST,
        excluded_reason=(
            "INERT while use_custom_stoploss = True. NOT dead as an option - see "
            "native_trailing below, which switches the architecture instead."),
    ),
    Candidate(
        "hyperopt_exit_config",
        "a parameter search proposed grace 4.3h / take-profit 0.005",
        ADJUST,
        excluded_reason=(
            "REJECTED on measurement: wallet -1.04 / -0.13 / -1.24pp and Calmar "
            "+6.17 / +0.08 / -11.63 - the sign reverses in the distant window. "
            "Its higher win rate comes from harvesting at +0.5%, converting "
            "would-be losers into small winners, which is why take-profit pinned "
            "to the floor of its range."),
    ),
]

# ---------------------------------------------------------------------------
# LIVE - each cites its motivating measurement (FR-009, SC-005)
# ---------------------------------------------------------------------------

LIVE = [
    # -----------------------------------------------------------------------
    # REPRIORITISED after Phase 1 (T005-T008). The attribution changed the
    # ranking: the ratchet accounts for 72 of 83 stopped trades (87%) -- 31/31
    # of the profitable stops and 41/52 of the losing ones were cut ABOVE the
    # -5% static stop. Only 11 of 83 ever reached the configured stop level.
    # Everything else on this list is marginal next to that.
    # -----------------------------------------------------------------------
    Candidate(
        "ratchet_explicit",
        "THE dominant lever: the ratchet cut 72 of 83 stopped trades (87%) -- "
        "100% of the profitable stops and 79% of the losing ones fired above the "
        "-5% static stop. It has no tunable distance (the constant 1.0), and it "
        "works via two uncontracted freqtrade behaviours. Phase 1 says both "
        "populations want MORE ROOM, which is exactly what a distance parameter "
        "provides",
        ADJUST,
        needs_framework_edit=True,
    ),
    Candidate(
        "native_trailing",
        "freqtrade's own trailing stop carries an OFFSET - do not trail until up "
        "X% - which is precisely the missing control. The profitable stops fire "
        "at a stop sitting +2.92% above entry after a median 0.50h; an offset "
        "would stop the trail engaging that early. Reached by setting "
        "use_custom_stoploss = False, untested because trailing_* is inert while "
        "it is True",
        ADJUST,
        needs_framework_edit=True,
    ),
    Candidate(
        "ratchet_cap",
        "a bounded variant of the ratchet: 100% of profitable stops fire with the "
        "stop ABOVE entry (median +2.92%), so capping how far it may rise -- or "
        "forbidding it above entry until a profit threshold -- targets the 31 "
        "trades that gave up a median +11.19% of further upside",
        ADJUST,
        needs_framework_edit=True,
    ),
    Candidate(
        "grace_duration",
        "losing stops cluster JUST PAST the 5.6h boundary (p25/p50/p75 = "
        "5.75/6.12/8.31h). Non-obvious direction: because freqtrade only "
        "tightens, the elevated ratchet stop PERSISTS after grace expires, so a "
        "SHORTER grace freezes the ratchet earlier and may protect, while a "
        "longer one lets it keep climbing. Worth testing both ways",
        ADJUST,
    ),
    Candidate(
        "time_based_exit",
        "\"more time\" is literally the Phase 1 finding -- 63.5% of losing stops "
        "reach ROI within 24h - and the unclog path currently forces an exit at "
        "12h. Low ceiling though: only 6 of 167 exits take this path today",
        ADJUST,
    ),
    Candidate(
        "roi_ladder",
        "DEMOTED by Phase 1. ROI fires 36 times at exactly +3.00%, but the "
        "problem is trades being cut BEFORE they reach it, not ROI firing "
        "wrongly. Fixing the ratchet is upstream of this; a ladder only matters "
        "for trades that survive to collect it",
        ADJUST,
    ),
    Candidate(
        "take_profit_upward",
        "DEMOTED, and possibly harmful. Raising take-profit keeps those 42 trades "
        "open LONGER under the ratchet that is already cutting 87% of stops, so "
        "it may convert take-profit exits into ratchet exits. Test only after the "
        "ratchet is fixed",
        ADJUST,
    ),
    Candidate(
        "enable_model_exits",
        "COUNTER-INDICATED by Phase 1. The finding is that the strategy exits too "
        "EAGERLY -- losing stops recover 63.5% of the time and profitable ones "
        "give up +11.19%. Adding the model's exit signal adds MORE exit pressure, "
        "the opposite direction. Retained only so the evidence against it is "
        "recorded; it is also the sole enable_dormant candidate and therefore the "
        "highest-risk one",
        DORMANT,
    ),
]

ALL = EXCLUDED + LIVE


def validate() -> None:
    """Enforce the registry invariants (FR-010, FR-012, FR-027)."""
    names = [c.name for c in ALL]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    inert = set(BASELINE.inert)
    for c in LIVE:
        # A live candidate must not target a setting measured as doing nothing.
        if c.name in ("grace_level",) or any(
            tok in c.name for tok in ("trailing_params", "exit_guard", "exit_close_norm")
        ):
            raise ValueError(f"{c.name}: targets an INERT setting - {sorted(inert)}")
    if not LIVE:
        raise ValueError("no live candidates")


if __name__ == "__main__":
    validate()
    print(f"EXCLUDED ({len(EXCLUDED)}) - settled by research, never re-tested:\n")
    for c in EXCLUDED:
        print(f"  {c.name}")
        print(f"     why excluded : {c.excluded_reason}")
        print()
    print(f"LIVE ({len(LIVE)}) - each cites its motivating measurement:\n")
    for c in LIVE:
        flag = "  [needs Framework edit: revert after each arm]" if c.needs_framework_edit else ""
        print(f"  {c.name}  ({c.kind}){flag}")
        print(f"     motivated by : {c.motivating_measurement}")
        print()
