"""Deliverable 3's second half: EVERY buy-side LabelMethod is accounted for.

The deliverable asked to "express the existing LabelMethod family's conditions as triggers
where they reduce to one, AND REPORT WHICH DO NOT AND WHY". The first half was done (15
triggers registered, 2 declared unscoreable). The second half was not, which is why it
looked as though two-thirds of the label family had silently vanished from the study.

This file is that report, made executable. It fails if a new `labels_*` function appears in
`Framework/TrainingSignals.py` without being classified, so the inventory cannot go stale
the way a prose table in a spec would -- the same reasoning as
`Framework/tests/test_exit_policy_single_source.py`'s allowlist.

THE LOAD-BEARING CONCLUSION, which belongs in the code and not only in the write-up:
**this instrument can only distinguish labels that differ in their ENTRY condition.**
Eleven of the 29 buy-side labelers differ from one another ONLY in their outcome definition,
and the design replaces every outcome with one shared definition derived from the real exit.
So those eleven are not merely unscored -- under this design they are THE SAME LABEL, the
no-trigger case, and no amount of further work on this instrument will separate them.
"""
import io
import re

import pytest

TRAINING_SIGNALS = "Framework/TrainingSignals.py"

# --- A. expressed as a trigger and scored on the board ------------------------------------
# `bands` and `trends` are here because they WERE expressed; both are then declared
# UNSCOREABLE because the columns their conditions need are absent from the feature frame.
# Declaring them beats porting them at reduced fidelity, which produced a number that looked
# comparable and was not (an earlier `bands` port silently dropped 6 of 8 columns).
TRIGGER_SCORED = {
    "labels_gbb": "gbb",
    "labels_geometry": "geometry",
    "labels_breakout": "breakout",
    "labels_breakout_vol": "breakout_vol",
    "labels_breakout_squeeze": "breakout_squeeze",
    "labels_breakout_bb": "breakout_bb",
    "labels_breakout_gbb": "breakout_gbb",
    "labels_breakout_consensus": "votes_consensus",
    "labels_technical_indicators": "votes_tech",
    "labels_indicators2": "votes_ind2",
    "labels_indicators3": "votes_ind3",
    "labels_indicators4": "votes_ind4",
    "labels_trends": "trends (UNSCOREABLE: 4 regime-state columns absent)",
    "labels_bands": "bands (UNSCOREABLE: 6 of 8 band columns absent)",
}

# --- B. same trigger, different OUTCOME test -> collapses onto an existing row -------------
# These are not missing: their entry condition is already on the board and the only thing
# that differs is the forward test the design replaces. `labels_gbb_fwdmae` is the clearest
# case -- its body is literally `labels_gbb(...) & labels_forward_return_mae_cap(...)`.
SAME_TRIGGER_DIFFERENT_OUTCOME = {
    "labels_gbb_path": "gbb",
    "labels_gbb_fwdmae": "gbb",
    "labels_breakout_tb": "breakout",
}

# --- C. NO entry condition at all: a pure forward/outcome test -> all become `empty` -------
# THE CLASS THAT MATTERS. Each of these is a different answer to "what counts as a good
# outcome" -- forward MAE cap, triple barrier, forward Sharpe, Sortino, expectancy, a return
# quantile, an MFE/MAE ratio, a vote across horizons, the oracle, a forward-window local
# minimum. None carries an entry condition, so under one shared outcome they are all the
# SAME label: the no-trigger case. `labels_forward_return_mae_cap` is `forward_mae`, half of
# the feature's own motivating comparison (gbb +39.98pp vs forward_mae -0.83pp) -- which the
# board reproduces as "a trigger versus no trigger" by CONSTRUCTION rather than by
# measurement, and that distinction is the honest limit of this study.
OUTCOME_ONLY = {
    "labels_forward_return_mae_cap": "forward MAE-capped return (`forward_mae`)",
    "labels_triple_barrier": "triple barrier",
    "labels_quantile_future_return": "future-return quantile",
    "labels_mfe_mae_ratio": "MFE/MAE ratio",
    "labels_risk_adjusted_future_return": "risk-adjusted future return",
    "labels_future_sharpe": "forward Sharpe",
    "labels_future_sortino": "forward Sortino",
    "labels_future_expectancy": "forward expectancy",
    "labels_multi_horizon_vote": "vote over forward_return_mae_cap at 3 horizons",
    "labels_optimal_signals": "oracle over the forward window",
    "labels_local_extrema": "minimum of a FORWARD window (close[t:t+h])",
}

# --- D. has a condition that LOOKS entry-side but reads forward ---------------------------
# `close[t] == min(close[t-3 : t+4])` is CENTRED, so it peeks 3 bars ahead. A trigger must be
# computable at bar t from bar t's information, so this cannot be one. Legitimate in a label
# (labels may look ahead); simply not expressible as the entry rule this design varies.
NOT_A_TRIGGER_LOOKAHEAD = {
    "labels_local_min_followthrough": "centred +/-3-bar local minimum reads 3 bars ahead",
}

CLASSIFIED = {
    **TRIGGER_SCORED, **SAME_TRIGGER_DIFFERENT_OUTCOME,
    **OUTCOME_ONLY, **NOT_A_TRIGGER_LOOKAHEAD,
}


def _buy_side_labelers() -> set[str]:
    src = io.open(TRAINING_SIGNALS, encoding="utf-8").read()
    return {n for n in re.findall(r"^def (labels_[a-z_0-9]+)\(", src, re.M)
            if not n.endswith("_sell")}


def test_every_buy_side_labelmethod_is_classified():
    found = _buy_side_labelers()
    unclassified = sorted(found - set(CLASSIFIED))
    stale = sorted(set(CLASSIFIED) - found)
    msg = []
    if unclassified:
        msg.append(
            "New buy-side labelers are not classified in this file. Decide which class each "
            "belongs to -- do NOT just add it to a dict:\n  "
            + "\n  ".join(unclassified)
            + "\n  A. expresses an ENTRY condition -> register it as a trigger and score it.\n"
              "  B. an existing trigger with a different OUTCOME test -> record which row it\n"
              "     collapses onto.\n"
              "  C. a pure FORWARD/outcome test with no entry condition -> it is the\n"
              "     no-trigger case; record it, and note that this design cannot separate it\n"
              "     from the others in class C.\n"
              "  D. a condition that reads forward -> not expressible as a trigger; say which\n"
              "     bars it peeks at."
        )
    if stale:
        msg.append("Classified here but no longer in TrainingSignals.py:\n  "
                   + "\n  ".join(stale))
    assert not msg, "\n\n".join(msg)


def test_the_four_classes_are_disjoint_and_total():
    """A labeler in two classes would make the inventory's arithmetic meaningless."""
    dicts = [TRIGGER_SCORED, SAME_TRIGGER_DIFFERENT_OUTCOME, OUTCOME_ONLY,
             NOT_A_TRIGGER_LOOKAHEAD]
    total = sum(len(d) for d in dicts)
    assert total == len(CLASSIFIED), "a labeler appears in more than one class"
    assert total == len(_buy_side_labelers())


def test_the_scoreable_classes_point_at_REGISTERED_triggers():
    """Class A and B name trigger rows; those rows must actually exist in the registry, or
    the inventory is claiming coverage the board does not have."""
    from research.outcomes import triggers as T
    T.ensure_builtins()
    registered = set(T.names())
    for labeler, row in SAME_TRIGGER_DIFFERENT_OUTCOME.items():
        assert row in registered, f"{labeler} collapses onto absent trigger {row!r}"
    for labeler, row in TRIGGER_SCORED.items():
        name = row.split()[0]
        assert name in registered, f"{labeler} claims trigger {name!r}, not registered"


def test_class_C_is_the_studys_blind_spot_and_says_so():
    """The point of the inventory. Eleven labelers differ ONLY in their outcome definition,
    so this design -- one shared outcome, trigger as the only variable -- maps them all to
    the same label. That is not an omission to be fixed later; it is the instrument's reach.
    Asserted so the count cannot drift without someone re-reading the reasoning."""
    assert len(OUTCOME_ONLY) == 11
    assert "forward_mae" in OUTCOME_ONLY["labels_forward_return_mae_cap"]
    assert "ENTRY condition" in __doc__
