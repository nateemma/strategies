# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""BreakoutVolOpen — the `breakout_vol` TRIGGER, traded with NO model.

Feature 021's board scored `breakout_vol` the strongest `selection_value` row with the
tightest interval of any non-withdrawn trigger: economic value **+0.00185 vs the `empty`
null, 95% CI [+0.00093, +0.00278]**, surviving calendar blocks at every length up to the
full 960-bar hold. This strategy is the only instrument that can say whether that TRADES.

WHY A NO-MODEL STRATEGY IS THE RIGHT TEST. Three measurements point here:
  * the model's selection inside `gbb` is reproducible by a plain threshold, and beaten by
    one 5.6x at matched selectivity (2026-10-09 diagnostic);
  * a conditional probe finds almost nothing predictable within any trigger's population
    (RIDGE +0.02..+0.05, LSTM -0.006 -- a non-linear model does no better);
  * this repository's standing findings: "guard ablation: model adds NOTHING" and
    "gate/mechanism move P&L, not model" (7 interventions, 2 families, every win gate-side).
So the hypothesis under test is that the trigger plus the shared exit IS the strategy.

THE ENTRY IS THE REGISTERED TRIGGER ITSELF, not a reimplementation of it.
`get_entry_conditions` calls `research.outcomes.triggers.evaluate("breakout_vol", ...)`, the
same function the leaderboard scored. A hand-copied `close > high.rolling(20).max().shift(1)
& rvol > 1.5` would be a second definition that could drift from the thing the measurement
was about -- exactly the duplication feature 021 exists to remove. If the import ever fails
this strategy REFUSES to trade rather than falling back to a copy, because a silent fallback
would make the arm untraceable to the row it is testing.

EXIT POLICY: the Open operating point, unchanged, so the trade outcomes are the ones the
outcome table simulated (`ExitConfig.from_operating_point`). Guards are OPEN (every quality
threshold non-binding) because the point is to measure the TRIGGER, not the trigger plus six
filters. `volume > 0` and `rvol` stay live -- they are tradeability constraints, not quality
filters, and `rvol > 1.5` is part of the trigger anyway.

WHAT THIS CANNOT SHOW. The table's economic value is a lookahead bound (FR-017). A positive
bound has twice failed to become harness return in this repository -- "trigger relaxation:
return FALLS; ceiling rises monotonically, return doesn't", and feature 003's rejected tail
labelling. A third such result is the single most likely outcome here and would be a real
finding, not a disappointment.

REFACTORED 2026-10-09 onto `TriggerOnly.TriggerOnlyStrategy`, which now holds the exit
block, the open guards and the entry path that every trigger-only arm shares. The
resolved class attributes were snapshotted BEFORE the change and compared after: this
arm's 15 harness folds are already recorded, so a refactor that altered any attribute
would have made those records irreproducible. Only TRIGGER_NAME lives here now.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class BreakoutVolOpen(TriggerOnlyStrategy):
    """`breakout_vol` + the Open exit, no model, guards open."""

    TRIGGER_NAME = "breakout_vol"
