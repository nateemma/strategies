# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GbbOpen — the `gbb` TRIGGER, traded with NO model. THE COLUMN-ADJUDICATION ARM.

This arm exists to eliminate one of three competing economic-value columns on evidence.
Feature 021 scored every trigger three ways over the same outcome table, and on `gbb` the
columns DISAGREE IN SIGN:

    column                                              gbb         breakout_vol
    board (fillable bars, per-pair non-overlap)      -0.002009       +0.000582
    concurrency-aware, UNFILTERED, equal-weighted    +0.001508       -0.001548
    stake-weighted by __outcome_fillable_fraction    -0.000921       +0.000279

`BreakoutVolOpen` already tested the right-hand column and returned -63.19pp
(-17.17pp/yr, 6/15 folds, 8,290 trades). There the UNFILTERED equal-weighted figure was
the only one that matched live (to 3.6e-5); the board and the stake-weighted column both
got the sign WRONG. But that is n=1, and the unfiltered column has a known defect -- it
weights a trade capped to a quarter of its intended stake the same as a full one.

So `gbb` is the discriminating case, and it discriminates in the opposite direction:
  * if this arm returns POSITIVE, the unfiltered equal-weighted column is vindicated twice
    and the board's column is wrong in both directions it was tested;
  * if it returns NEGATIVE, the board and the stake-weighted column are vindicated here,
    `breakout_vol` was the unfiltered column getting one sign right by luck, and the
    inversion really was the weighting artefact it looks like.
Either outcome kills one column. That is the point; neither is the hoped-for result.

WHY `gbb` SPECIFICALLY. It is also the incumbent label, so its trigger is the one with the
most external evidence -- but NOT evidence that transfers to this arm. The +39.98pp /
15-of-15 record belongs to `gbb` + 24 indicators + a per-fold threshold, volume-matched to
1.4735% of bars. This arm trades ALL ~13.3% of bars the trigger fires on, with no model.
Those populations differ by an order of magnitude; treating the incumbent's return as a
prediction for this arm was an error already made and retracted once in this feature.

THE ENTRY IS THE REGISTERED TRIGGER ITSELF, via
`research.outcomes.triggers.evaluate("gbb", ...)` -- the same function the leaderboard
scored, including its deliberately ASYMMETRIC NaN convention (a bar missing `guard_metric`
fires; a bar missing `bb_width` does not). A hand-copied condition would quietly normalise
that and stop being the thing measured. If the import fails this strategy REFUSES to trade.

EXIT POLICY: the Open operating point, unchanged (FR-019), so outcomes match the table.
Guards are OPEN. `entry_rvol_threshold` stays permissive because the `gbb` trigger carries
its own `rvol > 1.0` condition and the gate's rvol would filter it twice.

WHAT THIS CANNOT SHOW. The table's economic value is a lookahead bound (FR-017); only the
pooled harness in evaluation/ can make the P&L claim, and it is the harness number -- not
any column above -- that settles this.

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


class GbbOpen(TriggerOnlyStrategy):
    """`gbb` + the Open exit, no model, guards open."""

    TRIGGER_NAME = "gbb"
