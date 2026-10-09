# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GeometryOpen — the `geometry` TRIGGER, traded with NO model. NEVER TESTED BEFORE.

The DWT-smoothed zero-lag geometric valley condition implemented in `labels_geometry`
and, per feature 021's own spec, NEVER TESTED. Deliverable 3 named it explicitly for
that reason, so leaving it unrun would leave the feature's own brief unfinished.

Board: EV (fillable) -0.001526, CI [-0.001425, +0.000922], verdict `indeterminate`,
base rate 2.87%. Two things make it worth an arm despite the null verdict:
  * it is the trigger whose sign the signal-bar leak FLIPPED (+0.001210 paired at `i`,
    -0.001526 paired at `i+1`) -- the only one that changed direction, so it is the most
    sensitive row on the board to the pairing being right;
  * it is the counter-example that keeps the EV/gapfill confound from being stronger
    (lowest gapfill of any real trigger at 25.6%, yet near-worst EV), so it tests whether
    that confound or the EV column itself is the better predictor.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class GeometryOpen(TriggerOnlyStrategy):
    """`geometry` + the Open exit, no model, guards open."""

    TRIGGER_NAME = "geometry"
