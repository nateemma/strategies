# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GbbGuardedOpen — the `gbb` trigger PLUS the tuned quality guards, still NO model.

THE QUESTION. Six trigger-only arms established that no trigger trades on its own: five
lost 21-71pp and `gbb` was flat (+2.64pp, ~+0.7pp/yr, below the harness floor). Yet
`gbb` + 24 indicators + a per-fold threshold earns +39.98pp in 15/15 folds. Something
between the bare trigger and the incumbent carries the return. This arm removes the model
and keeps the guards, so whatever it earns is attributable to the guard chain alone.

WHY THIS IS WORTH A RUN -- the repository has already measured it from the other side.
`Framework/BaseStrategy.py:290` records that when these guards were tuned, "the guard chain
selected ~99% of entries, which made the model irrelevant: a PERFECT model was worth only
0.68pp more than the real one, so no model, GAN or architecture work could ever pay." If
that is right, this arm with NO model should land near the guarded-and-modelled result,
because the model was never doing the selecting. It is the direct test of
"guards bind, not the z-gate" and "gate/mechanism move P&L, not model".

WHAT IT IS *NOT*, and this correction matters. It is NOT a reconstruction of the incumbent.
The incumbent's +39.98pp was measured GUARDS-OPEN (spec.md:13) -- every quality threshold
neutralised -- with its 1.4735% selectivity coming from the MODEL's per-fold threshold, not
from the guards. So "guards + threshold cut 14.19% of bars to 1.47%" was wrong: the guards
were open in the incumbent too. This arm therefore tests a DIFFERENT operating point, and
the informative comparison is against the guards-OPEN trigger-only arm:

    gbb, guards OPEN,  no model   +2.64pp   7/15   (GbbOpen)
    gbb, guards TUNED, no model   ?                (this arm)

THE GUARD VALUES are BaseStrategy's own pre-2026-09-06 tuned block
(`Framework/BaseStrategy.py:257-267`), used verbatim rather than re-tuned, because a tuned
setting of my choosing would measure my tuning rather than the guard chain this repository
actually ran. Note `entry_rvol_threshold` 1.9 sits on top of the `gbb` trigger's own
`rvol > 1.0`, so rvol is filtered twice here -- that is a property of the configuration
under test, not a defect, and it is why this arm does not correspond to a leaderboard row.

EXIT: the Open operating point, unchanged (FR-019), identical to all six trigger-only arms
so the comparison above is exit-for-exit.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class GbbGuardedOpen(TriggerOnlyStrategy):
    """`gbb` + the tuned quality guards, no model."""

    TRIGGER_NAME = "gbb"

    #: BaseStrategy's pre-2026-09-06 tuned guard block, verbatim. `entry_enable_guards` stays
    #: True for the same reason it does in the base: the flag also gates the `volume > 0`
    #: data-sanity check, and 34.26% of bars in this dataset have volume <= 0.
    buy_params = {
        **TriggerOnlyStrategy.buy_params,
        "entry_enable_guards": True,
        "entry_adx_threshold": 10.0,
        "entry_atr_pct": 0.013,
        "entry_bb_width_threshold": 0.094,
        "entry_close_norm_threshold": 0.6,
        "entry_guard_threshold": 0.1,
        "entry_rvol_threshold": 1.9,
    }
