# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Ora_Patient — the JOINT BOUND (feature 006, T010). A LOOKAHEAD CEILING.

**This arm is a BOUND, never a candidate. It is not promotable under any
result** — it disables the stop entirely, which no live configuration may do.

It answers one question: of the distance between the real model and a perfect
entry, how much does the EXIT stack consume? Every exit mechanism is lifted out
of the way so the trade runs to the horizon its label actually promises:

  * `use_atr_adaptive_stoploss = False`  — the ATR stop is the live stop lever
    on this family (`self.stoploss` is inert while it is True), so it must be
    switched off at the class level. The runner then widens `--stoploss` too.
  * `HOLD_WINDOW_HOURS = 13`  — via the EXISTING HoldWindowMixin (feature 003),
    imported not rebuilt. This matters more than it looks: `unclog_12h` is
    hardcoded in `BaseStrategy.custom_exit` and fires at 12 hours, which is
    EXACTLY the label horizon (HORIZON 48 bars x 15m). Without deferring it the
    "patient" arm would be closed at the very moment its label matures. 13 hours
    clears it; `unclog_1d` at 24h and `max_hold` still bound the trade, so this
    cannot run away.
  * take-profit, ROI and max-days are lifted by the runner's flags.

Per research R5 this is deliberately a JOINT change — Constitution VIII is
waived here and only here, justified in plan.md's Complexity Tracking, because
the bound decides whether the per-mechanism attribution is worth running at all.
Every arm that could actually be ADOPTED moves exactly one thing.

Model weights are aliased through NNMT_Oracle, so this trains nothing, and it
inherits the 2026-09-07 augmentation fix — without which the entries would be up
to two bars early and the bound meaningless.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "NNNC"))

from _hold_window import HoldWindowMixin
from NNMT_Oracle_Lbl02 import NNMT_Oracle_Lbl02


class NNMT_Ora_Patient(HoldWindowMixin, NNMT_Oracle_Lbl02):

    # The ATR stop is the live lever; self.stoploss is inert while this is True.
    use_atr_adaptive_stoploss = False

    # Just past the 12-hour label horizon that unclog_12h would otherwise cut.
    HOLD_WINDOW_HOURS = 13.0
