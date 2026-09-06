# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Oracle_Lbl02 — the ceiling at the INHERITED label (0.02). A LOOKAHEAD BOUND.

Companion to NNMT_Oracle, which measures the ceiling at the PRE-INHERITANCE
label (0.007). Together they answer FR-005 — "measure the ceiling at both the
previously configured and the newly inherited label setting" — and they answer
it for FREE, because an oracle replaces the model's entry opinion with the
training label and therefore never consults the model at all. No retraining is
needed to compare what the two labels COULD pay (Constitution VII: validate
cheaply before building).

What this does NOT answer: whether a model can LEARN the better label. A ceiling
bounds what a label can pay, not what a network will realise of it — on NNNC the
model captured under 4% of a large ceiling gain. That question needs training,
and is only worth paying for if the ceiling separates the two labels first.

The profit head is deliberately NOT pinned here. Pinning exists to keep the
TRAINED arm single-variable (Constitution VIII), and T013 established the profit
head carries loss weight 0.0 and is untrained in any case. An oracle trains
nothing, so there is nothing to isolate: this arm measures the label exactly as
a retrain at 0.02 would construct it.

NOTHING MEASURED WITH THIS CLASS IS PROMOTABLE.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_Oracle import NNMT_Oracle


class NNMT_Oracle_Lbl02(NNMT_Oracle):

    # The inherited value. NNMT_Oracle runs the pre-inheritance 0.007.
    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
