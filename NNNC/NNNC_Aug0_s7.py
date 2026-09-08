# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_Aug0_s7 — label-dilation width 0 (feature: recall-vs-precision follow-up).

BaseNNStrategy.augment_training_signals "Trick 1" copies every buy label onto
the TWO PRECEDING bars, deliberately blurring the decision boundary to give the
classifier more positive examples. Those bars do NOT meet the label's own
condition.

WHY TEST THIS NOW. `regime/RECALL_NOT_PRECISION.md` measured that on NNNC's open
gate the model's bad picks cost almost nothing (-0.216pp) while it misses 1006
opportunities worth +63.7pp -- and that buying recall by lowering the threshold
costs -53.2pp in bad picks against +31.7pp recovered. **Precision is the
expensive axis.** Blurring the boundary is defensible when precision is cheap;
it is not obviously defensible here.

This arm sets the dilation width to 0 (no dilation at all -- only bars meeting the condition are positive).

IMPLEMENTATION. The dilation is masked rather than reimplemented: the parent's
augment_training_signals still runs in full, so "Trick 2" (sells override buys)
and "Trick 3" (a buy followed by a sell is overridden) are byte-identical. Only
buys OUTSIDE the permitted dilation window are cleared afterwards. At width 2
this reproduces the shipped behaviour exactly, which makes NNNC_Aug2 a control
that validates the mechanism rather than a re-run of the default.

`Framework/` is untouched -- a live dry run imports it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Open_s7 import NNNC_MLX_Open_s7


class NNNC_Aug0_s7(NNNC_MLX_Open_s7):

    AUG_DILATION = 0

    def augment_training_signals(self, buys, sells):
        orig = np.asarray(buys, dtype=float) > 0.5
        allowed = orig.copy()
        for i in range(1, int(self.AUG_DILATION) + 1):
            if i < len(orig):
                allowed[:-i] |= orig[i:]          # permit bar t-i when orig fires at t
        b, s = super().augment_training_signals(buys, sells)
        b = np.array(b, dtype=float, copy=True)   # super() may return a read-only view
        b[~allowed] = 0.0
        return b, s

    def get_model_path(self) -> str:
        name = self.__class__.__name__
        return self.get_storage_location() + name + "/" + name + ".keras"
