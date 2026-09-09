# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_Unguarded_s1 — UNGUARDED label. Seed 1.

The production label is a CONJUNCTION of three conditions:

    (guard_metric < -0.2) AND (bb_width > 0.035) AND (future_gain >= 0.02)

The first two duplicate quantities the ENTRY GUARD CHAIN already filters, and
the duplication is severe: on SOL, 15,056 bars pass the label but only 120
(0.8%) also survive the entry guards. `entry_guard_threshold` is entirely
REDUNDANT -- zero labelled bars fail it, because the label's -0.2 is strictly
tighter than the entry's +0.3 -- which is why that guard was measured as INERT
(byte-identical trades, 3/3 windows). `entry_bb_width_threshold` is worse than
redundant: at 0.06 it is TIGHTER than the label's 0.035 and discards 55.8% of
the label set.

So the model is trained on a population it can act in only 0.8% of, and its
learned discrimination is largely irrelevant to the bars it actually trades.

This arm removes the two guard conditions FROM THE LABEL and lets the entry
chain do the filtering it exists for. The label becomes, in effect,
"price rises >= 2% within the horizon without being stopped first" -- close to
`triple_barrier` in structure (Jaccard 61%; the residual difference is barrier
levels, gbb_path using stop_frac -0.05 against triple_barrier's -0.02).

Label population on SOL: 14,566 -> 82,230 bars (7.3% -> 41.3%), 5.6x more
positives. Those include the momentum run-ups the guarded label cannot see:
Aug 19/21/22/27 2026 all rose 8-12.6% within the horizon and none was labelled,
while the single labelled region rose a median 3.41%.

EXPECT A LOW val_mcc. A 41%-positive target is far harder to separate than a
7% one, and the metric is not the point -- this repository has recorded seven
interventions where the classification metric and realised P&L moved
independently. Judge on W1 realised return.

Implemented via LabelParamsMixin (no new labeler): relaxing guard_threshold to
+999 and bb_width_threshold to -999 makes both conditions vacuously true.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from LabelParamsMixin import LabelParamsMixin
from NNNC_MLX import NNNC_MLX


class NNNC_Unguarded_s1(LabelParamsMixin, NNNC_MLX):

    # Vacuously-true guard conditions => the label reduces to the gain test.
    LABEL_PARAMS = {"guard_threshold": 999.0, "bb_width_threshold": -999.0}
    LABEL_PARAMS_SELL = {"guard_threshold": -999.0, "bb_width_threshold": -999.0}

    def get_model_path(self) -> str:
        name = self.__class__.__name__
        return self.get_storage_location() + name + "/" + name + ".keras"
