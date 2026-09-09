# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_RvL20 — rvol added to the BUY LABEL at 2.0. Full alignment to NNNC's gate. Collapse risk HIGH.

The entry gate has always filtered `rvol > entry_rvol_threshold` (NNNC 2.0,
NNMT 0.2) while the LABEL never saw rvol, so the model was trained on a
population it can act in only a fraction of. This arm closes that gap.

Label population over 1,813,739 bars (committed gbb_path):
    no rvol  133,610 (7.37%)   >0.2 101,390 (5.59%)   >0.5 74,433 (4.10%)
    >1.0      45,630 (2.52%)   >2.0  19,765 (1.09%)
~5.5% is this repo's observed all-Hold collapse line, so full alignment to
NNNC's gate (2.0) is deliberately BELOW it -- that is a property of a gate
that passes 0.08% of bars, not of the parameter.

Judge on W1 realised return against NNNC_MLX's control, then W2/W3: three sweeps
this session produced W1 winners that reversed on W2, so a single-window gain
is not evidence.

This arm TRAINS ITS OWN MODEL (the label differs), so get_model_path is
deliberately NOT pinned -- the default class-name behaviour is correct here.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_RvL20(NNNC_MLX):

    LABEL_RVOL_THRESHOLD = 2.0
