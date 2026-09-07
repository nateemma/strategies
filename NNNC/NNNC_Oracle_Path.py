# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNNC_Oracle_Path — oracle-gate screen for the PATH-AWARE label on NNNC.

`LabelMethod.gbb_path` was built and screened on NNMT, where it PASSED the gate
(+23.991% guards-off W1 against the incumbent's +17.791%) and then produced a
NULL on realised return (-0.091pp). This arm asks the same question of NNNC,
whose family differs in ways that could matter:

  * NNNC is single-task (MLXClassifierNary), not multi-task.
  * Its entry gate is far TIGHTER — entry_rvol 2.0 against NNMT's 0.2 — and its
    guards were previously measured to reject ~97% of what the model would take.
  * Its take-profit is 0.024 and its hold window 10 days, against NNMT's 0.022
    and 21 days. The sell-side pre-emption rate depends directly on the
    take-profit, so the path constraint bites differently here.

Held FIXED against NNNC_Oracle so the labeler is the only difference. Both alias
NNNC_MLX's weights and neither trains.

NOTHING MEASURED WITH THIS CLASS IS PROMOTABLE.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_Oracle import NNNC_Oracle


class NNNC_Oracle_Path(NNNC_Oracle):

    # LabelMethod.gbb_path (27). The incumbent is gbb (17).
    TRAINING_TYPE = 27
