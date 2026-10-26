# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNMT_MLX -- the NNMT family's default model since 2026-10-26 (user decision): NNMT_LR_RI, i.e. a plain
shared LSTM (ARO's design, a dense decode head per task) on NNMT's 24 indicators + raw8, at ARO's operating
point (guards open, prediction threshold 0.80), with NNMTStrategy's defaults (24 h grace, tr_scale).

040 evidence (research/nnmt/report.md): vs the old network at the same operating point it is ahead in both
halves on both seeds (9.48 / 13.75 vs 8.55 / 10.04; paired UNCHANGED), every adopted NNMT result (g24,
tr_scale) was measured on it, and it trains much faster. A subclass with nothing changed, so it has
NNMT_LR_RI's model key and shares its weights. The pre-2026-10-26 network is NNMT_MLX_Conv.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_LR_RI import NNMT_LR_RI


class NNMT_MLX(NNMT_LR_RI):
    pass
