# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""033 T018 instrument twin (GENERATED for the smoke check): `NNNC_WR_ARO` with flat stakes
(STAKE_WEIGHTS (1.0, 1.0)), so only the stake weights differ from the sizer. Its f00 trades_hash
must equal the width book's (`w_aro`). Same model key (STAKE_WEIGHTS is not a key input)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from NNNC_WR_ARO import NNNC_WR_ARO


class NNNC_WR_ARO_FLAT(NNNC_WR_ARO):
    STAKE_WEIGHTS = (1.0, 1.0)
