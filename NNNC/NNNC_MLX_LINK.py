# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC_MLX_LINK — single-pair (LINK) variant of NNNC_MLX. Feature 008.

IDENTICAL to NNNC_MLX in gate, exit stack, label and architecture. The ONLY difference is the
pair universe, supplied by config/config_link.json. The body is deliberately empty: any override
here would make the comparison measure two things at once.

Its own model directory comes for free -- BaseNNStrategy.get_model_path keys off
self.__class__.__name__, so this class cannot overwrite NNNC_MLX's weights.

It carries its own NNNC_MLX_LINK.json, a FULL copy of NNNC_MLX.json. That is not optional: the
tuned gate lives in the FILE and is NOT inherited (the class buy_params it merges from have every
guard OPEN), and a buy-ONLY file silently reverts the exit stack to BaseStrategy defaults, which
was measured at -2.4pp on 2026-09-09.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX


class NNNC_MLX_LINK(NNNC_MLX):
    pass
