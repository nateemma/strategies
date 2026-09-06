import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02


class NNNC_MLX_Open_s1(NNNC_MLX_Lbl02):
    """Seed-robustness replicate of NNNC_MLX_Open at MLX_SEED=1.

    Inherits the LABEL config (gbb, gain 0.02, H=48); the guards-open /
    threshold-0.80 operating point is pinned in NNNC_MLX_Open_s1.json, identical to
    NNNC_MLX_Open.json.

    Deliberately does NOT override get_model_path -- this arm must train its OWN
    weights at its own seed, which is the entire point. Train with
    `MLX_SEED=1 test_strat.sh -f ...`.

    Baseline replicate is NNNC_MLX_Open itself at the default MLX_SEED=42:
    W1 431tr +6.94%, W2 763tr +16.91%, W3 295tr +10.25%, summed +34.10%.
    """
