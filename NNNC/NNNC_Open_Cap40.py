import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Open import NNNC_MLX_Open


class NNNC_Open_Cap40(NNNC_MLX_Open):
    """CAPACITY probe: QUOTE_VOLUME_HEADROOM_MULT 40 (default 10).

    `custom_stake_amount` caps the stake at 1/MULT of the candle's quote volume,
    so raising MULT is equivalent to trading a proportionally LARGER book against
    the same market: MULT 20 ~ double the wallet, MULT 40 ~ quadruple it. This is
    the capacity curve without touching config or retraining.

    Model is aliased so this differs from its parent in the cap ALONE --
    get_model_path keys off the class name, and a renamed strategy with no model
    of its own trains one inside the backtest.
    """

    QUOTE_VOLUME_HEADROOM_MULT = 40.0

    def get_model_path(self) -> str:
        return self.get_storage_location() + "NNNC_MLX_Lbl02/" + "NNNC_MLX_Lbl02.keras"
