import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Open import NNNC_MLX_Open


class NNNC_Open_Cap5(NNNC_MLX_Open):
    """LOOSER fill cap: QUOTE_VOLUME_HEADROOM_MULT 5 = take up to 20% of a
    candle's quote volume (default 10 = 10%).

    UNTESTED DIRECTION. The only prior sweep of this constant (momentum basket,
    2026-08) went TIGHTER and found that tightening improved PF and drawdown.
    Nobody has measured loosening.

    READ ANY GAIN HERE WITH SUSPICION. The backtest applies NO slippage model --
    entry_pricing uses order_book_top=1, so fills happen at the touch regardless
    of size. Taking 20% of a 15m alt candle at the touch price is not
    physical, so extra return at this setting is bought from an ASSUMPTION, not
    from edge.
    """

    QUOTE_VOLUME_HEADROOM_MULT = 5.0

    def get_model_path(self) -> str:
        return self.get_storage_location() + "NNNC_MLX_Lbl02/" + "NNNC_MLX_Lbl02.keras"
