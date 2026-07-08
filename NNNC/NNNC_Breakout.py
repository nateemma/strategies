# type: ignore
# pylint: disable=import-error
"""NNNC_Breakout — NNNC trained on BREAKOUT (momentum) labels instead of the
gbb dip-buyer, with the BTC-uptrend state filter now ALIGNED (breakouts buy
strength, so the filter helps rather than fights — cf.
feedback_btc_uptrend_gate_incompatible).

Three coupled changes vs NNNC_MLX:
  1. TRAINING_TYPE = breakout (LabelMethod 20): labels = close breaks a new
     lookback-high AND follows through. Learnable (MCC ~0.66-0.70) but sparser
     than gbb -> use a lower gain threshold to keep aug_risk out of HIGH.
  2. Neutralise the two DIP entry guards (close_norm<0.6 / guard_metric<0.1)
     that would otherwise block every breakout; keep volume/rvol/atr/adx guards
     (which confirm real breakouts).
  3. BtcRegimeGate uptrend mode — only take breakouts with a BTC tailwind.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
from NNNC_MLX import NNNC_MLX
from utils.BtcRegimeGate import BtcRegimeGate
from Framework.TrainingSignals import LabelMethod


class NNNC_Breakout(NNNC_MLX, BtcRegimeGate):

    # --- label change: momentum breakouts, not dips ---
    TRAINING_TYPE = int(LabelMethod.breakout)   # 20
    MIN_BUY_GAIN_THRESHOLD = 0.003              # keeps aug_risk MEDIUM, MCC ~0.70
    MIN_SELL_LOSS_THRESHOLD = 0.003

    # --- neutralise the dip guards (breakouts have HIGH close_norm/guard_metric);
    #     keep volume/rvol/atr/adx/bb_width guards, which confirm real breakouts ---
    buy_params = {
        **NNNC_MLX.buy_params,
        "entry_close_norm_threshold": 5.0,
        "entry_guard_threshold": 5.0,
    }

    # --- BTC-uptrend state filter, now aligned with a momentum strategy ---
    btc_gate_enable = True
    btc_gate_mode = "uptrend"

    def informative_pairs(self):
        return list(super().informative_pairs() or []) + self.btc_informative_pairs()

    def add_additional_indicators(self, dataframe):
        dataframe = super().add_additional_indicators(dataframe)
        return self.add_btc_regime(dataframe)

    def get_entry_conditions(self, dataframe):
        c = super().get_entry_conditions(dataframe)
        return c if c is None else c & self.btc_favorable_mask(dataframe)
