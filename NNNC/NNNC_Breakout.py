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
    # Use the INVERSE-GBB breakout (guard_metric HIGH + bb_width + follow-through) —
    # a FEATURE-based breakout that screens nearly as learnable as gbb (MCC 0.73-0.77)
    # with comparable/better EV (2.96-4.15%), robust SOL/LINK/XRP/BCH. The crude
    # price-based labels_breakout (20) was far less learnable (see
    # feedback_breakout_labels_unlearnable).
    TRAINING_TYPE = int(LabelMethod.breakout_gbb)   # 24
    MIN_BUY_GAIN_THRESHOLD = 0.005                  # MEDIUM aug_risk, MCC ~0.77
    MIN_SELL_LOSS_THRESHOLD = 0.005

    # --- neutralise the dip guards (breakouts have HIGH close_norm/guard_metric);
    #     keep volume/rvol/atr/adx/bb_width guards, which confirm real breakouts ---
    # NNNC_Breakout has no <Strategy>.json, so it inherits the RAW un-tuned
    # BaseStrategy guard defaults (entry_bb_width_threshold=0.094,
    # entry_atr_pct=0.013) which pass only ~1% of bars each and ~0% combined —
    # they annihilate the (dense) predict_buy signal. Relax the volatility/volume
    # guards to permissive values so the model + BTC gate are the real filters;
    # keep a light rvol confirmation (breakouts want volume). The two dip guards
    # (close_norm/guard_metric) stay neutralised.
    buy_params = {
        **NNNC_MLX.buy_params,
        "entry_close_norm_threshold": 5.0,
        "entry_guard_threshold": 5.0,
        "entry_bb_width_threshold": 0.0,     # was 0.094 (1% pass) — neutralised
        "entry_atr_pct": 0.0,                # was 0.013 (1% pass) — neutralised
        "entry_rvol_threshold": 1.0,         # was 1.9 — light volume confirmation
        "prediction_threshold": 0.15,
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
