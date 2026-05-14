# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# type: ignore

"""
NNNC_MLX_HighPrecision — same classifier, same training data, raised decision
threshold to test whether the high-`gan_target_ratio` effect can be reproduced
by a direct precision/recall threshold shift instead of via training noise.

Background (2026-05-14):
  - NNNC_MLX (no GAN) — val_precision ≈ 0.34, backtest -63% (market -32%).
  - NNNC_DDPM_MLX_LSTM_MT at gan_target_ratio=0.8 — val_precision ≈ 0.49,
    backtest -10% (above market by +21pp).
  - Diagnostic confirmed the GAN's output quality at ratio=0.8 was identical
    to ratio=0.1 — only the volume changed. 65% of training data became synth
    with broken joint correlations and zero hold-class samples, which made
    the classifier converge to a more conservative precision/recall regime
    (130+ grad clips per epoch — substantial training instability).
  - Hypothesis: the trading P&L improvement is from threshold shift, not
    GAN signal quality. If true, raising prediction_threshold directly on
    the no-GAN classifier should reproduce most of the effect with none of
    the GAN-pipeline complexity.

Test plan:
  1. Run this strategy at prediction_threshold=0.6 (this file's default).
  2. If it improves over NNNC_MLX, try 0.55 and 0.65 to bracket the sweet spot.
  3. If even 0.6 falls well short of the -10% backtest result, the GAN aug is
     contributing real signal beyond the threshold shift, and the full pipeline
     (with high gan_target_ratio) is justified.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from freqtrade.strategy import DecimalParameter


class NNNC_MLX_HighPrecision(NNNC_MLX):
    # The prediction_threshold is a DecimalParameter on BaseStrategy with a
    # default of 0.4. Raising the LOWER BOUND forces a more conservative
    # default (preserved at backtest time when no hyperopt JSON exists), and
    # raising the default itself replaces the inherited 0.4.
    prediction_threshold = DecimalParameter(
        0.50, 0.80, default=0.60, decimals=2, space="buy", optimize=True, load=True
    )
