# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error

"""
NNNC_BreakoutOpen — NNNC_MLX trained on the INVERSE-GBB BREAKOUT label, run with
the entry guards DISABLED and the model's exit signal ENABLED.

WHY THE GUARDS COME OFF. The shared guard chain is built for a dip-buyer:
`guard_metric < threshold` (buy weakness) and `close_norm < threshold`. A
breakout label fires where `guard_metric` is HIGH, so those two guards reject
essentially every bar the model wants. The previous attempt (`NNNC_Breakout`)
worked around this by setting them to absurd pass-through values (5.0), which
leaves a chain of guards nominally "on" and easy to misread. Here the chain is
switched OFF outright via `entry_enable_guards`, so the model IS the entry rule
and nothing silently filters it.

WHY EXITS ARE ON. With `enable_exit_signal = False` the network has no say in
exits at all. Measured 2026-09-03 on NNMT, enabling it is worth +10.4pp at an
OPEN gate against +0.03pp at a tight one — the exit signal only matters when the
gate is not already doing the selecting. An open-gate breakout arm is exactly
that case.

LABEL PARAMETERS ARE PINNED, NOT INHERITED. `TrainingConfig` supplies
HORIZON=48 / threshold=0.007, both tuned for the *gbb* label by its own
H x threshold viability sweep. Inheriting them here would silently apply
gbb's tuning to a different label. All three are declared explicitly below.
`guard_threshold` (0.2) and `bb_width_threshold` (0.035) are NOT passed by
`BaseNNStrategy` — it sends only horizon/min_gain/min_loss — so those correctly
fall back to `labels_breakout_gbb`'s own defaults. Note the sign flip that makes
this a breakout at all: `labels_gbb` buys `guard_metric < -0.2`,
`labels_breakout_gbb` buys `guard_metric > +0.2`.

SELL LABEL IS MEAN-REVERSION, NOT THE MIRROR. The default sell for method 24 is
`labels_breakout_gbb_sell`, which sells when `guard_metric` is LOW — i.e. sell
the dip, which is wrong-signed for a momentum long. Method 17's sell
(`labels_gbb_sell`, overbought + future down = sell the rip) is the correct exit
direction. This is recorded as the single biggest improvement found in the
earlier breakout work (-3.07% -> -1.38%) and is the one component of it flagged
as worth reusing.

NO BTC GATE. `NNNC_Breakout` enabled BtcRegimeGate uptrend mode; that gate is
recorded as helping neither family and is not repeated here.

STATUS: breakout was previously taken to a FINAL VERDICT of "no tradeable long
edge at any timeframe with any exit", best result -1.11%. That verdict was
reached entirely under the native-trailing-stop defect fixed on 2026-09-02,
which cut ~87% of stopped trades above the configured stop level — and the
verdict's core argument was about stops outnumbering winners. This arm re-tests
it on the repaired stack. It is an experiment, not a promotion candidate.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX import NNNC_MLX
from Framework.TrainingSignals import LabelMethod


class NNNC_BreakoutOpen(NNNC_MLX):

    # --- label: inverse-gbb breakout, all parameters PINNED ------------------
    # 24 = breakout_gbb: guard_metric > +0.2 AND bb_width > 0.035 AND
    # future_gain >= min_gain. Feature-based, so it screens nearly as learnable
    # as gbb (MCC 0.73-0.77) unlike the price-based Donchian variant (0.63-0.67).
    TRAINING_TYPE = int(LabelMethod.breakout_gbb)

    # Pinned so a change to TrainingConfig (which is tuned for gbb) cannot
    # silently retune this label. 0.005 keeps aug_risk out of HIGH for this
    # label; H=48 is the window at which breakout_gbb screened green at 15m.
    HORIZON = 48
    MIN_BUY_GAIN_THRESHOLD = 0.005
    MIN_SELL_LOSS_THRESHOLD = 0.005

    # --- entry: dip guards OFF, QUALITY guards ON ---------------------------
    # CORRECTED 2026-09-03. This first read "entry_enable_guards": False, which
    # disables all SEVEN guards. Only two of them -- guard_metric and close_norm
    # -- are dip-directional and genuinely incompatible with a breakout. The
    # other five (volume, rvol, adx, atr_pct, bb_width) are NON-DIRECTIONAL
    # quality filters, and discarding them cost the whole edge:
    #
    #   guards OFF (all 7)    4749 trades, summed -17.55%, avgWin 1.63%
    #   dip guards only        291 trades, summed  +0.98%, avgWin 2.99%
    #
    # avgWin nearly doubled against a median labelled 12h gain of 2.87%, i.e.
    # the quality guards select bars with ABOVE-median follow-through -- they
    # are not merely a volume throttle. That moved the arm from 3.8pp BELOW its
    # own break-even win rate to 1.2pp above it.
    #
    # The two dip guards are neutralised by threshold rather than by disabling
    # the chain: both conditions are `x < threshold`, so a large value passes
    # everything while leaving the other five active.
    buy_params = {
        **NNNC_MLX.buy_params,
        "entry_enable_guards": True,
        "entry_guard_threshold": 5.0,        # neutralised (dip-directional)
        "entry_close_norm_threshold": 5.0,   # neutralised (dip-directional)
        "prediction_threshold": 0.50,        # best of a 0.20-0.69 sweep
    }

    # --- exit: the model participates ---------------------------------------
    sell_params = {
        **NNNC_MLX.sell_params,
        "enable_exit_signal": True,
    }

    # Dip-buyer trend filter is meaningless for a breakout arm; keep it off.
    entry_trend_filter_enable = False
