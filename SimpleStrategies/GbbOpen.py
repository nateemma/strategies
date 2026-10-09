# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GbbOpen — the `gbb` TRIGGER, traded with NO model. THE COLUMN-ADJUDICATION ARM.

This arm exists to eliminate one of three competing economic-value columns on evidence.
Feature 021 scored every trigger three ways over the same outcome table, and on `gbb` the
columns DISAGREE IN SIGN:

    column                                              gbb         breakout_vol
    board (fillable bars, per-pair non-overlap)      -0.002009       +0.000582
    concurrency-aware, UNFILTERED, equal-weighted    +0.001508       -0.001548
    stake-weighted by __outcome_fillable_fraction    -0.000921       +0.000279

`BreakoutVolOpen` already tested the right-hand column and returned -63.19pp
(-17.17pp/yr, 6/15 folds, 8,290 trades). There the UNFILTERED equal-weighted figure was
the only one that matched live (to 3.6e-5); the board and the stake-weighted column both
got the sign WRONG. But that is n=1, and the unfiltered column has a known defect -- it
weights a trade capped to a quarter of its intended stake the same as a full one.

So `gbb` is the discriminating case, and it discriminates in the opposite direction:
  * if this arm returns POSITIVE, the unfiltered equal-weighted column is vindicated twice
    and the board's column is wrong in both directions it was tested;
  * if it returns NEGATIVE, the board and the stake-weighted column are vindicated here,
    `breakout_vol` was the unfiltered column getting one sign right by luck, and the
    inversion really was the weighting artefact it looks like.
Either outcome kills one column. That is the point; neither is the hoped-for result.

WHY `gbb` SPECIFICALLY. It is also the incumbent label, so its trigger is the one with the
most external evidence -- but NOT evidence that transfers to this arm. The +39.98pp /
15-of-15 record belongs to `gbb` + 24 indicators + a per-fold threshold, volume-matched to
1.4735% of bars. This arm trades ALL ~13.3% of bars the trigger fires on, with no model.
Those populations differ by an order of magnitude; treating the incumbent's return as a
prediction for this arm was an error already made and retracted once in this feature.

THE ENTRY IS THE REGISTERED TRIGGER ITSELF, via
`research.outcomes.triggers.evaluate("gbb", ...)` -- the same function the leaderboard
scored, including its deliberately ASYMMETRIC NaN convention (a bar missing `guard_metric`
fires; a bar missing `bb_width` does not). A hand-copied condition would quietly normalise
that and stop being the thing measured. If the import fails this strategy REFUSES to trade.

EXIT POLICY: the Open operating point, unchanged (FR-019), so outcomes match the table.
Guards are OPEN. `entry_rvol_threshold` stays permissive because the `gbb` trigger carries
its own `rvol > 1.0` condition and the gate's rvol would filter it twice.

WHAT THIS CANNOT SHOW. The table's economic value is a lookahead bound (FR-017); only the
pooled harness in evaluation/ can make the P&L claim, and it is the harness number -- not
any column above -- that settles this.
"""
import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent.parent))

from Framework.BaseStrategy import BaseStrategy

#: The trigger name on feature 021's leaderboard. Kept as a constant so the arm, the row and
#: this strategy are provably the same object.
TRIGGER_NAME = "gbb"


class GbbOpen(BaseStrategy):
    """`breakout_vol` + the Open exit, no model, guards open."""

    # ---- no model, no training -------------------------------------------------------------
    # BaseStrategy's default StrategyConfig already has model_type NONE and
    # needs_training False, so nothing is fitted and there is no per-fold artefact. That is
    # what makes this arm runnable over the 15 walk-forward folds without a training pass,
    # and it removes the whole class of "a renamed strategy self-trains inside its own
    # backtest" hazard this repository has recorded.

    timeframe = "15m"
    inf_timeframe = "15m"
    startup_candle_count = 200
    process_only_new_candles = True
    can_short = False

    use_custom_stoploss = True
    use_exit_signal = True

    # ---- EXIT: the Open operating point, verbatim ------------------------------------------
    # These are the values `ExitConfig.from_operating_point()` reads, so a trade opened here
    # closes the way the outcome table simulated it. Changing any of them breaks the
    # correspondence between this arm and the leaderboard row it exists to test (FR-019 holds
    # the exit fixed for exactly this reason).
    minimal_roi = {"0": 0.05}
    stoploss = -0.05
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    atr_stoploss_multiplier = 2.5
    atr_stoploss_floor = -0.04
    atr_stoploss_cap = -0.02
    use_atr_adaptive_stoploss = False
    use_volume_confirmation_stoploss = True

    sell_params = {
        **getattr(BaseStrategy, "sell_params", {}),
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 10,
        "cexit_take_profit": 0.022,
        "enable_exit_signal": False,
        "stoploss_grace_hours": 6.5,
        "stoploss_grace_level": -0.27,
    }

    # ---- ENTRY: guards OPEN ----------------------------------------------------------------
    # Every QUALITY threshold neutralised, matching the Open operating point. `rvol` is left
    # at -1.0 (permissive) because the trigger carries its OWN rvol > 1.5 condition; leaving
    # the gate's rvol live as well would filter the trigger twice and the arm would no longer
    # correspond to the scored row.
    buy_params = {
        **getattr(BaseStrategy, "buy_params", {}),
        "entry_enable_guards": False,
        "entry_rvol_threshold": -1.0,
    }
    entry_bear_filter_enable = False
    entry_trend_filter_enable = False

    def get_entry_conditions(self, dataframe: DataFrame):
        """The leaderboard's `breakout_vol` trigger, by import.

        Evaluated through the registry so the registry's guarantees still hold here: it
        cannot read an outcome column (there are none in a live dataframe anyway) and it
        cannot read the pair.

        NOTE ON THE ENTRY LAG. The leaderboard pairs a signal on bar `i` with an entry at
        `open[i+1]`, because the trigger reads bar `i`'s close (`labels.entry_mask`;
        booking bar `i` itself was a real defect, worth 91-94% of the breakout family's
        apparent value). freqtrade already does exactly this: a signal set on the last
        closed candle is filled on the NEXT candle's open. So the live path and the measured
        pairing agree WITHOUT a manual shift here -- adding one would double-lag the entry.
        """
        try:
            from research.outcomes import triggers as _triggers
            _triggers.ensure_builtins()
            return _triggers.evaluate(TRIGGER_NAME, dataframe).to_numpy()
        except Exception as exc:  # noqa: BLE001
            # REFUSE rather than fall back to a copy. A hand-written duplicate of the
            # condition could drift from the row this arm is testing, and an arm that
            # cannot be traced to its measurement is worse than no arm.
            print(
                f"*** {self.__class__.__name__}: could not evaluate trigger "
                f"{TRIGGER_NAME!r} ({type(exc).__name__}: {exc}). Entering NO trades -- "
                f"this arm must not run on a fallback definition.",
                flush=True,
            )
            return np.zeros(len(dataframe), dtype=bool)
