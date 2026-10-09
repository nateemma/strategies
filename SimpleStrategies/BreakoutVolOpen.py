# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""BreakoutVolOpen — the `breakout_vol` TRIGGER, traded with NO model.

Feature 021's board scored `breakout_vol` the strongest `selection_value` row with the
tightest interval of any non-withdrawn trigger: economic value **+0.00185 vs the `empty`
null, 95% CI [+0.00093, +0.00278]**, surviving calendar blocks at every length up to the
full 960-bar hold. This strategy is the only instrument that can say whether that TRADES.

WHY A NO-MODEL STRATEGY IS THE RIGHT TEST. Three measurements point here:
  * the model's selection inside `gbb` is reproducible by a plain threshold, and beaten by
    one 5.6x at matched selectivity (2026-10-09 diagnostic);
  * a conditional probe finds almost nothing predictable within any trigger's population
    (RIDGE +0.02..+0.05, LSTM -0.006 -- a non-linear model does no better);
  * this repository's standing findings: "guard ablation: model adds NOTHING" and
    "gate/mechanism move P&L, not model" (7 interventions, 2 families, every win gate-side).
So the hypothesis under test is that the trigger plus the shared exit IS the strategy.

THE ENTRY IS THE REGISTERED TRIGGER ITSELF, not a reimplementation of it.
`get_entry_conditions` calls `research.outcomes.triggers.evaluate("breakout_vol", ...)`, the
same function the leaderboard scored. A hand-copied `close > high.rolling(20).max().shift(1)
& rvol > 1.5` would be a second definition that could drift from the thing the measurement
was about -- exactly the duplication feature 021 exists to remove. If the import ever fails
this strategy REFUSES to trade rather than falling back to a copy, because a silent fallback
would make the arm untraceable to the row it is testing.

EXIT POLICY: the Open operating point, unchanged, so the trade outcomes are the ones the
outcome table simulated (`ExitConfig.from_operating_point`). Guards are OPEN (every quality
threshold non-binding) because the point is to measure the TRIGGER, not the trigger plus six
filters. `volume > 0` and `rvol` stay live -- they are tradeability constraints, not quality
filters, and `rvol > 1.5` is part of the trigger anyway.

WHAT THIS CANNOT SHOW. The table's economic value is a lookahead bound (FR-017). A positive
bound has twice failed to become harness return in this repository -- "trigger relaxation:
return FALLS; ceiling rises monotonically, return doesn't", and feature 003's rejected tail
labelling. A third such result is the single most likely outcome here and would be a real
finding, not a disappointment.
"""
import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent.parent))

from Framework.BaseStrategy import BaseStrategy

#: The trigger name on feature 021's leaderboard. Kept as a constant so the arm, the row and
#: this strategy are provably the same object.
TRIGGER_NAME = "breakout_vol"


class BreakoutVolOpen(BaseStrategy):
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
