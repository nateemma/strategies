# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""One trigger, no model: the shared scaffold for feature 021's trigger-only arms.

WHY A BASE CLASS AND NOT SIX COPIES. Each arm differs from the others in exactly one
token -- the trigger name. Feature 021 exists to remove duplicated arithmetic, so six
hand-copied strategies that each restate the same exit block, the same open guards and the
same entry path would contradict the feature in its own deliverable. Here the exit and the
entry path exist ONCE and a subclass supplies `TRIGGER_NAME` and nothing else. A subclass
that needs to override an exit parameter is a subclass that has stopped being comparable,
which is why none of them do.

WHAT THESE ARMS ARE FOR. The leaderboard's economic-value column scored four
`selection_value` rows (all breakout family) from the outcome table alone, with no model.
These arms are the only instrument that can say whether that column predicts a RETURN.
Each trades one trigger, no model, guards open, under one fixed exit (FR-019) -- so the arms
differ from each other in the trigger and in nothing else, and they are comparable
fold-for-fold because they share the control's fold plan.

THE ENTRY IS THE REGISTERED TRIGGER ITSELF, by import, via
`research.outcomes.triggers.evaluate(TRIGGER_NAME, ...)` -- the same function the leaderboard
scored. A hand-written copy of a condition could drift from the row the arm is testing, and
an arm that cannot be traced to its measurement is worse than no arm. On import failure the
strategy REFUSES to trade rather than falling back to a copy.

ENTRY LAG. The leaderboard pairs a signal on bar `i` with an entry at `open[i+1]`, because
the trigger reads bar `i`'s close (`labels.entry_mask`; booking bar `i` itself was a real
defect worth 91-94% of the breakout family's apparent value). freqtrade already does exactly
this -- a signal set on the last closed candle fills at the next candle's open -- so the live
path and the measured pairing agree WITHOUT a manual shift. Adding one would double-lag.

WHAT THESE ARMS CANNOT SHOW. The table's economic value is a lookahead bound (FR-017). Only
the pooled harness in `evaluation/` can make a P&L claim, at its ~3-4pp/yr floor.
"""
import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent.parent))

from Framework.BaseStrategy import BaseStrategy


class TriggerOnlyStrategy(BaseStrategy):
    """`TRIGGER_NAME` + the Open exit, no model, guards open. Subclass and set the name."""

    #: The trigger name on feature 021's leaderboard. A subclass MUST set this; the base
    #: refuses to trade rather than guess, so a subclass that forgets cannot silently
    #: become a different arm.
    TRIGGER_NAME: str | None = None

    # ---- no model, no training -------------------------------------------------------------
    # BaseStrategy's default StrategyConfig has model_type NONE and needs_training False, so
    # nothing is fitted and there is no per-fold artefact. That is what makes these arms
    # runnable over 15 folds without a training pass, and it removes the recorded hazard
    # where a renamed strategy self-trains inside its own backtest (get_model_path keys off
    # the class name).

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
    # correspondence between these arms and the rows they test, and breaks comparability
    # BETWEEN the arms -- FR-019 holds the exit fixed for exactly this reason.
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
    # permissive because several of these triggers carry their OWN rvol condition
    # (`breakout_vol` needs rvol > 1.5, `gbb` needs rvol > 1.0); leaving the gate's rvol live
    # as well would filter those triggers twice and the arm would no longer correspond to the
    # scored row. `volume > 0` stays live -- it is a tradeability constraint, not a quality
    # filter.
    # 2026-10-09: this block previously set `entry_enable_guards: False`; it is now True, which
    # matches BaseStrategy's own default and is the repo's convention for "guards open".
    #
    # CORRECTED 2026-10-10 -- the ORIGINAL REASON GIVEN HERE WAS WRONG. It cited
    # BaseStrategy.py's `buy_params` comment, which claims the flag "ALSO disables the
    # `volume > 0` data-sanity check". That comment is STALE: commit e7cf8cf (2026-09-08,
    # "volume+rvol mandatory") moved `volume > 0` AND `rvol` outside the flag in
    # `populate_entry_trend`, where only the five QUALITY thresholds remain behind it.
    #
    # So the earlier arms never lost the volume check, and the identical `trades_hash`
    # measured when the flag was flipped is explained by that -- NOT, as first claimed, by
    # `confirm_trade_entry`'s headroom check acting as a backstop. The measurement was right;
    # the explanation attached to it was not. (34.26% of bars here do have volume <= 0; that
    # figure is true and was simply irrelevant to this flag.)
    #
    # The flag stays True because it is what APPLIES the quality thresholds at all.
    #
    # The correct "open" configuration is the flag TRUE with every quality threshold
    # neutralised, which is exactly BaseStrategy's own default (adx -1.0, atr_pct -1.0,
    # bb_width -1.0, close_norm 1.1, guard 1.1, rvol -1.0). Inheriting it unchanged is
    # therefore both the right setting and the one least able to drift from the repo's
    # definition of "guards open".
    buy_params = {
        **getattr(BaseStrategy, "buy_params", {}),
        # left explicit because it is the whole point of this block, and because the
        # previous value here was the defect
        "entry_enable_guards": True,   # thresholds below are open; this keeps volume > 0 live
        "entry_rvol_threshold": -1.0,  # several triggers carry their own rvol condition
    }
    entry_bear_filter_enable = False
    entry_trend_filter_enable = False

    def get_entry_conditions(self, dataframe: DataFrame):
        """The leaderboard's `TRIGGER_NAME` trigger, by import.

        Evaluated through the registry so the registry's guarantees still hold: it cannot
        read an outcome column (there are none in a live dataframe anyway) and it cannot
        read the pair.
        """
        name = getattr(self, "TRIGGER_NAME", None)
        if not name:
            print(
                f"*** {self.__class__.__name__}: TRIGGER_NAME is not set. Entering NO "
                f"trades -- a trigger-only arm with no trigger would silently measure the "
                f"guards instead.",
                flush=True,
            )
            return np.zeros(len(dataframe), dtype=bool)
        try:
            from research.outcomes import triggers as _triggers
            _triggers.ensure_builtins()
            return _triggers.evaluate(name, dataframe).to_numpy()
        except Exception as exc:  # noqa: BLE001
            # REFUSE rather than fall back to a copy. A hand-written duplicate of the
            # condition could drift from the row this arm is testing, and an arm that
            # cannot be traced to its measurement is worse than no arm.
            print(
                f"*** {self.__class__.__name__}: could not evaluate trigger {name!r} "
                f"({type(exc).__name__}: {exc}). Entering NO trades -- this arm must not "
                f"run on a fallback definition.",
                flush=True,
            )
            return np.zeros(len(dataframe), dtype=bool)
