"""LabelParamsMixin — forward ARBITRARY parameters to any labeler.

`TrainingSignals.get_train_buy_signals` dispatches as `fn(df, **params)`, so a
labeler's own keyword arguments are reachable in principle. But
`BaseNNStrategy.get_train_buy_signals` hardcodes the params dict to
horizon / min_gain / min_loss, so every OTHER labeler argument has always run at
its default and has never been varied in this repo.

For `labels_gbb` those hidden arguments are the label's entire trigger:

    guard_threshold      default  0.3    (was -0.2 before 2026-09-08)
    bb_width_threshold   default  0.035

and relaxing them moves the oracle ceiling by up to +71pp (regime/LABEL_TRIGGER_GATE.md).

Rather than expose those two by name, this mixin takes a DICT merged over the
base params, so it generalises to any of the 27 labelers and any argument they
accept — `LabelMethod.triple_barrier`'s barriers, `labels_breakout_bb`'s band
position, and so on — without another mixin each time.

    class MyArm(LabelParamsMixin, NNNC_MLX):
        LABEL_PARAMS = {"guard_threshold": 1.0}          # buy side
        LABEL_PARAMS_SELL = {"guard_threshold": -1.0}    # sell side, optional

Buy and sell are separate so a trigger can be varied on one side alone
(Constitution VIII). The parent's peak-filter step is preserved exactly.
No Framework/ change.
"""

import numpy as np
import pandas as pd

from Framework import TrainingSignals


class LabelParamsMixin:
    #: merged OVER {horizon, min_gain, min_loss}; empty = stock behaviour
    LABEL_PARAMS: dict = {}
    #: sell-side equivalent; None = mirror LABEL_PARAMS, {} = leave sell alone
    LABEL_PARAMS_SELL: dict | None = {}

    def _label_params(self, side: str) -> dict:
        base = {
            "horizon": self.HORIZON,
            "min_gain": self.MIN_BUY_GAIN_THRESHOLD,
            "min_loss": self.MIN_SELL_LOSS_THRESHOLD,
        }
        if side == "buy":
            # Mirror BaseNNStrategy's buy path. This dict is built fresh, so any key
            # added there MUST be added here or mixin arms silently diverge from
            # production (the "fresh dicts drop parent keys" failure). Buy-only:
            # the sell labelers take no rvol_threshold.
            base["rvol_threshold"] = getattr(self, "LABEL_RVOL_THRESHOLD", -1.0)
            extra = self.LABEL_PARAMS
        else:
            extra = (self.LABEL_PARAMS if self.LABEL_PARAMS_SELL is None
                     else self.LABEL_PARAMS_SELL)
        return {**base, **(extra or {})}

    def _filtered(self, future_df, labels, side: str) -> pd.Series:
        peaks = np.where(labels.values > 0.5)[0]
        # identical to the parent: filter at the LABEL horizon, not PEAK_WINDOW
        filtered = self.filter_peaks_by_future_performance(
            future_df, peaks, side,
            window=(self.LABEL_FILTER_WINDOW or self.HORIZON),
        )
        signals = pd.Series(np.zeros(np.shape(future_df)[0], dtype=float))
        signals[filtered] = 1
        if getattr(self, "dbg_verbose", False):
            self.debug_print(
                f"        {side.capitalize()} signals: {len(peaks)} peaks -> "
                f"{len(filtered)} filtered   params={self._label_params(side)}"
            )
        return signals

    def get_train_buy_signals(self, future_df):
        labels = TrainingSignals.get_train_buy_signals(
            future_df, method=self.TRAINING_TYPE, params=self._label_params("buy"))
        return self._filtered(future_df, labels, "buy")

    def get_train_sell_signals(self, future_df):
        labels = TrainingSignals.get_train_sell_signals(
            future_df, method=self.TRAINING_TYPE, params=self._label_params("sell"))
        return self._filtered(future_df, labels, "sell")
