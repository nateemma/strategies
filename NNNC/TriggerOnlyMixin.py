"""TriggerOnlyMixin — stop asking the model to re-derive a threshold rule.

`labels_gbb` is a CONJUNCTION:

    buy = (guard_metric < -0.2) AND (bb_width > 0.035) AND (future_gain >= thr)

The first two terms are deterministic functions of features the model already
receives, so two thirds of the label definition is free at inference. Measured
2026-09-05 (regime/LABEL_IS_HALF_A_RULE.md): a trigger-only predictor scores
MCC 0.695 with 100% recall — HIGHER than the trained network's 0.608 — because
`label` is a strict subset of `trigger`.

So the network spends capacity re-deriving a rule it is handed the inputs to,
and the only genuinely hard part, P(future_gain >= thr | trigger), is currently
51.1% — near maximum entropy.

This mixin makes that split explicit:

  TRAINING   every non-trigger bar gets sample weight 0, so the loss is computed
             ONLY where the trigger fires and all capacity goes to the outcome.
             Rows are NOT dropped -- the model consumes 16-bar sequences, and
             deleting rows would splice unrelated bars together. Masking by
             weight preserves sequence continuity.

  INFERENCE  the trigger is applied as a RULE and the model acts purely as the
             outcome filter. This is required, not optional: the model is never
             trained on non-trigger bars, so its output there is undefined.

Implementation reuses the framework's existing per-sample weight path
(`%pnl_weight` -> `sample_weights` -> the classifier's focal loss). With
`pnl_loss_alpha = 1.0` the blend `w = (1-alpha) + alpha*(mag/mean)` reduces to
pure masking: 0 for non-trigger, ~1/density for trigger. No Framework/ change.
"""

import numpy as np
import pandas as pd


class TriggerOnlyMixin:
    # 1.0 makes the weight blend pure masking. Anything lower leaves the model
    # partly learning the trigger again.
    pnl_loss_alpha = 1.0
    train_seed = None

    # labels_gbb defaults. The strategy passes only horizon/min_gain/min_loss to
    # the labeler, so these ARE the values the label is built with.
    GUARD_T = -0.2
    BBW_T = 0.035

    def _trigger_mask(self, dataframe) -> np.ndarray:
        n = len(dataframe)
        gm = np.nan_to_num(
            np.asarray(dataframe.get("guard_metric", pd.Series(np.zeros(n))), dtype=float)
        )
        bw = np.nan_to_num(
            np.asarray(dataframe.get("bb_width", pd.Series(np.zeros(n))), dtype=float)
        )
        return (gm < self.GUARD_T) & (bw > self.BBW_T)

    def get_training_labels(self, dataframe):
        labels = super().get_training_labels(dataframe)
        trig = self._trigger_mask(dataframe)
        dataframe["%pnl_weight"] = np.where(trig, 1.0, 0.0)
        if getattr(self, "dbg_verbose", False):
            self.debug_print(
                f"        TriggerOnly: {trig.sum()} / {len(trig)} bars carry loss "
                f"({100.0 * trig.mean():.2f}%)"
            )
        return labels

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf = super().get_classifier(classifier_type, pair, seq_len, num_features)
        if clf is not None:
            clf.pnl_loss_alpha = self.pnl_loss_alpha
            clf.train_seed = self.train_seed
        return clf

    def get_entry_conditions(self, dataframe):
        """Model prediction AND the trigger rule.

        Mandatory, not cosmetic: with the loss masked to trigger bars the model
        has never been trained off-trigger and its output there is meaningless.
        """
        cond = super().get_entry_conditions(dataframe)
        trig = pd.Series(self._trigger_mask(dataframe), index=dataframe.index)
        return cond & trig if cond is not None else trig
