"""The model as a capital-neutral stake SIZER on the width book (feature 033, contract C2).

Placed before WidthBookMixin in the bases: entries are exactly the width book's (gbb rule AND
bb_width >= the fold's cut); the model only sets the stake. Each width-book signal row's tag gets
":agree" when the model's own decision fires on that row (predict_buy: argmax BUY and p_buy >
prediction_threshold), else ":disagree". custom_stake_amount then sizes from the tag:

  stake = proposed * STAKE_WEIGHTS[agree ? 0 : 1] / (s + STAKE_WEIGHTS[1] * (1 - s)),  s = AGREE_SHARE

so the expected mean stake equals the width book's at the pre-registered agree share (1.333x /
0.667x at s = 0.5), and total pp stays comparable with the width book. BaseStrategy's fillable
(liquidity) cap and freqtrade's max_stake still apply; shortfalls are measured, never bypassed.

prediction_threshold is redefined with 4 decimals on this mixin only: the base parameter rounds
picked thresholds to 2 (032 finding). Existing classes are unaffected.

AGREE_RULE "p_buy" (034) tags agree on p_buy > prediction_threshold alone, without the BUY-argmax
condition: for balanced three-class labels BUY rarely wins the argmax, so the decision rule cannot
reach the agree share. The default "decision" is the rule above.
"""

import numpy as np
from freqtrade.strategy import DecimalParameter


class WidthSizerMixin:
    prediction_threshold = DecimalParameter(0.0, 1.0, default=0.5, decimals=4, space="buy",
                                            load=True, optimize=False)
    AGREE_SHARE = 0.5
    STAKE_WEIGHTS = (1.0, 0.5)
    AGREE_RULE = "decision"
    MODEL_KEY_IGNORED = ("AGREE_SHARE", "STAKE_WEIGHTS", "AGREE_RULE")
    MODEL_KEY_METHODS_IGNORED = ("sizer_score",)

    def sizer_score(self, dataframe):
        """The AGREE_RULE "p_buy" score: the model's p_buy. A subclass may supply another source."""
        return dataframe["%predict_prob_buy"].to_numpy(float)

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)  # the width book; predictions written
        tags = self._st_tags
        if mask is None or tags is None:
            raise ValueError(f"{type(self).__name__}: the sizer needs the width book's tagged mask")
        if self.AGREE_RULE == "p_buy":
            p = np.nan_to_num(np.asarray(self.sizer_score(dataframe), dtype=float), nan=0.0)
            agree = p > float(self.prediction_threshold.value)
        else:
            agree = dataframe["predict_buy"].to_numpy() > 0.5
        m = np.asarray(mask, dtype=bool)
        tags = tags.copy()
        tags[m & agree] = [f"{t}:agree" for t in tags[m & agree]]
        tags[m & ~agree] = [f"{t}:disagree" for t in tags[m & ~agree]]
        self._st_tags = tags
        return mask

    def stake_weight(self, entry_tag) -> float:
        s, (wa, wd) = float(self.AGREE_SHARE), self.STAKE_WEIGHTS
        agree = bool(entry_tag) and str(entry_tag).endswith(":agree")
        return (wa if agree else wd) / (s * wa + (1.0 - s) * wd)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, min_stake,
                            max_stake, leverage, entry_tag, side, **kwargs):
        stake = super().custom_stake_amount(
            pair, current_time, current_rate, proposed_stake * self.stake_weight(entry_tag),
            min_stake, max_stake, leverage, entry_tag, side, **kwargs)
        return min(stake, max_stake)
