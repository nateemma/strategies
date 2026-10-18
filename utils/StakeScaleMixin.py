"""Stake scale for strategies without a sizer (research/tb_model/preregistration_liq.json).

`STAKE_SCALE = 1.0` (default) changes nothing. Otherwise the proposed stake is multiplied by
STAKE_SCALE before the parent's sizing, so the stake is min(STAKE_SCALE x proposed, fillable cap,
max_stake): NNNC_CS_ARO_ALL3K's all-available entry, for a book with no stake_weight.
"""

from __future__ import annotations


class StakeScaleMixin:
    STAKE_SCALE: float = 1.0
    MODEL_KEY_IGNORED = ("STAKE_SCALE",)
    MODEL_KEY_METHODS_IGNORED = ("custom_stake_amount",)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, min_stake,
                            max_stake, leverage, entry_tag, side, **kwargs):
        stake = super().custom_stake_amount(
            pair, current_time, current_rate, proposed_stake * float(self.STAKE_SCALE),
            min_stake, max_stake, leverage, entry_tag, side, **kwargs)
        return min(stake, max_stake)
