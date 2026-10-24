# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""A unit stake weight for an NNMT book with no sizer, so a post-model scaler that multiplies
`stake_weight` (040 follow-up: TrScaleMixin, NNNC/_transition_signal.py) has a base to scale. Weight 1.0
leaves the stake as the parent sizes it. Post-model only (model key IGNORED)."""


class StakeWeightMixin:
    MODEL_KEY_METHODS_IGNORED = ("stake_weight", "custom_stake_amount")

    def stake_weight(self, entry_tag) -> float:
        return 1.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, min_stake,
                            max_stake, leverage, entry_tag, side, **kwargs):
        stake = super().custom_stake_amount(
            pair, current_time, current_rate, proposed_stake * self.stake_weight(entry_tag),
            min_stake, max_stake, leverage, entry_tag, side, **kwargs)
        return min(stake, max_stake)
