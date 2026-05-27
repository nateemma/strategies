"""
TrainingConfig — canonical training-signal configuration.

Single source of truth for the threshold and method values that must agree
across three independent code paths:

  1. The strategy generates training labels via
     ``Framework.TrainingSignals.get_train_buy_signals/get_train_sell_signals``,
     parameterised by ``MIN_BUY_GAIN_THRESHOLD``, ``MIN_SELL_LOSS_THRESHOLD``,
     and ``TRAINING_TYPE``.
  2. The Create-class trainer persists these values in the saved GAN
     metadata at training time.
  3. The strategy at runtime validates its current values against the
     loaded GAN metadata and raises ``GANMetadataMismatchError`` if they
     disagree.

If any of the three drift, the GAN load fails loudly — useful as a safety
net, but painful when the cause is unintentional drift between hardcoded
values in different files. Reading from this class in one place removes
the drift risk entirely.

## Overriding

To override a value for a specific strategy, set the corresponding class
attribute directly on that strategy class (subclass override):

    class MyDebugStrategy(NNMTStrategy):
        TRAINING_TYPE = 1  # use triple-barrier for this experiment only

Do NOT mutate ``TrainingConfig`` at runtime — it's the canonical default
that other strategies inherit. Per-strategy overrides via class-attribute
reassignment are the supported customization point.
"""

from __future__ import annotations


class TrainingConfig:
    """Canonical training-signal configuration constants.

    All values here are defaults — any strategy can override by setting the
    same-named class attribute directly. See module docstring for details.
    """

    # Method id for label generation. See Framework.TrainingSignals.LabelMethod
    # for the full enum. Common values:
    #   1  = triple_barrier (poorly learnable — confirmed by
    #        DebugSignalLearnability, MCC ~0.03 across pairs)
    #   16 = indicators3
    #   17 = gbb (highest MCC, tightest cross-pair variance)
    #   19 = indicators4 (highest total throughput score)
    TRAINING_TYPE: int = 17

    # Minimum forward gain (loss) magnitude required for a candidate bar to
    # be labelled as a Buy (Sell). Lower values produce more training data
    # but smaller per-signal expected value. The DebugSignalLearnability
    # sweep across ICP/ETH/BTC/ADA/LINK shows the learnability curve peaks
    # at ~0.003 — see docs/superpowers/specs/ if a learnability rerun is
    # needed before changing.
    MIN_BUY_GAIN_THRESHOLD: float = 0.005
    MIN_SELL_LOSS_THRESHOLD: float = 0.005
