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
    # 2026-09-08: 17 (gbb) -> 27 (gbb_path). The path-aware pair drops buy
    # labels whose stop would fire before the target and sell labels the
    # take-profit would pre-empt, so the label certifies a move the strategy
    # can actually harvest. Ceilings improved on BOTH families -- guards-off
    # +13.4pp (NNMT) and +15.6pp (NNNC), guards-on +2.4pp and +1.2pp -- while
    # realised return was unresolved either way. Adopted as the better-specified
    # target. See regime/PATH_LABEL_GATE.md and ORACLE_CORRECTION_2026-09-07.md.
    TRAINING_TYPE: int = 27

    # Minimum forward gain (loss) magnitude required for a candidate bar to
    # be labelled as a Buy (Sell). Lower values produce more training data
    # but smaller per-signal expected value.
    #
    # 0.007 chosen 2026-05-31 from the H=[24,36,48] × thr=[0.007,0.010,0.013]
    # sweep with the new MLP+aug_risk learnability tooling. (HORIZON=48,
    # thr=0.007) is the first combo where MCC > 0.30 AND aug_risk != HIGH
    # AND EV/signal > 3% (the stop_loss noise floor) hold simultaneously
    # across XRP/SOL/LINK. Buy-side MCC 0.75-0.77, EV 3.20-3.80% — clears
    # the floor with margin on every pair tested. See
    # project_h48_viable_combo_found.md.
    #
    # The (HORIZON, threshold) pair MUST be retuned jointly — earlier
    # univariate sweeps (e.g. 2026-05-28 H sweep) missed the inflection
    # because at H<=24 there is no threshold that clears all three
    # viability criteria together. See project_horizon_threshold_learnability_finding.md.
    # 2026-09-06: 0.007 -> 0.02. A 0.7% target against the -4%..-6% ATR stop
    # plus round-trip fees cannot pay HOWEVER PERFECTLY it is predicted -- at the
    # ceiling (perfect prediction, guards open) 0.007 returns -18.89% on W1 while
    # 0.02 returns +26.54%, and every window flips from loss to gain.
    # See regime/LABEL_TRIGGER_GATE.md and regime/CLOSING_THE_LEARNABILITY_GAP.md.
    #
    # Affects the NEXT retrain of every NN family, not inference by models
    # already on disk. NNMT and the GAN variants have NOT yet been evaluated at
    # this label -- that is pending work, and their existing models were trained
    # at 0.007.
    MIN_BUY_GAIN_THRESHOLD: float = 0.02
    MIN_SELL_LOSS_THRESHOLD: float = 0.02

    # Forward window (in bars) used by the gbb labeler and other label
    # generators that look ahead from a candidate bar. Coupled with the
    # gain/loss thresholds above — both must be retuned together.
    #
    # H=48 chosen 2026-05-31 = 12-hour forward window (48 × 15min). This
    # is the shortest horizon at which the 2026-05-31 viability sweep
    # found combos passing MCC > 0.30 AND aug_risk != HIGH AND EV > 3.0%
    # on all three test pairs. Shorter horizons fail because EV/signal
    # cannot clear the ~3% stop_loss noise floor without forcing the
    # gain threshold so high that label density collapses and DDPM
    # augmentation drifts.
    HORIZON: int = 48
