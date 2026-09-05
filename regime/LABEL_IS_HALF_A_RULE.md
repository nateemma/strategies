# The gbb label is half a threshold rule — and the rule beats the network

2026-09-05. Measured on XRP/SOL/LINK, H=48, threshold 0.02.

## The structure

    labels_gbb:  buy = (guard_metric < -0.2) AND (bb_width > 0.035)
                       AND (future_gain >= min_gain)

**Two of the three conjuncts are deterministic functions of features the model
already receives at time t.** `guard_metric` and `bb_width` are both in
`include_list`. Only `future_gain` requires prediction.

So the label factorises into a TRIGGER, free at inference, and an OUTCOME, the
only genuinely hard part. And because the label is a conjunction,
**label ⊆ trigger** — every labelled bar is a trigger bar.

## What the trigger alone is worth

    pair          bars   trigger %   label %   P(gain|trigger)   P(gain) all bars
    XRP_USDT    111280       6.99%     3.47%             49.6%              26.7%
    SOL_USDT    198764      13.84%     7.57%             54.7%              41.7%
    LINK_USDT   162517      10.62%     5.18%             48.8%              36.4%

The trigger lifts the hit rate from **34.9% to 51.1%**. That is the whole of its
contribution, and it is available from two comparisons.

## A trigger-only predictor scores HIGHER than the trained network

Predict 1 on every trigger bar, forecast nothing:

    pair      precision   recall     MCC
    XRP           49.6%   100.0%   0.692
    SOL           54.7%   100.0%   0.714
    LINK          48.8%   100.0%   0.678
    mean          51.0%   100.0%   0.695

    TRAINED MODEL (NNNC_MLX_Lbl02):  val_mcc 0.6078,  val_precision 0.450

**Recall is 100% by construction** — label is a strict subset of trigger, so a
trigger-only rule cannot miss a positive.

**MCC 0.695 for two thresholds against 0.608 for a 51-feature, 12,483-parameter
network.** The network does not merely fail to beat the rule; it fails to
reproduce it, despite receiving both of its inputs.

## Caveat on the comparison

The 0.695 is computed on raw `labels_gbb` over three pairs; `val_mcc` is on a
validation split of the 11-pair training set AFTER
`filter_peaks_by_future_performance`. The populations differ, so the two numbers
are not exactly commensurable and the gap should not be read to three decimals.
The STRUCTURAL claims are exact regardless: the label is a conjunction, two
conjuncts are observable at t, and recall is 100% by construction.

## What this means for architecture

**It closes the architecture question.** Attention, Transformer and Mamba all
differ in how they model TEMPORAL structure. The model's deficit is not temporal
— it is underperforming a STATIC threshold on two features it is already fed.
No sequence-modelling architecture addresses that, which is consistent with nine
variants all landing inside val_mcc 0.5957–0.6078.

## What this means for feature prioritisation

Prioritising `guard_metric` and `bb_width` — by feature-wise attention,
squeeze-excitation, or any other weighting — is the right instinct and it aims
at the right two features. But its entire achievable payoff is the trigger, and
the trigger is worth 34.9% → 51.1%. It cannot help with the coin flip that
remains.

**The stronger move is to stop asking the model to learn the trigger at all.**

1. **Give it the trigger as an explicit binary feature.** Cheapest; removes the
   need to rediscover a threshold rule from continuous inputs.
2. **Train ONLY on trigger bars, predicting the outcome.** The trigger is
   deterministic, so learning it consumes capacity for nothing. This converts a
   3-class problem — where two thirds of the definition is free — into the
   binary problem that is exactly the hard part: P(gain >= 2% | trigger),
   currently 51.1% and therefore close to maximum entropy. At inference the
   trigger is applied as a rule and the model acts purely as the outcome filter.
   This is the principled version and is a training-pipeline change, not an
   architecture change.
3. **Feature-wise attention** — worth having only if 1 and 2 do not settle it.

Option 2 also explains the threshold-0.80 result cleanly: what the model is
currently ranking well at the top of its confidence distribution IS the outcome
signal, buried under a trigger it wastes capacity re-deriving.
