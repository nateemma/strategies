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

---

# Option 2 BUILT and REFUTED — the model is not confined to the label's logic

`NNNC/TriggerOnlyMixin.py`: loss masked to trigger bars in training (via the
existing `%pnl_weight` -> `sample_weights` path, `pnl_loss_alpha = 1.0`), trigger
applied as a rule at inference. No `Framework/` change. Two arms, so the
inference gate and the training change separate.

**The masking engaged exactly as designed** — `alpha=1.0
w[min/mean/max]=0.000/1.000/7.825`, and 7–18% of bars carried loss per pair.

    arm                                     W1 tr  W1 ret%  W2 tr  W2 ret%  W3 tr  W3 ret%   summed
    CONTROL   no trigger anywhere             431     6.94    763    16.91    295    10.25    34.10
    TrigGate  trigger at inference only       400     4.21    720    12.96    277     9.62    26.79
    TrigOnly  masked training + inference     281     6.00    459    10.28    159     5.40    21.69

    delta vs control                           W1       W2       W3    summed
    TrigGate                                -2.73    -3.95    -0.62     -7.30
    TrigOnly                                -0.93    -6.63    -4.85    -12.41
    training change alone (TrigOnly-TrigGate) +1.79   -2.68    -4.22     -5.10

## The premise was wrong

**Applying the trigger at inference costs 7.30pp on its own**, before any
training change. The control model enters on bars the trigger REJECTS, and
those entries are profitable — 31 of its 431 W1 trades are off-trigger and
removing them costs 2.73pp.

So the model was never merely re-deriving the trigger. It had learned to enter
on bars outside the label's own trigger condition, profitably, and confining it
to the trigger destroys that. The reasoning behind option 2 — "capacity is
wasted on a rule, free it for the outcome" — does not survive contact with the
measurement.

The masked training does beat its own like-for-like control on W1 (+1.79pp,
TrigOnly vs TrigGate, same inference gate) and loses on W2 and W3. Not
persistent; not a result.

## What still stands, and what does not

**STANDS**: the label decomposition. `labels_gbb` IS a conjunction, two thirds of
it IS observable at time t, `label ⊆ trigger` so recall is 100% by construction,
and a two-threshold rule DOES score MCC 0.695 against the network's 0.608. Those
are facts about the label.

**DOES NOT STAND**: the prescription drawn from them. "The model wastes capacity
re-deriving the trigger, so mask the loss to trigger bars" is refuted. The
network's value is partly in exactly the region the trigger excludes.

## Note on val_mcc 0.1095

TrigOnly's val_mcc collapsed from 0.6078 to 0.1095. **This is expected and is
NOT the failure.** val_mcc is computed over ALL validation bars, including the
82–92% the model was never trained on; scoring badly there is what the masking
was designed to produce. The metric is measuring something the model was not
optimising. The realised return is the acceptance basis, and that is where the
arm actually fails.
