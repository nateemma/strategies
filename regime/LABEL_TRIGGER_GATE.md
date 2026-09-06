# The label's trigger was throwing away most of the edge

Oracle gate applied to `labels_gbb`'s TRIGGER thresholds — the two arguments
`BaseNNStrategy` never passes, so they have run at their defaults for the life
of this repo. All figures are LOOKAHEAD CEILINGS, guards off, gain threshold
0.02. Never promotable.

    arm                            W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%
    CONTROL guard<-0.2 bbw>0.035    1333    26.54    2230    43.29    1191    28.76
    guard<0.0    (W1 only)          1407    35.09       --       --      --       --
    guard<+0.2                      1499    43.60    2519    63.32    1363    43.28
    guard REMOVED                   1968    97.97    3225   134.43    1919   108.59
    bb_width REMOVED                2477    80.14    3465    94.05    2407    87.57

    delta vs control                       W1        W2        W3
    guard<+0.2                        +17.06    +20.03    +14.52
    guard REMOVED                     +71.43    +91.14    +79.83
    bb_width REMOVED                  +53.59    +50.76    +58.81

**Every relaxation PASSES the gate, on every window, monotonically in how much
of the trigger is removed.** Removing the `guard_metric` term entirely nearly
QUADRUPLES the ceiling on W1 and roughly triples it on W2 and W3.

This confirms the off-trigger characterisation (`regime/OFF_TRIGGER_ENTRIES.md`)
at scale. The 29 off-trigger trades the model found on its own were not a
curiosity — they were a visible corner of a large excluded population.

## What the trigger was actually buying

Two things move in opposite directions as the trigger is relaxed:

    trigger                 ceiling (W1)   free MCC from the trigger
    guard<-0.2, bbw>0.035        +26.54%                      0.695
    guard REMOVED                +97.97%                    ~0 (no trigger left)

With the guard removed, `labels_gbb` reduces to `future_gain >= min_gain` — a
pure forward-return target with no observable anchor. The trigger was buying
LEARNABILITY (an MCC 0.695 head start, since `label ⊆ trigger` gives 100% recall
free) and paying for it in TRADEABILITY (~71pp of ceiling).

**That is a terrible exchange rate**, and it is the same trade-off found on the
gain threshold — learnability and tradeability opposed — but here the price is
an order of magnitude larger.

## What this does NOT establish

The gate answers "could a model trained on this label pay". It does NOT answer
"can a model learn it", and the two have already been shown to move in opposite
directions on this exact axis. A guard-free label is pure forward-return
prediction, which is the hardest possible target and is precisely where the
information ceiling bites.

So the next step is NOT to adopt the guard-free label. It is to find where on
this curve the PRODUCT of ceiling and learnability is maximised — the two
measurements now exist independently and cheaply:

    ceiling      oracle gate, minutes, no training      (this document)
    learnability DebugSignalLearnability, minutes       (peak filter ON)

`guard<+0.2` is the interesting middle: +17pp of ceiling on W1 and +14 to +20pp
on every window, while KEEPING a trigger (so keeping some free recall). It is
the cheapest candidate to train.

## Infrastructure

`NNNC/LabelParamsMixin.py` forwards an arbitrary `LABEL_PARAMS` dict to any
labeler, with `LABEL_PARAMS_SELL` for the sell side so one side can be varied
alone. It generalises beyond `labels_gbb` to all 27 labelers and any argument
they accept, and preserves the parent's peak-filter step exactly — verified by
re-running `ora_g10` after the refactor and reproducing its trade hash
byte-identically (1968tr, +97.97%, hash 059a09c48e03).
