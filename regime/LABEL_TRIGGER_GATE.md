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

## What the trigger was actually buying — CORRECTED

I first wrote that the trigger bought LEARNABILITY and paid ~71pp of
TRADEABILITY for it, "a terrible exchange rate". **That was wrong**, and the
measurement below refutes it. `regime/tradeoff.py`:

    guard <  trigger %   label %  P(gain|trig)  free MCC  W1 ceiling
       -0.4      9.35%     4.82%         51.5%     0.701          --
       -0.2     11.12%     5.78%         52.0%     0.701      26.54%
        0.0     12.76%     6.70%         52.5%     0.701      35.09%
        0.2     14.43%     7.63%         52.9%     0.700      43.60%
        0.5     17.26%     9.19%         53.3%     0.697          --
        1.0     25.78%    13.39%         52.0%     0.667      97.97%

**The free MCC is essentially FLAT across the entire guard axis** — 0.701 at
−0.4 through 0.700 at +0.2, and still 0.667 with the guard removed entirely.

Two errors in the earlier reading:

1. I claimed that removing the guard leaves "no trigger, free MCC ~0". It does
   not. `guard_threshold = 1.0` removes only the GUARD term; `bb_width > 0.035`
   still applies, so the label remains a conjunction with a free part — 25.78%
   of bars, still 100% recall, still MCC 0.667.
2. There is therefore no meaningful learnability/tradeability exchange along
   this axis. Relaxing the guard from −0.2 to +0.2 costs **0.001 of free MCC**
   and buys **+17pp of W1 ceiling**. Removing it entirely costs 0.034 and buys
   +71pp.

`P(gain | trigger)` also barely moves (51.5% → 53.3%), so the admitted bars are
not systematically worse — there are simply far more of them.

**This makes relaxation MORE attractive than I said, not less.** The earlier
caution was mine, and it was unfounded on this axis. The genuine open question
is no longer the trade-off but whether a real model can convert a larger,
looser label — which is a question about the information ceiling, not about a
learnability price.
## Infrastructure

`NNNC/LabelParamsMixin.py` forwards an arbitrary `LABEL_PARAMS` dict to any
labeler, with `LABEL_PARAMS_SELL` for the sell side so one side can be varied
alone. It generalises beyond `labels_gbb` to all 27 labelers and any argument
they accept, and preserves the parent's peak-filter step exactly — verified by
re-running `ora_g10` after the refactor and reproducing its trade hash
byte-identically (1968tr, +97.97%, hash 059a09c48e03).

---

# The real models do NOT convert the extra ceiling

Both relaxed labels trained and evaluated. The first comparison was run at a
fixed threshold 0.80 and was CONFOUNDED — a looser label has more positives
(7.63% of bars at guard<+0.2 vs 5.78%), which shifts the confidence
distribution, so 0.80 is not the same operating point. Trade counts collapsed
(431 -> 186 -> 71). Re-swept per label:

    label            thr      tr     ret%     PF   win%
    guard<-0.2      0.69    1860   -12.71   0.91   58.0
    guard<-0.2      0.80     431    +6.94   1.16   65.4
    guard<+0.2      0.69    1827   -10.56   0.92   58.3
    guard<+0.2      0.75    1040    -6.92   0.92   59.6
    guard<+0.2      0.80     186    +5.21   1.22   64.0
    guard REMOVED   0.60    2033   -10.30   0.93   58.8
    guard REMOVED   0.75     519    -0.06   1.00   62.0
    guard REMOVED   0.80      71    +6.43   1.52   77.5

    best at >= 200 trades (the gate's floor):
      guard<-0.2 (control)   thr 0.80   431tr   +6.94%
      guard<+0.2             thr 0.75  1040tr   -6.92%
      guard REMOVED          thr 0.75   519tr   -0.06%

**The control label still wins.** Re-sweeping removed the confound and did not
rescue the relaxed labels.

At MATCHED trade volume the three labels are nearly indistinguishable — around
1800-1900 trades they return −12.71 / −10.56 / −15.16; around 2300 they return
−23.73 / −24.05 / −18.42. The label change is absorbed at the level of realised
return, and what actually moves P&L is the operating point.

## The metrics improved and the return did not

    label            val_mcc   val_precision   W1 ceiling   best real W1
    guard<-0.2        0.6078          0.4496       26.54%         +6.94%
    guard<+0.2        0.6191          0.4742       43.60%         -6.92%
    guard REMOVED     0.6192          0.4897       97.97%         -0.06%

Both classifier metrics rose monotonically with relaxation, the ceiling nearly
quadrupled, and realised return fell. The model captures roughly 12-16% of the
available ceiling regardless of how much ceiling is created.

This is the sharpest form of the information-ceiling result yet. Creating
headroom does not create edge: the constraint is the model's ability to convert
it, and that did not move when the label got easier to learn BY ITS OWN METRICS.

## Status of this line

**Label-trigger relaxation is a NEGATIVE result on realised return** and should
not be adopted. The ceiling measurements stand and are worth keeping — they show
the label discards a large tradeable population — but nothing currently
available can harvest it.

Do not re-run guard/bb_width relaxation expecting P&L. The remaining lever is
unchanged and is the same one the architecture study landed on: information not
derivable from OHLCV.
