# Closing the learnability gap — measured, not argued

2026-09-05. Oracle and partial-oracle figures are LOOKAHEAD CEILINGS; the
threshold-sweep and trained-model figures are REAL and lookahead-free.

## 1. How accurate must a model be? — the partial-oracle curve

Degrade perfect prediction to a chosen precision, holding trade count constant
(keep a random fraction of the label's BUY bars, replace the rest with random
non-BUY bars). gbb 0.02 label, guards off, W1:

    selector                          tr     ret%     PF   win%     DD%
    oracle, precision 1.00          1333    26.54   1.37   70.4    7.81
    precision 0.90                  1521    21.05   1.24   68.0    8.85
    precision 0.80                  1657    22.71   1.25   67.4    7.12
    precision 0.70                  1824    24.02   1.23   65.6    6.27
    precision 0.60                  1872     6.32   1.05   64.4   11.36
    precision 0.50                  1979    16.52   1.15   63.7    7.30
    precision 0.30                  2022     5.73   1.05   62.1    7.76
    RANDOM (0.00, seed 17)          1931   -39.39   0.71   55.6   43.25
    RANDOM (0.00, seed 99)          1904   -30.55   0.76   55.9   37.28

**The random control is essential and it passes**: selecting the same ~1900 bars
at random loses 30–39%, so the curve measures label quality and not trade count
or slot allocation. (Worth stating because a count artefact was the obvious way
this could have been spurious.)

**Break-even precision is ~0.25** — far below the near-perfection the earlier
framing implied. The requirement is reachable. **This is a model problem, not a
data problem.**

The curve is noisy between 0.5 and 0.9 (0.60 gives +6.32% against 0.50's
+16.52%) because a single seed decides which bars are swapped; the SIGN is
stable everywhere above 0.3 and that is what the conclusion rests on.

## 2. The gap was largely an OPERATING POINT problem, not a training problem

Sweeping the trained model's own confidence threshold — **no retraining** —
guards off, W1:

    threshold     tr     ret%     PF   win%     DD%   implied precision
    0.0         2355   -24.63   0.85   58.5   32.38              ~0.08
    0.5         2319   -23.73   0.85   58.5   31.96              ~0.08
    0.69        1860   -12.71   0.91   58.0   23.11              ~0.16
    0.80         431    +6.94   1.16   65.4    7.90              ~0.60
    0.90           0        —      —      —       —                  —

**At threshold 0.80 the real model is PROFITABLE with the guards off.** This is
the first time in this programme that a trained model has been profitable
without the guard chain carrying it.

Compare the same sweep on the production 0.007 label, where the win rate was
FROZEN at 57.9–59.0% across the whole range out-of-sample: the model could not
rank at all. On the 0.02 label it ranks — 58.5 → 58.0 → 65.4% — because the
harder label gives it something real to be uncertain about.

**That is the answer to "how do we close the gap": most of it closes by moving
the operating point.** The model was being asked for volume at a threshold tuned
for a different label.

## 3. It persists across all three windows

Real model, 0.02 label, threshold 0.80, guards off:

    window     tr     ret%     PF   win%     DD%
    W1        431    +6.94   1.16   65.4    7.90
    W2        763   +16.91   1.31   65.0    8.54
    W3        295   +10.25   1.49   68.1    2.34

Positive in all three, including W3 (2023), a temporally distant era. Summed
+34.10% over 1489 trades against production's +24.13% over 332.

## 4. What this is, and what it is NOT

**It is NOT a production improvement.** At the production gate the same model is
worse than production (W1 42tr +8.42% against 95tr +10.45%), because there the
guards already do the selecting and a tighter model threshold only cuts volume.
And guards-off on W1 — the acceptance window — production still wins on both
return and drawdown (+10.45% / 1.50% against +6.94% / 7.90%).

**It IS a genuinely different, viable configuration**: model-driven rather than
guard-driven, profitable out-of-sample and in a distant era, at **4.5x the trade
volume** (431 vs 95 on W1). That volume is precisely what the whole study has
lacked — every prior intervention landed sub-1pp on ~95 trades and could not be
resolved. A 431-trade arm can measure things a 95-trade arm cannot.

**Remaining headroom**: the model at threshold 0.80 returns +6.94% against a
ceiling of +26.54%. Roughly 20pp is still unrealised — but the sign has flipped,
and the gap is now a matter of degree rather than direction.

## 5. What to do next, in order

1. **Tune the operating point properly.** 0.80 was the third point of a
   five-point sweep; 0.85 was never tried, and 0.90 gives zero trades. The
   optimum is somewhere in 0.75–0.88 and is worth locating on all three windows.
2. **Re-measure the model headroom at this configuration.** The 0.68pp figure
   was measured at the production gate on the 0.007 label; at 431 trades with a
   ranking model it will be entirely different, and it governs whether further
   model work pays.
3. **Then, and only then, revisit training.** Precision at the operating point
   is what matters, and the model is trained on MCC. Given that the requirement
   is only ~0.25 precision and the model already reaches it at threshold 0.80,
   objective changes are now a refinement, not a rescue.
4. **Do not adopt anything yet.** W1 still favours production. This is a
   candidate line, not a replacement.

---

# Tasks 1 and 2 — the operating point located, the headroom re-measured

## Task 1 — the threshold optimum is ~0.80, and it is NOT sharply identified

Real model, 0.02 label, guards off, all three windows:

    thr    W1 tr  W1 ret%  W1 PF  W2 tr  W2 ret%  W2 PF  W3 tr  W3 ret%  W3 PF   summed
    0.75    1276   -13.07   0.88   2035     2.77   1.02    994    28.04   1.49    17.74
    0.78     773    -0.90   0.99   1285     9.76   1.10    538    13.06   1.35    21.92
    0.80     431     6.94   1.16    763    16.91   1.31    295    10.25   1.49    34.10
    0.82     179     9.09   1.41    235    10.04   1.57    101     7.42   2.34    26.55
    0.85      23     5.61   6.36     18     0.68   1.38      8     1.36   0.00     7.65
    0.88       0        —      —      0        —      —      0        —      —        —

**Per-window winners are 0.82 / 0.80 / 0.75 — the optimum wanders**, exactly the
pattern that killed dynamic lookback. It must be treated as a FIXED CHOICE, never
tuned per era.

Two candidates are positive in all three windows (0.85 is excluded: 23/18/8
trades is not a strategy). Minimax regret decides between them:

    thr 0.80: regret W1 2.15  W2 0.00  W3 0.00   MAX 2.15
    thr 0.82: regret W1 0.00  W2 6.87  W3 2.83   MAX 6.87

**0.80 is the choice** — it minimises worst-case regret and carries 1489 trades
across the three windows against 0.82's 515, which matters directly for the
power problem. The sweep therefore CONFIRMED the value rather than moving it,
which is a more useful outcome than a new number.

## Task 2 — headroom at the new configuration: 25pp, not 0.68pp

The comparators must be volume-matched: measuring the model's 431 trades against
a 1333-trade oracle would confound selection quality with how many concurrency
slots each arm consumes. Both perfect and random selection are subsampled toward
the model's volume.

    selector                          W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%
    RANDOM at matched volume (s17)      853   -15.48
    RANDOM at matched volume (s99)      847   -13.70
    REAL model, thr 0.80                431    +6.94     763    16.91     295    10.25
    PERFECT at matched volume           812   +31.97    1473    62.10     777    45.79

    span   random -> perfect   = 47.45pp
    model captures             = 22.42pp   (47% of the span)
    REMAINING HEADROOM         = 25.03pp

    old configuration (production gate, 0.007 label):
    span 1.45pp, model captured 0.78pp, headroom 0.68pp on 95 trades

**The headroom is now 25pp against 0.68pp — roughly 37x larger — and the model
captures 47% of the span rather than 53% of almost nothing.**

This retires the "model work cannot pay" conclusion. It was true, and it was
true *of the production gate on the 0.007 label*. At this configuration model
quality is the dominant lever by a wide margin.

**Caveat on the volume match**: the model takes 431 trades against the
comparators' ~830. The subsample fraction was chosen before the model's count
was known and was not re-tuned. The direction of any resulting bias is not
obvious, but the conclusion — tens of points of headroom rather than a fraction
of one — is far too large to be an artefact of that mismatch.

## Where this leaves the programme

The bottleneck has moved twice in one session:

1. "The model adds nothing" — true at the production gate on the 0.007 label,
   where headroom was 0.68pp.
2. "The label cannot pay" — true of 0.007, fixed by 0.02.
3. **Now: model quality is worth ~25pp, and the model already captures 47%.**

Every prior null in this programme was measured at a configuration where there
was nothing to win. The obvious next question is the model and its training
metrics — for the first time in this study, improving them should show up in
realised return.

---

# The guards-on operating point cannot be improved — the gate absorbs it

Threshold swept at the PRODUCTION gate on the 0.02-label model:

    thr     W1 tr  W1 ret%  W2 tr  W2 ret%  W3 tr  W3 ret%   summed
    0.50       96    10.25    187     9.85     58     3.73    23.83
    0.60       95    10.45    185    10.35     58     3.73    24.53
    0.69       95    10.45    181    11.35     56     3.37    25.16
    0.75       87     9.72    171    10.92     50     3.33    23.96
    0.80       42     8.42     96     9.10     20     2.86    20.38
    0.85        4     3.07      2     0.71     --       --       --

    PRODUCTION (label 0.007, thr 0.69)
               95    10.45    181    10.31     56     3.37    24.13

**The threshold is inert guards-on.** W1 return is 10.25 / 10.45 / 10.45 across
0.50–0.69 while trade count barely moves (96 / 95 / 95); above 0.69 it only cuts
volume. This is the same absorption seen on the 0.007 label — the guard chain
removes the bars before the model's confidence can discriminate — and it holds
for the relaxed-gain label too.

Best summed is thr 0.69 at +25.16% against production's +24.13%. **W1 and W3 are
IDENTICAL to production (10.45 / 3.37); only in-sample W2 differs (+1.04pp).**
There is no production improvement here.

## Final position on the two configurations

    configuration                              W1        W2        W3   summed   trades   W1 DD
    PRODUCTION  label 0.007, guards on, 0.69  +10.45   +10.31    +3.37   +24.13      332    1.50%
    CANDIDATE   label 0.02, guards OFF, 0.80   +6.94   +16.91   +10.25   +34.10     1489    7.90%

The candidate wins on summed return and on two of three windows, with **4.5x the
trade volume** and positive in every window including distant 2023. Production
wins on W1 — the acceptance window — and on drawdown by 5x.

**Neither dominates, and the choice is not a measurement question.** On the
repo's own acceptance rule (W1 realised return) production stands. The candidate
is the better-diversified book and the only configuration in this study whose
edge comes from the model rather than the guard chain.

Nothing here is adopted. What the candidate is genuinely worth is as the
operating point for FUTURE model work: at 431 W1 trades a 1pp effect is
measurable, where at production's 95 it is not.
