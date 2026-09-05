# The label trade-off: tradeability and learnability move in OPPOSITE directions

Measured 2026-09-05, following feature 004's oracle gate. Oracle figures are
LOOKAHEAD CEILINGS — diagnostic bounds, never achievable returns, never
promotable.

## 1. A real model does NOT realise the ceiling

`NNNC_MLX_Lbl02` = production `NNNC_MLX` with `MIN_BUY_GAIN_THRESHOLD` raised
0.007 → 0.02, HORIZON and every guard, stop and exit unchanged. Trained
`20220101-20250831`.

**Production gate (guards ON):**

    arm                                    W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%
    REAL model, label 0.007 (production)      95    10.45     181    10.31      56     3.37
    REAL model, label 0.02  (new)             95    10.45     181    11.35      56     3.37
    CEILING label 0.007                       89    11.12     163     9.00      55     4.41
    CEILING label 0.02                        85    13.93     156    14.80      50     5.72

**W1 and W3 trades are BYTE-IDENTICAL to production** — the guard chain absorbed
the change completely. Only W2 (in-sample) moves, +1.04pp.

**Guards off, W1 — directly comparable to the gate measurement:**

    REAL model, label 0.007   1970tr  -14.46%   PF 0.90   win 57.9%
    REAL model, label 0.02    1860tr  -12.71%   PF 0.91   win 58.0%
    CEILING    label 0.007    1760tr  -18.89%   PF 0.84   win 59.3%
    CEILING    label 0.02     1333tr  +26.54%   PF 1.37   win 70.4%

The ceiling improved by **+45.44pp**. The real model captured **+1.75pp** — under
**4%** of what was available. The label is fine; **the model cannot learn it.**

Note what this does to the headroom figure: at label 0.007 the guards-on ceiling
was 11.12% against a real 10.45%, i.e. 0.68pp. At label 0.02 the ceiling is
13.93% against a real 10.45% — **3.48pp**, five times larger, and the model
realises none of it.

## 2. Why — learnability collapses as tradeability rises

`DebugSignalLearnability`, gbb (method 17), H=48, XRP, peak filter applied:

    threshold   signals      MCC     EV/signal
    0.007          6086    0.779         3.60%
    0.010          5527    0.726         3.88%
    0.020          3882    0.541         4.90%
    0.030          2496    0.410         6.25%
    0.050          1180    0.277         8.95%

**MCC falls monotonically while EV per signal rises monotonically.** The
assumption that a higher threshold costs learnability is confirmed, and it is
the mechanism behind §1: at 0.02 the label is far more tradeable (ceiling
−18.89% → +26.54%) and far harder to learn (MCC 0.78 → 0.54).

This is the sharpest statement yet of the repo's oldest finding — learnability
is not edge. Here the two are not merely uncorrelated but **actively opposed**:
the most learnable label is the least tradeable one, and the label the model
predicts best is the one that cannot pay.

## 3. Feature 003's tail reasoning was RIGHT

003 raised the breakout label's threshold, improved reward:risk 0.57 → 1.28, and
realised return did not follow. It could not tell whether the label was wrong or
the model could not learn it, because it never measured its ceiling. Measured
now, on the breakout host, W1:

    threshold        gate           tr     ret%     PF   win%
    0.005 (003 base) guards ON      80     1.90   1.23   63.7
    0.02             guards ON      67     5.82   2.08   79.1
    0.03             guards ON      63     8.25   3.06   84.1
    0.005 (003 base) guards OFF   1697    33.32   1.43   63.8
    0.02             guards OFF   1167    86.78   3.39   80.2
    0.03             guards OFF    898    94.76   5.56   86.4

    for reference, 003's TRAINED arms:  label 0.005  97tr  -0.74%
                                        label 0.02   98tr  -0.20%

**The breakout ceiling rises monotonically with threshold at both gates.** 003's
economic reasoning was sound: raising the threshold really does make the label
more tradeable, by +3.92pp at the production gate and +53pp guards-off. The
trained model realised +0.54pp of it, inside noise.

**003's null was a LEARNABILITY failure, not a refutation of the idea.** Its
recorded conclusion — "label economics do not predict return, and the
relationship is non-monotonic" — describes what a TRAINED MODEL delivers, and
stands as such. It should not be read as "the tail hypothesis was wrong". The
hypothesis was right about the label and wrong only in assuming a model could
exploit it.

**And the breakout label is far more tradeable than gbb.** At its own baseline
its guards-off ceiling is **+33.32%** against gbb's **−18.89%**. The earlier
framing — breakout parked at +1.31% against gbb's +25.32% — measured trained
models, and inverts at the ceiling. The breakout label was never the problem.

## 4. What this changes

The bottleneck has moved. It is no longer "which label pays" — that is now
answerable in minutes, and several labels pay handsomely at the ceiling. It is
**that the model cannot learn a label worth trading, and the guard chain
absorbs whatever it does learn.**

Three consequences:

1. **Do not adopt label 0.02 on NNNC.** It is byte-identical to production on
   W1 and W3. There is no realised gain to bank.
2. **Any future label work must be paired with a learnability measurement**, not
   just an oracle ceiling. The gate answers "could this pay"; it does not answer
   "can this be learned". Both are now cheap and both are necessary.
3. **The open problem is the gap itself**: a label with MCC 0.54 and a +45pp
   ceiling delivers 1.75pp. Closing that is worth more than any further label
   search — and nothing in this repo has yet attacked it.
