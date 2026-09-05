# T023 — Phase 4 screening results

**Every figure here is a LOOKAHEAD CEILING** (FR-012, SC-009). The oracle's
entry signal is computed from future bars. Nothing below is an achievable
return, and no configuration here is promotable toward live trading.

Baseline: `oracle_nog` W1 = **1760 trades, −18.89%**.

## The screening table (oracle, guards OFF, W1)

    variant                                tr     ret%     PF   win%     DD%  vs base  gate
    BASELINE oracle_nog                  1760   -18.89   0.84   59.3   24.57     0.00  control
    --- exit side ---
    take-profit 0.04                     1704   -28.06   0.78   55.8   33.38    -9.16  fail
    take-profit 0.06                     1680   -25.76   0.80   54.8   31.38    -6.87  fail
    take-profit 0.10                     1662   -23.34   0.81   54.3   29.88    -4.44  fail
    profit checks OFF                    1654   -21.85   0.83   54.1   29.72    -2.96  fail
    ATR 6.0 / -10..-6%                   1714   -18.68   0.84   61.6   25.83    +0.21  fail
    ATR 3.0 / -4.5..-3%                  1764   -17.68   0.85   58.2   23.31    +1.21  fail
    ATR 2.0 / -3..-2%                    1760   -13.58   0.88   57.2   21.61    +5.31  fail
    --- label side ---
    label threshold 0.02                 1333   +26.54   1.37   70.4    7.81   +45.44  PASS
    label threshold 0.03                 1022   +46.60   1.96   80.3    6.40   +65.49  PASS
    --- combined ---
    tight stop x label 0.02              1336   +26.50   1.38   68.6    8.33   +45.39  PASS
    tight stop x TP 0.015                1834   -10.46   0.90   63.4   19.13    +8.43  fail

## All three windows for the passing variants

    arm                          W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%
    BASELINE label 0.007          1760   -18.89    2669   -36.96    1588   -11.69
    label 0.02                    1333   +26.54    2230   +43.29    1191   +28.76
    label 0.03                    1022   +46.60    1802   +93.67     838   +51.17
    tight stop x label 0.02       1336   +26.50    2240   +41.95    1184   +26.35

W1 is the acceptance window. W2/W3 corroborate the MECHANISM — every window
flips from a large loss to a large gain, and the ordering by threshold is
identical in all three — but they never confer acceptance (R10, FR-022).

## Separated contributions (FR-015, SC-005), W1

    stop alone    (ora_stight - baseline)   =  +5.31pp
    label alone   (ora_lbl02  - baseline)   = +45.44pp
    stop | label  (ora_comb   - ora_lbl02)  =  -0.05pp
    label | stop  (ora_comb   - ora_stight) = +40.08pp
    combined      (ora_comb   - baseline)   = +45.39pp

Each comparator differs from its arm in exactly one respect. **The label
accounts for essentially all of it.** The tighter stop is worth +5.31pp on its
own, but once the label is fixed it contributes −0.05pp — the label change
absorbs it entirely.

## RETRACTION — the T012 diagnosis drew the wrong conclusion

T012 concluded: *"the label is not what fails. The EXIT GEOMETRY fails on
ordinary bars, and the guard chain works by admitting only the extraordinary
ones."* **The exit-geometry half of that is refuted by measurement.**

The diagnosis was right that the ATR stop is the largest single destroyer of
P&L (717 trades, −11,967) and right that the stack's payoff ratio is ~0.45:1.
It was wrong to infer the geometry was therefore the fault. Its own prediction
— raise the take-profit cap and the return should improve — failed in the
strongest possible way: raising the cap made things **monotonically worse**
(−28.06 → −25.76 → −23.34 against −18.89 at the 0.024 baseline). Winners were
never being cut short; those bars give their gains back.

**What actually fails is the label threshold.** `MIN_BUY_GAIN_THRESHOLD` =
0.007 asks the model to find bars that rise 0.7% — against a −4% to −6% stop
and round-trip fees. Such a trade cannot pay however perfectly it is predicted.
Raise the threshold to 0.02 and the same exit stack, on the same bars, turns
+45pp. The stop was the mechanism through which the loss ARRIVED, not the
reason it existed.

The lesson is the one the feature was built to enforce and that I still got
wrong on the first pass: the largest P&L contributor is not necessarily the
cause. The forensics localised where the money went; only varying the parameter
showed why.

## FR-016 — status of the passing variants

`label 0.02` and `label 0.03` are **CANDIDATES FOR TRAINING, not adopted
changes.** The gate says only that a model trained on this label could pay —
because perfect prediction of it does. It does not say a real model will
realise it.

That caveat has teeth here. Feature 003 raised the breakout label's threshold on
exactly this reasoning, improved its reward:risk from 0.57 to 1.28, and the
trained model's realised W1 return did **not** improve — it moved
non-monotonically and ended negative. The difference is that feature 003 never
measured its ceiling: it could not tell whether the change failed because the
label was wrong or because the model could not learn the harder target. This
gate now separates those two, and the next step is to train `label 0.02` and
find out which applies to gbb.

`label 0.03` scores higher but takes 1022 trades against 1333, and its
threshold is 4.3x the current value; `label 0.02` is the cheaper first test.
