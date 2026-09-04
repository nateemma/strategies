# US1 result — the label effect in isolation (T015-T021)

**Verdict: UNCHANGED (indistinguishable).** Arm `l1` = candidate C1_h48_t02,
gain threshold 0.005 -> 0.02, HORIZON unchanged at 48.

## Liveness — the change DID reach the trades

    window   arm      control   identical
    W1        98tr      97tr    False
    W2       121tr     119tr    False
    W3        80tr      77tr    False

`exit_config_hash` 203297442969c7da on both sides, so the exit machinery is
identical and the difference is attributable to the label alone.

This is the first model-side change in this study NOT absorbed by the entry
gate. Six of six prior attempts on NNNC/gbb produced byte-identical trades
(R6); the breakout arm, with two of its seven guards neutralised, lets a label
change through on all three windows. The premise that motivated choosing this
arm holds.

## W1 — OUT-OF-SAMPLE, the only acceptance window (FR-013)

    arm       98tr   -0.20%   PF 0.98   win 60.2%   DD 3.80%
    control   97tr   -0.74%   PF 0.94   win 58.8%   DD 3.74%
    delta            +0.54pp

**+0.54pp is below the 1.0pp power threshold on 97 control trades (FR-017,
R9), so it is INDISTINGUISHABLE from noise, not an improvement.** Every
component moves the right way — return, profit factor, win rate — and one
extra trade, at essentially unchanged drawdown. But a 4x better-specified
label bought half a point on a window where the arm is still negative.

## W2 / W3 — IN-SAMPLE, excluded from acceptance

    W2   +1.36%  vs  +0.34%   (+1.03pp)
    W3   +1.53%  vs  +1.71%   (-0.18pp)

Both fall inside the 20220101-20250831 training window. They do not
corroborate W1 and are not evidence of persistence — note they disagree in
sign with each other.

## Context only — never an acceptance input (FR-014)

val_mcc 0.5165, val_precision 0.397. Lower than the control's 0.7116, while
W1 return improved. That is the fourth independent instance in this study of a
classifier metric pointing the opposite way to realised return, and the reason
FR-014 exists.

## What this means for the label hypothesis

The screening economics were real and the label reached the trades, so the
mechanism is not in doubt. What is in doubt is the SIZE of the effect: fixing
reward:risk from 0.57 to 0.86 moved out-of-sample return by half a point.
C3_h96_t03_TAIL carries the economics the study was actually built around
(r:r 1.28, 66.8% clearing the stop) and remains unmeasured because it needs
the hold-window change to be measured honestly (Phase 5, FR-011).

Read against C1's result, the reasonable expectation for C3 is a further
sub-point move, not the 2-3pp the arm needs to clear break-even. That is a
judgement about magnitude, not a refutation, and Phase 5 is the measurement
that would settle it.
