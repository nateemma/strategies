# What the model finds that the label forbids

The control model's 1489 trades across all three windows, joined back to the
indicator data at each entry bar and split by the label's own trigger,
`(guard_metric < -0.2) AND (bb_width > 0.035)`. Script `regime/offtrigger.py`.

## They are 1.9% of trades and 9.0% of profit

    group              n    win%   mean P&L    sum P&L  mean hrs
    OFF-trigger       29   75.9%      10.55      305.9       5.0
    ON-trigger      1460   65.5%       2.13     3103.6       6.5

**5x the per-trade profit, a 10pp higher win rate, and shorter holds.**

## Why the label rejects them

    guard_metric >= -0.2 only : 19
    bb_width <= 0.035 only    :  9
    BOTH fail                 :  1

Two thirds are rejected by the `guard_metric` condition alone — the
oversold test.

## They are a DIFFERENT TRADE ARCHETYPE

Median indicator values at entry:

    indicator         ON-trigger  OFF-trigger
    guard_metric          -0.614       -0.176
    close_norm            -0.744        0.263
    bb_position           -0.892       -0.261
    fisher_wr             -0.753       -0.104
    mfi_scaled            -0.496       -0.185
    rvol                   0.639        0.805
    adx                   25.931       22.919

**On-trigger entries are deep mean reversion**: heavily oversold, at the lower
Bollinger band (`bb_position` −0.89), near the bottom of the rolling range
(`close_norm` −0.74), on below-average volume.

**Off-trigger entries are not oversold at all**: mid-range on every oscillator,
`close_norm` ABOVE the midpoint at +0.26, `bb_position` near the middle, and
notably **higher relative volume** (0.81 vs 0.64).

That is not a marginal version of the same trade. It is closer to a
volume-confirmed continuation than to dip-buying.

## Their exits differ the same way

    reason                ON n  ON meanP   OFF n  OFF meanP
    take_profit            793     16.77      19      20.67
    trailing_stop_loss     497    -21.08       7     -17.39
    unclog_12h             133      5.07       2       5.45
    roi                     30     30.24       1      24.04
    stop_loss                6   -216.91       0       0.00

**66% of off-trigger trades reach take-profit against 54% on-trigger**, and 24%
stop out against 34%. They resolve faster and more often in profit — consistent
with entering on confirmation rather than catching a falling knife.

## CAVEAT — n = 29

Twenty-nine trades across three windows. A mean of 10.55 on that sample is high
variance, and the P&L figures should not be leaned on. What is more robust is
the DIRECTION and the mechanism: the indicator separation is large, consistent
across every oscillator, and interpretable; and the exit mix moves the same way.
Treat this as a well-formed hypothesis, not an established effect.

## What it suggests

The label's `guard_metric < -0.2` term encodes an assumption — that the strategy
is buying dips — which the model does not share, and where the model departs
from it, it does better. The label is not merely half a rule; **it is half a rule
that excludes a profitable regime.**

The cheap next test is a label variant with the guard condition RELAXED
(`guard_threshold` 0.0 or +0.2 rather than −0.2), screened through the oracle
gate before any training. If the ceiling rises, the label has been discarding
tradeable bars all along and the trigger should be widened. If it falls, these
29 trades are the model exploiting something narrower than a regime, and the
right response is different.

`guard_threshold` and `bb_width_threshold` are `labels_gbb` parameters that the
strategy does NOT currently pass — it sends only horizon/min_gain/min_loss — so
a variant needs them plumbed through or set on a labeler subclass.
