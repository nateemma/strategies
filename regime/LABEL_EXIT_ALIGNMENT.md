# Label target vs take-profit — the alignment question

Prompted by the observation that a PERFECT model still exits 47 of 237 trades at
−5.65%. If the label promises a gain, why does the stop fire first?

## 1. It is NOT the path

The label says "close rises >= 2% within 48 bars" and says nothing about the
route. The natural fix is a max-adverse-excursion constraint. Measured on
129,003 labelled bars (`regime/path_label.py`), taking the worst excursion
BEFORE the target is first reached — not whole-horizon MAE, since a drop after
the take-profit fires is irrelevant:

    worst adverse excursion before +2% is reached
        p1  -7.01%   p5  -3.68%   p10  -2.55%   p25  -1.25%   p50  -0.29%

    stop level   labels killed   % of labels
       -4%               5,369          4.2%
       -5%               3,169          2.5%
       -6%               1,931          1.5%

**A path constraint would remove 1.5-4.2% of labels.** Real, but small — its
entire upside is bounded there. The median labelled bar drops 0.29% on its way
to +2%.

## 2. The label certifies a move the exit cannot harvest

    label target        2.0%   (MIN_BUY_GAIN_THRESHOLD 0.02)
    cexit_take_profit   2.4%
    base sell_params    3.2%

A labelled bar reaching +2.0% triggers nothing, because the exit needs +2.4%.
The trade is held PAST the point the label promised:

    must reach   reachable   never reached   killed@-4%  killed@-5%  killed@-6%
       2.0%        129,003            0.0%         4.2%        2.5%        1.5%
       2.4%        112,770           12.6%         3.9%        2.3%        1.4%
       3.2%         85,880           33.4%         3.3%        2.0%        1.3%

**12.6% of labelled bars never reach the take-profit at all.** Those trades run
until a stop, an unclog or max_hold closes them — which matches the oracle's
19.8% stop rate far better than the path story does.

## 3. But ALIGNMENT is not the mechanism — refuted

If misalignment were the cause, aligning should fix it. It does not. Oracle,
guards open, summed over three windows:

    label / take-profit                summed    delta
    2.0% / 2.4%   MISALIGNED            98.59       --
    2.0% / 2.0%   aligned by lowering   99.50    +0.91
    2.4% / 2.4%   aligned by raising   142.79   +44.20
    2.6% / 2.4%   label above          159.69   +61.10
    3.2% / 2.4%   label well above     198.47   +99.88
    3.2% / 3.2%   aligned high         202.13  +103.54

**Lowering the take-profit to meet the label is worth +0.91pp — nothing.** And
once the label is high the take-profit barely matters (3.2/2.4 vs 3.2/3.2 differ
by 3.66 on a ~200 base). The driver is the LABEL THRESHOLD ALONE, monotonically
— the same relationship already found at 0.007 -> 0.02, extended further. The
"label >= take_profit" framing was a plausible story the data does not support.

## 4. Does a real model realise it? Mostly NO

Trained at 0.024 and 0.026, two seeds each, operating point RE-SWEPT per label
(a fixed threshold across labels is confounded):

    label / seed        W1 tr  W1 ret%  W2 ret%  W3 ret%   summed
    0.020 / 42 CONTROL    431     6.94    16.91    10.25    34.10
    0.020 /  7 CONTROL    413     2.97    11.15    11.88    26.00
    0.024 / 42            355     8.39    14.19     4.88    27.47
    0.024 /  7            330     7.68    12.48     9.68    29.84
    0.026 / 42            304     8.05    10.07    10.48    28.60
    0.026 /  7            232    10.32     7.22     5.12    22.66

    PAIRED delta vs 0.020 at the same seed:
      0.024/42   +1.46  -2.72  -5.37   -6.63
      0.024/ 7   +4.71  +1.33  -2.20   +3.85
      0.026/42   +1.11  -6.83  +0.23   -5.50
      0.026/ 7   +7.35  -3.93  -6.76   -3.33

**W1 improves 4 of 4** (+1.11 to +7.35, mean +3.66pp) — and W1 is the stated
acceptance window. **Summed improves 1 of 4**, with W2/W3 mostly degrading.

`val_mcc` fell monotonically as predicted: 0.6078 -> 0.567/0.561 -> 0.544/0.541.

**Trade counts fall 20-45%** (431->355->304 at seed 42). Volume is the entire
reason this operating point was adopted — it is what makes effects measurable —
so trading it away for a W1 gain undermines the configuration's rationale.

## Verdict — HOLD at 0.020

The W1 evidence is genuinely 4/4 and is not dismissed. But "wins the acceptance
window, loses the others, loses volume" is the profile this repo has repeatedly
found non-persistent — the same shape as the `NoRes` architecture arm and the
seed42/N4 ponder arm, each of which posted the best single window inside a
clearly negative result. And the ceiling gain that motivated the test (+44 to
+61pp) manifestly did not transfer: the model realises a few points at best.

A third seed per setting would settle it — 4/4 on W1 against a consistent summed
loss is ambiguous at two seeds per setting.

**0.020 remains the framework default**, so the pending NNMT feature inherits a
settled value rather than a contested one.
