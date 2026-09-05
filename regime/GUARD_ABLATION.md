# NNNC guard ablation — does the guard chain, or the model, pick the trades?

Run 2026-09-05 on `NNNC_MLX` (production non-GAN base), guards at their tuned
values, `prediction_threshold` 0.69. W1 `20250901-20260831` is out-of-sample;
W2/W3 fall inside training and are reported but excluded from any verdict.

## Headline

**The guard chain is the strategy. The model's entry opinion is worth less than
the study can resolve, and is net-negative in-sample.**

    arm                             W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%
    model + guards (production)        95    10.45     181    10.31      56     3.37
    guards ONLY, model bypassed        99     9.67     196    11.44      62     3.68
    model contribution (pp)                  +0.78            -1.12           -0.31

W1 full stats: production 95tr / +10.45% / PF 2.34 / win 74.7% / DD 1.50%;
guards-only 99tr / +9.67% / PF 2.13 / win 72.7% / DD 1.51%.

+0.78pp is BELOW the ~1pp power threshold on ~95 trades, and both in-sample
windows are NEGATIVE. Removing the neural network's entry vote entirely — while
changing nothing else — reproduces the strategy.

This explains the whole preceding study at a stroke: six model-side changes
giving byte-identical trades, label economics failing to predict return, and EV,
precision, `val_mcc` and learnability MCC each failing as P&L proxies. None of
those were anomalies. **The model was never driving entry selection.**

### How to measure this correctly

Lowering `prediction_threshold` does NOT bypass the model.
`argmax_with_threshold` returns the argmax class whenever `max_prob > threshold`,
so at threshold 0.0 the model still rejects every bar whose argmax is HOLD; that
measures the CONFIDENCE gate, not the model. `NNNC/NNNC_NoModel.py` overrides
`get_entry_conditions` to accept every bar (calling `super()` first, or
`process_predictions` never runs and the exit side raises
`KeyError: 'predict_buy'`).

## Leave-one-out: which guards are load-bearing

Each guard neutralised alone, `volume > 0` always ON, everything else fixed.

    arm                      W1 tr  W1 ret%  W1 DD%   W1 PF   W1 delta   trades
    CONTROL                     95    10.45    1.50    2.34      0.00        --
    -guard_metric               95    10.45    1.50    2.34     +0.00        +0
    -bb_width                  124    10.27    2.42    1.95     -0.17       +29
    -close_norm                100     9.44    1.50    2.06     -1.01        +5
    -rvol                      149     7.02    3.52    1.49     -3.42       +54
    -adx                       290     4.49    5.72    1.12     -5.95      +195
    -atr_pct                   253     3.11    5.01    1.12     -7.33      +158
    ALL 6 OFF (volume ON)     1970   -14.46   25.68    0.90    -24.91     +1875

**`guard_metric` is completely inert** — byte-identical trades and returns on all
three windows, zero extra trades. It is the eighth recorded parameter that is
accepted, applied and does nothing. It can be deleted with no effect. (The tuned
`0.3` sits inside a `[-1, 0.5]` space while `guard_metric` spans `[-1, 1]`; at
this operating point no admitted bar ever reaches it.)

**No guard is dead weight — none can be removed for gain.** Every other removal
costs W1 return. `atr_pct` and `adx` are the heaviest lifters.

**The guards are strongly super-additive.** Individual removals cost −0.17 to
−7.33pp, summing to about −17.9pp, but removing all six costs −24.91pp. They are
not independent filters.

**Every guard's effect flips sign across eras** (except the inert one):

    arm                W1       W2       W3    consistent?
    -bb_width       -0.17    +4.70    -2.54    NO
    -close_norm     -1.01    +1.36    +0.51    NO
    -rvol           -3.42    -1.75    +0.91    NO
    -adx            -5.95    -0.49    +4.64    NO
    -atr_pct        -7.33    -1.30    +0.18    NO
    ALL 6 OFF      -24.91   -21.21   +15.92    NO

In W3 (2023) removing the guards HELPS, by a lot. The chain is tuned to
W1/W2-like conditions, matching the era-dependence seen in the day-of-week and
dynamic-lookback studies.

## A hypothesis of mine that was REFUTED

I proposed that the historical "guards off destroys the strategy" result was
confounded, because `volume > 0` sits inside the same all-or-nothing
`entry_enable_guards` block and is a data-sanity check rather than a quality
guard. It is not a material confound: all-six-off with volume ON is −14.46% and
`entry_enable_guards=False` (volume off too) is −14.65%, the same 1970 trades.
The volume check admits essentially nothing extra. The collapse is real and is
caused by the quality guards.

## Harness defect found and fixed

`run_arm.py` wrote metrics from `latest_result()` whether or not the backtest
succeeded. A broken arm produced three windows all reporting an identical
59 trades / +3.86% — a stale file from a previous run, and entirely plausible
looking. It now refuses to record metrics when the backtest fails, or when a
"successful" run wrote no new result file. An audit found 4 of 482 arm records
affected, all four from the broken arm in this session; no prior conclusion is
contaminated.

## What this means

The open question is no longer "which guards are better". It is **why a model
that trains to val_mcc 0.5-0.7 adds nothing to a seven-condition hand-crafted
filter.** Options, cheapest first:

1. Measure entry-set overlap between guards-only and production directly — how
   many of the 95 production trades are in the 99 guards-only set? If overlap is
   near-total the model is decorative; if the sets differ but returns match, both
   are picking from an equally good pool and the edge is the guard chain's
   regime selection, not bar selection.
2. Test the model where the guards are NOT binding — the guards-off pool of 1970
   bars is where the model has room to discriminate. Its inability to rank there
   (win rate frozen at 58.2% across a 3.5x threshold range, recorded earlier) is
   the real defect and is now the most valuable thing to fix.
3. Treat the guard chain as the strategy and tune it directly, dropping the NN
   from the entry path. This is the honest reading of the table above, but it
   discards the model rather than fixing it.
