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

---

# Follow-up 1: can the model rank at all?

Guard chain neutralised so the model has room to discriminate, `prediction_threshold` swept.

    W1 (OUT-OF-SAMPLE)                  W3 (in-sample, distant era)
    thresh  trades   ret%  win%   PF    thresh  trades   ret%  win%   PF
    none      6015 -42.12  58.2  0.88   none      5745  -5.38  60.5  0.98
    0.0       2357 -22.63  59.0  0.86   0.0       2064  10.82  62.4  1.09
    0.5       2313 -23.56  58.7  0.85   0.5       2033  12.40  62.3  1.10
    0.69      1970 -14.46  57.9  0.90   0.69      1712  19.29  63.9  1.20
    0.8       1586 -12.19  58.2  0.90   0.8       1374  19.77  65.1  1.25

**The model ranks IN-SAMPLE and not OUT-OF-SAMPLE.** On W3 win rate and PF rise
monotonically with confidence (60.5 -> 65.1%, 0.98 -> 1.25) — real ranking skill.
On W1 the win rate is FROZEN at 57.9-59.0% across the whole range while trade
count falls 2357 -> 1586. The confidence score carries no information about win
probability on unseen data. This is a clean generalisation failure, not an
inability to learn.

# Follow-up 2: the ceiling — what would a PERFECT model buy?

`NNNC/NNNC_Oracle.py` replaces the model's entry opinion with the TRAINING LABEL
itself (deliberate lookahead). Identical guards, stops, exits, ROI.

    arm                                W1 tr  W1 ret%  W1 PF  W1 win%   W3 tr  W3 ret%
    GUARDS ON   no model at all           99     9.67   2.13     72.7      62     3.68
    GUARDS ON   real model (production)   95    10.45   2.34     74.7      56     3.37
    GUARDS ON   PERFECT model             89    11.12   2.71     75.3      55     4.41
    GUARDS OFF  no model at all          6015   -42.12   0.88    58.2    5745    -5.38
    GUARDS OFF  real model               1970   -14.46   0.90    57.9    1712    19.29
    GUARDS OFF  PERFECT model            1760   -18.89   0.84    59.3    1588   -11.69

Two findings, both large.

**1. Model improvement is capped at +0.68pp.** With guards on, the whole span
from no model at all to a perfect lookahead model is 1.45pp on W1. The real
model already captures 0.78pp of it. **No model, GAN, label or architecture
change can be worth more than about seven tenths of a point at this gate.** That
retro-explains every null in this study far better than any hypothesis about
absorption: there was never more than a point on the table.

**2. A PERFECT model LOSES MONEY once the guards are removed** — −18.89% on W1,
*worse* than the real model's −14.46%, and −11.69% on W3 where the real model
makes +19.29%. Perfect foreknowledge of the label, traded through this exit
stack, is unprofitable.

So the guards are not filtering bad model predictions. **They are filtering out
bars where the LABEL ITSELF is not profitable to trade**, and they are the only
thing standing between this strategy and a loss. The label and the exit
machinery disagree about what a good trade is; the guard chain silently repairs
the disagreement by admitting only the ~95 bars per window where they happen to
coincide.

The oracle validates itself on the guards-on rows: better PF (2.71 vs 2.34) and
win rate (75.3 vs 74.7) than production, exactly as a superior signal should.
The guards-off rows are therefore a real property of the label, not a broken arm.

## What this changes

Stop tuning the model, the labels-as-learning-target, and the guards. The
binding problem is that **the training label does not describe a trade this exit
stack can profit from.** The productive question is the one the oracle makes
measurable for the first time: change the label or the exits until
`oracle GUARDS OFF` is PROFITABLE. Only then is there an edge for a model to
learn, and only then does model quality pay.

That is now a cheap inner loop — the oracle needs no training, so label and exit
variants can be evaluated in minutes each.

## Note on Debug/DebugNNStrategy

The perfect-prediction idea came from `Debug/DebugNNStrategy.py`, but that class
carries its own `populate_entry_trend` whose guard block has DRIFTED from
`BaseStrategy`: it omits the `rvol` guard entirely and applies `atr_pct` outside
the `entry_enable_guards` block. rvol is worth -3.42pp on W1, so an oracle built
on that path is not comparable with production. `NNNC_Oracle` subclasses
`NNNC_MLX` instead and overrides only `get_entry_conditions`, so the entry path
is byte-for-byte production. DebugNNStrategy's drift is worth repairing if it is
used for anything quantitative.

---

# Follow-up 3 (feature 004): the headroom conclusion, and what the oracle found

## T025 — Model headroom, and what it forecloses

**On W1, with the guard chain at its production settings, the entire span from
no model at all to a perfect model is 1.45pp** (9.671% → 11.122%), of which the
real model already captures 0.775pp, leaving **0.676pp** for any conceivable
model improvement. Measured 2026-09-05 on `NNNC_MLX`, W1 `20250901-20260831`.

This forecloses, at the CURRENT gate:

- model architecture changes
- GAN augmentation of any kind or fidelity
- label changes evaluated *for learnability*
- any intervention whose effect reaches the trades only through the model's
  entry opinion

None can be worth more than ~0.7pp, and the study's power floor at ~95 trades is
about 1pp. **Such a change is not measurable, let alone valuable.**

**The figure is specific to the gate.** A configuration admitting materially
more trades has a different, unmeasured ceiling — and feature 004 found exactly
such a configuration (below), so this conclusion must be re-measured if the
label changes.

## T026 — The test a future model or GAN proposal must pass

Before any training budget is spent on this family, the proposal must show that
**the oracle's guards-off return improves under the proposed configuration** —
`regime/ORACLE_GATE.md`, PASS = W1 guards-off return > 0 with at least 200
trades. A proposal that does not move the ceiling cannot move the realised
return, because the ceiling is what the realised return is bounded by.

This is answerable in minutes without training. A proposal that has not been
through it should not be run.

## What feature 004 then found

Applying that gate to the label itself:

    variant                    tr     ret%     PF   win%    vs baseline
    BASELINE (label 0.007)   1760   -18.89   0.84   59.3          0.00
    label threshold 0.02     1333   +26.54   1.37   70.4        +45.44
    label threshold 0.03     1022   +46.60   1.96   80.3        +65.49

**`MIN_BUY_GAIN_THRESHOLD` = 0.007 is the defect.** A 0.7% target against a
−4%..−6% stop plus round-trip fees cannot pay however perfectly it is
predicted. Raise it to 0.02 and the identical exit stack, on the identical
bars, turns +45pp at the ceiling — on all three windows.

The 0.68pp headroom above is therefore a statement about the CURRENT label, not
about this strategy family for all time. It is why the guard chain has to be so
severe: it is compensating for a label that does not describe a payable trade.

**These are lookahead ceilings, not achievable returns.** The next step is to
train `label 0.02` and find out whether a real model can realise any of it —
which is precisely the question feature 003 could not answer for the breakout
arm, because it never measured its ceiling.
