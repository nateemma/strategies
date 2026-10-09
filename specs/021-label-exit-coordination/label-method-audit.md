# T025 — the `LabelMethod` audit (FR-012)

**Date**: 2026-10-08. **Method**: read every labeler in `Framework/TrainingSignals.py`, then
verify each claimed reduction by CONTAINMENT on real data — if a label ever fires where its
candidate trigger does not, the reduction is wrong. SOL/USDT, 164,918 bars, 11 labeler/trigger
pairs, **zero violations**.

**29 members, not 28.** The spec says 28; `gbb_fwdmae` was added during this feature's own
work, so the spec's count predates it.

## The question

A `LabelMethod` mixes a backward-looking entry condition with its own forward-looking outcome
test. Only the first half is a trigger. A member *reduces* if its mask factors as

```
mask = B(data up to and including this bar)  AND  F(data after this bar)
```

so that dropping `F` and conjoining `__outcome_positive` instead leaves the entry condition
untouched. Three outcomes are possible, and all three occur:

1. **Reduces** — a separable backward factor exists.
2. **Reduces to `empty`** — there is no entry condition at all; the member *is* an outcome
   definition, which this design supersedes rather than ports.
3. **Does not reduce** — an entry condition exists but is NOT causal, so porting it would put
   lookahead on the leaderboard.

## Research R7 was wrong on three of its five predictions

R7 predicted `mfe_mae_ratio`, `future_sharpe`, `future_sortino`, `future_expectancy` and
`risk_adj_future` would not reduce, "their conditions being purely forward-path properties".

Correct for `mfe_mae_ratio` and `risk_adj_future`. **Wrong for the three `future_*` members**,
which all end with

```python
adx_scaled = df.get("adx_scaled", pd.Series(np.zeros(n)))
labels = np.where(adx_scaled > -0.2, labels, 0)
```

described in the live code as filtering out sideways markets. That is a pure function of the
current bar, so it is a trigger — and it is **binding, not decorative**: it blocks **38.35%**
of bars pooled over five pairs (28.5% on AAVE to 48.8% on XRP). It cannot be waved away as
inert the way `gbb`'s `rvol > -1.0` clause legitimately can.

Two defects in that clause, reported not fixed (label-side, outside this feature's scope):

- **It sits INSIDE the `for t in range(n)` loop**, re-reading `adx_scaled` and rewriting the
  whole `labels` array on every one of n iterations. The final result is correct — the last
  iteration's `np.where` governs the whole array — but it is O(n²) work for an O(n) filter, in
  three separate labelers.
- `df.get("adx_scaled", zeros)` means a frame without the column silently gets **zeros**, and
  `0.0 > -0.2` is True, so the filter becomes a no-op rather than an error. A missing column
  changes the label and says nothing.

## The audit table

| member | verdict | backward factor | notes |
|---|---|---|---|
| `forward_mae` | → `empty` | none | the no-trigger case by construction; the arm that lost to cash |
| `triple_barrier` | → `empty` | none | purely forward barriers |
| `quantile_future` | → `empty` | none | forward return quantile |
| `mfe_mae_ratio` | → `empty` | none | **R7 correct** |
| `risk_adj_future` | → `empty` | none | **R7 correct** |
| `multi_horizon_vote` | → `empty` | none | votes over forward horizons |
| `optimal_signals` | → `empty` | none | forward high/low window |
| `local_extrema` | → `empty` | none | forward window `close[t:t+h]` only |
| `future_sharpe` | **reduces** | `adx_scaled > -0.2` | **R7 WRONG**; registered as `adx` |
| `future_sortino` | **reduces** | `adx_scaled > -0.2` | **R7 WRONG** |
| `future_expectancy` | **reduces** | `adx_scaled > -0.2` | **R7 WRONG** |
| `gbb` | **reduces** | `guard_metric<0.3 & bb_width>0.035 & rvol>-1.0` | registered as `gbb`; bit-identical port |
| `gbb_path` | **reduces** | same as `gbb` | its extra `survives` clause is a forward stop-aware path test |
| `gbb_fwdmae` | **reduces** | same as `gbb` | `gbb AND forward_mae`; both halves outcome-side but the trigger is gbb's |
| `breakout` | **reduces** | `close > high.rolling(20).max().shift(1)` | registered as `breakout`; cleanly causal |
| `breakout_tb` | **reduces** | same as `breakout` | forward triple-barrier on top |
| `breakout_vol` | **reduces** | `breakout & rvol > min` | |
| `breakout_squeeze` | **reduces** | `breakout & squeeze` | |
| `breakout_bb` | **reduces** | `bb_position & bb_width` thresholds | |
| `breakout_gbb` | **reduces** | `guard_metric>0.2 & bb_width>0.035` | registered; **guard direction OPPOSITE to `gbb`'s** |
| `breakout_consensus` | **reduces** | `votes >= min_indicators_buy` | vote family |
| `technical_indicators` | **reduces** | `votes >= min_indicators_buy` | vote family |
| `indicators2` | **reduces** | `votes >= min_indicators_buy` | vote family |
| `indicators3` | **reduces** | `votes >= min_indicators_buy` | vote family |
| `indicators4` | **reduces** | `votes >= min_indicators_buy` | vote family |
| `trends` | **reduces** | `buy_trend_mask` over regime/momentum/flow/risk | |
| `bands` | **reduces** | band crossing, `prev_close` vs `prev_band` | backward by construction |
| `local_min_followthrough` | **DOES NOT** | `is_local_min`, **centred window** | `close[t-k : t+k+1]` — forward by k |
| `geometry` | **DOES NOT** | `find_peaks` whole-series + 3-bar backfill | four separate defects; see below |

**Totals**: 8 reduce to `empty`, 19 reduce with a real backward factor, 2 do not reduce.

## The two that do not reduce — and they fail the same way

Neither is "purely forward-path" in R7's sense. Both have a genuine entry condition; both
compute it with information from after the bar.

- **`local_min_followthrough`** — `is_local_min[t]` is true when `close[t]` is the minimum of
  `close[t-k : t+k+1]`, a **centred** window. A bar's eligibility depends on the k bars after
  it.
- **`geometry`** — worse, and in four ways: it **raises with its own defaults**
  (`price_mean * min_gain` with `min_gain=None`); its **DWT smoothing is commented out**
  (`smoothed = close`, while `_dwt_smooth` sits defined and unused); `find_peaks` scans the
  **whole** array and the loop labels the **three bars preceding** each valley; and
  `prominence` comes from whole-series `mean`/`std`, making it both lookahead and **per-pair**
  (Constitution X). Measured: 272 of 500 bars (54%) at `min_gain=0.01`.

  The forward dependence is **measured, not inferred** — perturbing every bar after index 200
  changes the mask before index 200, and that control is a permanent test
  (`test_the_live_labels_geometry_FAILS_that_same_test`).

Both are fixable the same way: confirm the extremum k bars later, which is causal and
tradeable. That is exactly what the registered `geometry` trigger does. It fires 4 bars after
a local low at least 1% deep, base rate 3.07% against the labeler's 54%.

## Registered triggers, measured

SOL/USDT, 164,918 bars, against the production outcome table (`config_hash
247a8700f0b08f07`). Precision = share of triggered bars whose outcome is positive.

| trigger | base rate | label rate | precision |
|---|---|---|---|
| `empty` | 100.000% | 56.188% | 56.19% |
| `adx` | 54.802% | 31.128% | 56.80% |
| `gbb` | 15.086% | 8.672% | 57.48% |
| `breakout_gbb` | 10.974% | 6.455% | 58.82% |
| `breakout` | 6.403% | 4.196% | **65.53%** |
| `geometry` | 3.069% | 1.763% | 57.45% |

## A prediction recorded BEFORE T027 runs

`breakout` has the highest precision of any trigger, **+9.34pp over the null**. It is also the
trigger this repository has most firmly established as **learnable but not tradeable**: MCC
0.77 while losing money on every timeframe and exit tested, and −17.55% on a re-test after the
exit defect was repaired.

Those two facts cannot both be flattering. So, stated in advance and falsifiable:

> **If T027's economic value ranks `breakout` top, the economic-value column is not measuring
> what it claims to measure.** A trigger already known to lose money must not come out best on
> a column intended to identify collectable moments. If it does, the defect is in the column —
> most likely because the mean realised outcome is dominated by hit RATE while the losses are
> larger than the wins — and the leaderboard cannot be trusted until that is resolved.

This is why `breakout` was registered at all: it is the one row whose right answer is already
known, which makes it the instrument's own calibration check. A leaderboard with no such row
cannot be audited.

The complementary observation: precision is otherwise nearly FLAT (56.19% → 58.82% across
`empty`, `adx`, `gbb`, `breakout_gbb`, `geometry`), while base rates span 3% to 100%. If
economic value turns out equally flat, that is evidence for explanation **(b)** — the trigger
as decoration, selecting fewer moments rather than better ones. Not yet a verdict: precision
on a binary is not economic value, and a trigger can hold its hit rate while improving
magnitude. T027 measures magnitude, T028 gives it a block-bootstrap interval, and only then
does FR-014b's rule apply.

## Also noted for T029

`geometry`'s label rate is **1.76%**, below the ~5.5% line at which all-Hold collapse has been
seen in this repository. Its probe row may come back collapse-limited rather than
information-limited, which would be a property of the label's sparsity and not of the
trigger's quality. `labels.label_rate` exists so that is checkable before a training run.

---

# T027 — economic value measured, and the prediction resolved

Pooled over 11 pairs, 1,692,615 bars, outcome table `247a8700f0b08f07`.

## The prediction fired, and it found a real defect — but not the one it named

I recorded, before running: *"if T027's economic value ranks `breakout` top, the
economic-value column is not measuring what it claims to."* `breakout` did rank top. The
prediction earned its keep, but its stated conclusion is **not** established. What it
actually surfaced:

**Defect found: the all-bars mean is not comparable across triggers.** FR-014 asks for "the
economic value of trading every triggered bar", and that quantity is weighted by how often a
trigger REPEATS ITSELF. Triggered bars arrive in clusters, so one favourable episode gets
re-sampled many times. Measured clustering (distinct non-overlapping opportunities per
triggered bar):

| trigger | triggered | non-overlapping | ratio | EV all bars | EV non-overlapping |
|---|---|---|---|---|---|
| `breakout` | 109,200 | 27,102 | 0.248 | +0.002296 | **+0.003316** |
| `breakout_gbb` | 211,423 | 21,145 | 0.100 | −0.001024 | +0.000251 |
| `empty` | 1,692,097 | 45,103 | 0.027 | −0.001130 | −0.000993 |
| `geometry` | 48,528 | 24,709 | 0.509 | −0.000562 | −0.000706 |
| `adx` | 1,036,117 | 38,873 | 0.038 | −0.000818 | −0.000847 |
| `gbb` | 302,445 | 22,420 | 0.074 | +0.000822 | **−0.001262** |

`gbb` clusters 13.5-to-1 and **flips from positive to negative** under the correction, from
second place to last. So the non-overlapping figure is now computed as a first-class
statistic, per pair (the scan is only meaningful within one contiguous series) and then
pooled. Greedy EARLIEST-first, not best-first: choosing the best subset would be a lookahead
selection of its own.

Correcting it did **not** dethrone `breakout`, so the prediction's conclusion fails.

## The real limitation, and the measurement that found it

Ranked on the comparable figure, the column **inverts the harness**. `gbb` — the incumbent
returning **+42.71pp** over 15 pooled folds — ranks LAST, below the `empty` null.
`breakout`, which loses money, ranks first.

The resolving measurement: the arm's own 1,420 entries, looked up in the table.

| | value |
|---|---|
| arm entries that are `gbb`-triggered | 1,390 / 1,420 = **97.9%** |
| table EV over the ARM's entry bars | **+0.005204** |
| table EV over ALL `gbb` bars | **+0.000769** |
| arm bars as a share of `gbb` bars | **0.514%** |
| win rate, arm vs all `gbb` | 65.92% vs 59.45% |
| median, arm vs all `gbb` | +0.023308 vs +0.009600 |

**A selector picks 0.514% of `gbb`'s bars and gets 6.8× the economic value.** The trigger is
the gate — 97.9% of entries are inside it — but the trigger does not pick WHICH of its bars
to trade, and that choice is where the value is.

So the column is not broken: it measures exactly what it claims, the EV of trading every
triggered bar. It is INCOMPLETE as a way to rank triggers for use in a model-based strategy,
because a trigger's no-model EV is a LOWER bound on what a selector can extract from it, and
the gap here is 6.8×. A leaderboard row cannot be read as "this trigger would earn this".

## This bears directly on the feature's central question

The spec frames two rival explanations: **(a)** the trigger selects collectable moments, or
**(b)** the trigger hands the model an easy target and the model is decoration.

The measurement fits NEITHER. Against (a): `gbb`'s trigger has essentially no economic value
on its own (−0.001262, below the null). Against (b): the model is emphatically not
decoration — it finds a 0.5% subset worth 6.8× the average. What the evidence supports is a
third reading the spec did not enumerate:

> **The trigger defines a population; a selector discriminates within it; the value is in the
> discrimination.** The trigger's job is not to be collectable on average but to be a
> population a selector can work in.

That is consistent with the earlier guard-ablation finding rather than contradicting it. At
the GUARDED operating point the guards reject ~97% and the model adds nothing on top; at this
guards-OPEN point the model's per-fold threshold is the only selector there is. Either way
something downstream of the trigger does the choosing — which is why no trigger-only column
can reproduce the harness ranking.

## Do not interpret the ranking yet

Every gap in the table above is between 1e-4 and 4e-3, with **no confidence intervals**. The
clustering ratios (0.027 to 0.509) and mean holds (32–42 bars) mean an i.i.d. bootstrap would
understate the intervals badly — which is now an empirical justification for T028's block
bootstrap rather than an a-priori one, and sets its block length at ≥ the mean hold. Until
T028 supplies intervals, the only safe reading of the ranking is that `breakout` and `gbb`
sit at opposite ends of it, in the opposite order to the harness.

---

# T028 — intervals, and what they permit

Paired block bootstrap, `L=48` bars, non-overlapping selection, vs the `empty` null. Pooled
11 pairs. See research.md's R9 correction for why `L=48` and why these intervals behave the
opposite way to R9's prediction.

| trigger | delta | 95% CI | spans 0? |
|---|---|---|---|
| `breakout` | +0.004309 | [+0.004003, +0.004632] | **no** |
| `breakout_gbb` | +0.001244 | [+0.000902, +0.001596] | **no** |
| `geometry` | +0.000287 | [−0.000050, +0.000613] | yes |
| `adx` | +0.000146 | [−0.000029, +0.000326] | yes |
| `gbb` | −0.000269 | [−0.000610, +0.000078] | yes |

Fillable-only (FR-009b), the population that could actually have been traded:

| trigger | delta | 95% CI | spans 0? |
|---|---|---|---|
| `breakout` | +0.010348 | [+0.009705, +0.010988] | **no** |
| `breakout_gbb` | +0.004643 | [+0.004030, +0.005311] | **no** |
| `geometry` | +0.002482 | [+0.001377, +0.003591] | **no** |
| `adx` | +0.000146 | [−0.000173, +0.000464] | yes |
| `gbb` | **−0.004102** | **[−0.004793, −0.003402]** | **no (negative)** |

## What this does and does not license

It is now resolved that the ranking is not noise. `breakout` beats the null and `gbb` does
not, and on the fillable population `gbb` is **significantly WORSE** than trading every bar.

It is NOT resolved that `breakout` is a better trigger to build a strategy on. T027's
measurement stands: the arm's model-selected 0.514% subset of `gbb` bars realises +0.005204
against +0.000769 over all `gbb` bars, a factor of 6.8. A trigger's no-model economic value
is a LOWER bound on what a selector extracts from it, and nothing here measures the upper
bound. `gbb` returning +42.71pp in the harness while scoring below the null on this column is
not a contradiction — the two measure different things, and T027 named which.

The verdict rule (T030) therefore has a real decision to make rather than a formality: it
must assign a verdict on a column that is significant, comparable, and known to be an
incomplete predictor of strategy value.

---

# T029 — the learnability column, and why it cannot do its job as specified

54 fits (6 triggers x 3 windows x 3 seeds), RIDGE, split set v1, ind24, seq_len 16.
**0 failures, 0 collapsed.** Spread across both windows and seeds, per R9.

| trigger | mean MCC | SE | vs `empty` |
|---|---|---|---|
| `gbb` | +0.5927 | 0.0036 | +0.5585 |
| `breakout_gbb` | +0.5622 | 0.0074 | +0.5280 |
| `breakout` | +0.5440 | 0.0052 | +0.5098 |
| `adx` | +0.3883 | 0.0022 | +0.3540 |
| `geometry` | +0.3286 | 0.0035 | +0.2944 |
| `empty` | **+0.0342** | 0.0059 | — |

The `geometry` collapse risk flagged in T024 did NOT materialise: 9/9 fits trained, none
collapsed, despite a 1.76% label rate. Worth recording as a prediction that failed.

## The column measures TRIGGER RECOVERY, not outcome prediction

`empty`'s label is purely `__outcome_positive` — "will a trade opened here be profitable" —
and it scores **+0.0342**, indistinguishable from nothing. That is the clue, and one fit per
trigger settles it. Same window, same seed, three different labels:

| label | MCC |
|---|---|
| `gbb` trigger ALONE (no outcome) | **+0.7582** |
| `gbb` AND positive outcome | +0.5906 |
| `breakout` trigger ALONE | **+0.6801** |
| `breakout` AND positive outcome | +0.5463 |
| `geometry` trigger ALONE | **+0.4189** |
| `geometry` AND positive outcome | +0.3224 |
| positive outcome ALONE | **+0.0115** |

Three facts, and together they are decisive:

1. **A trigger alone is highly learnable** (0.42–0.76) — necessarily, since a trigger is a
   deterministic function of the very feature columns the probe is given. `gbb` is
   `guard_metric < 0.3 AND bb_width > 0.035`, and both columns are in `ind24`.
2. **The outcome alone is unlearnable** (+0.0115). A linear probe on 24 indicators cannot
   predict whether a trade opened at a bar will profit under the production exit.
3. **The conjunction scores BELOW the trigger alone**, by 0.10 to 0.17. Adding the outcome
   makes the label HARDER, because it adds a component the probe cannot predict.

So a trigger-defined label's MCC is essentially its trigger's own learnability, discounted
slightly by the unlearnable outcome noise. The column is dominated by a tautology.

## Consequence for FR-015a and the verdict rule

The spec's explanation **(b)** is "a trigger hands the model an easy target it can partly
read off the trigger's own input columns, so the apparent edge is the entry rule and the
model is decoration". Measured: that is **true of every trigger, by construction**, and the
magnitude is just how learnable the trigger is. The learnability column therefore cannot
DISCRIMINATE between triggers on explanation (b) — it ranks them by how easily a linear
model recovers a threshold rule, which is not a fact about trading.

This is the same thing feature 018 recorded as "the gbb label is half a threshold rule —
trigger-only beats the model with 100% recall by construction", arriving here from the
other direction and quantified.

FR-014b's rule says `decoration` when economic value does not exceed `empty`'s while
learnability does. Under this measurement **every** trigger's learnability exceeds `empty`'s
by 50–100 standard errors, so the rule's `decoration` branch will fire on any trigger that
fails the economic-value test, for a reason that has nothing to do with the trigger's
quality. The rule remains total, but its second branch carries no information.

## The informative quantity, and what it would take

What the column should ask is **conditional** learnability: among the bars the trigger
fires on, can a model separate the profitable from the unprofitable? That is a fact about
whether the trigger defines a population a selector can work in — which is exactly the
third explanation T027's measurement pointed at, where the arm's model-selected 0.514%
subset of `gbb` bars realised 6.8x the average.

`research/clfstudy/metrics.py` already has this machinery: `trigger_masks` and `score`'s
`buy_mask`/`sell_mask` compute metrics CONDITIONAL on the gbb trigger firing, added for
precisely this reason. The probe does not currently use it, because FR-015 specifies the
label's learnability and that is what was built.

**Recommended amendment**, not taken unilaterally: add a conditional-learnability column
(probe restricted to triggered bars), and either demote the unconditional column to a
diagnostic or drop it. This changes what FR-015/FR-014b measure, so it is a spec decision
rather than an implementation one.

---

# FR-015b — conditional learnability, the amended column

54 fits, same grid (6 triggers x 3 windows x 3 seeds), same RIDGE / v1 / ind24 / seq_len 16.
0 failures, 0 collapsed. The probe is trained AND scored only on the bars the trigger fires
on, with the label being `__outcome_positive` alone — so the trigger is no longer in the
label and cannot be recovered from the features.

| trigger | uncond MCC | **cond MCC** | cond SE | cond − empty | > SE? |
|---|---|---|---|---|---|
| `breakout` | +0.5440 | **+0.0546** | 0.0072 | +0.0204 | yes |
| `breakout_gbb` | +0.5622 | +0.0528 | 0.0025 | +0.0186 | yes |
| `gbb` | +0.5927 | +0.0518 | 0.0092 | +0.0175 | yes |
| `adx` | +0.3883 | +0.0349 | 0.0084 | +0.0007 | no |
| `empty` | +0.0342 | +0.0342 | 0.0059 | 0 | — |
| `geometry` | +0.3286 | **+0.0200** | 0.0044 | **−0.0142** | yes, NEGATIVE |

## What the amendment bought

**1. It confirms the unconditional column was measuring trigger recovery.** Every figure
drops by an order of magnitude — 0.33–0.59 becomes 0.02–0.05 — once the trigger is taken out
of the label. Nothing else changed.

**2. The null now works without special-casing.** `empty`'s conditional and unconditional
MCCs are identical at +0.0342, because conditioning on a trigger that fires everywhere IS
the unconditional problem. That identity is a check on the implementation as much as a
property of the design.

**3. The column finally discriminates, and in a DIFFERENT order.** `breakout`,
`breakout_gbb` and `gbb` sit significantly above the null; `adx` is indistinguishable from
it; and `geometry` is significantly BELOW — conditioning on `geometry` makes the outcome
*less* predictable than it is at large. Under FR-015 every trigger beat the null by 50–100
standard errors and the ranking was uninformative; here the spread is 1–3 standard errors
and the ordering means something.

**4. It is not an artefact of class imbalance.** The conditional label is 56.7%–62.8%
positive in every population, better balanced than the conjunction label was. A near-zero
MCC on balanced classes is genuine unpredictability, not a degenerate majority.

## What it says, and the tension it leaves

**Conditional learnability is near zero for every trigger** (0.020–0.055 against a null of
0.034). Within the moments any of these triggers selects, a linear probe on 24 indicators
barely separates the profitable from the unprofitable. The ridge-cannot-express-AND caveat no
longer applies — the label is the outcome alone, with no conjunction — so this is a clean
measurement of a genuinely hard problem rather than a handicapped one.

The standing caveat is narrower but real: RIDGE is still LINEAR. And that matters here,
because it collides with T027:

> The arm's model-selected 0.514% subset of `gbb` bars realised **+0.005204** against
> **+0.000769** over all `gbb` bars — 6.8×. Something in production IS selecting well.

Both cannot be dismissed. Either the production model finds non-linear structure a linear
probe cannot (which would be the first evidence in this repository that the MODEL, not the
gate, does real work — against the standing "guard ablation: model adds NOTHING" finding),
or the arm's 0.514% subset is favourable for a reason other than predictive skill (its
threshold correlates with something the table already prices, or the subset is small enough
to be a selection artefact).

That is a well-posed question with a cheap answer available: re-run the conditional probe
with a NON-LINEAR architecture from the study's registry, on `gbb` only, and see whether
conditional MCC rises from 0.05 toward something that could support a 6.8× selection. It is
out of FR-015b's scope — the probe is fixed by design, and swapping its model is exactly
what FR-015 forbids — so it belongs as a separate, clearly-labelled diagnostic, not as a
second leaderboard column.

## Both columns now agree on the shape of the result

| trigger | EV vs null (fillable) | cond learnability vs null |
|---|---|---|
| `breakout` | +0.010348 (sig) | +0.0204 (sig) |
| `breakout_gbb` | +0.004643 (sig) | +0.0186 (sig) |
| `geometry` | +0.002482 (sig) | −0.0142 (sig, negative) |
| `adx` | +0.000146 (ns) | +0.0007 (ns) |
| `gbb` | −0.004102 (sig, negative) | +0.0175 (sig) |

They agree that `breakout` leads both and that `adx` is inert on both. They DISAGREE on the
two ends: `gbb` has the worst economic value and the third-best conditional learnability,
and `geometry` the reverse. Those two rows are where FR-014b's verdict rule has actual work
to do, and they are exactly the `selection value` / `decoration` distinction the feature was
built to draw.

---

# DIAGNOSTIC (2026-10-09) — the model's selection is reproducible WITHOUT a model, and beaten by it

Two measurements, run to resolve the tension T027 left: the arm's model-selected 0.470% of
`gbb` bars realised 6.8× the EV of all `gbb` bars, while a linear probe found almost no
conditional structure. Either production finds non-linear structure, or the subset is
favourable for another reason.

## 1. A non-linear model finds NOTHING either

Conditional probe, `arch=LSTM`, window A, 3 seeds, features and seq_len held fixed at the
probe's own (ind24 / 16) so ONLY the architecture differs. Labelled a diagnostic: FR-015
forbids varying the model between leaderboard rows, and `empty` is fitted under the same
arch so the number has a baseline.

| arch | `gbb` cond MCC | `empty` cond MCC (null) |
|---|---|---|
| RIDGE | +0.0518 | +0.0342 |
| **LSTM** | **−0.0060** | −0.0029 |

Not a broken probe — 12 epochs, loss falling 0.8882 → 0.8359, `val_mcc` ~0.02 and never
improving, early-stopped on no improvement. A genuine run that found nothing. (Checked
deliberately: a probe that trains for one epoch reports ~0 exactly like a real null, and
"a broken probe reports 0, same as a real null" is a recorded lesson here.)

So the non-linear hypothesis is **refuted**. This CONFIRMS the standing "guard ablation:
model adds NOTHING" finding rather than overturning it.

## 2. Tightening the trigger reproduces the selection — and beats it 5.6×

If the probe learns essentially the trigger (T029), then high model confidence should mean
"deep inside the trigger's region", not "likely to profit". Testable with no model and no
GPU: tighten `gbb`'s own thresholds and read the EV off the table. All-bars basis, matching
T027's arm comparison exactly.

| selection | n | % of gbb bars | EV | vs gbb |
|---|---|---|---|---|
| `gbb` as shipped (`guard<0.3`, `bb>0.035`) | 302,445 | 100.000% | +0.000822 | — |
| **the arm's model-selected subset** | **1,420** | **0.470%** | **+0.005204** | +0.004382 |
| `guard<−0.2`, `bb>0.08` | 33,665 | 11.131% | +0.005035 | +0.004213 |
| `guard<−0.4`, `bb>0.10` | 15,491 | 5.122% | +0.008333 | +0.007511 |
| `guard<−0.6`, `bb>0.12` | 7,132 | 2.358% | +0.012108 | +0.011286 |
| `guard<−0.7`, `bb>0.15` | 2,872 | 0.950% | +0.020680 | +0.019858 |
| **`guard<−0.8`, `bb>0.18`** | **1,178** | **0.389%** | **+0.029014** | +0.028192 |
| `guard<−0.9`, `bb>0.25` | 206 | 0.068% | +0.050938 | +0.050116 |

Two readings, both striking:

- At **23× less selective** than the arm (11.1% vs 0.470%), a bare threshold already matches
  the model's EV (+0.005035 vs +0.005204).
- At **matched selectivity** (0.389% vs 0.470%), the threshold gets **+0.029014** against the
  model's +0.005204 — **5.6× better, with no model at all.**

The selection is CAUSAL: `guard_metric` and `bb_width` are backward-looking columns, so a
tighter threshold is a legitimate trigger, not lookahead.

## What this says, and the limit on it

**The economic value is in the trigger's MARGIN, not its binary firing.** EV rises
monotonically across four orders of magnitude of selectivity, from +0.0008 at 100% to +0.051
at 0.068%. A binary trigger discards that information, and the model recovers only part of it
— worse than reading it off the thresholds directly.

That is a third answer the spec did not enumerate, and it sharpens T027's reading: the
trigger does not merely define a population a selector works in; **the trigger's own margin
IS the selector**, and the model is a lossy approximation to it.

**THE LIMIT, and it is a hard one.** This is table EV — a lookahead bound, not a realisable
return (FR-017). This repository has already measured the specific trap: *"Trigger
relaxation: return FALLS — ceiling rises monotonically, return doesn't"*, and *"Learnability
vs tradeability OPPOSED"*. Tightening is the opposite direction to that experiment, but the
lesson is the same shape: a rising table ceiling has repeatedly failed to become harness
return. At 0.389% of `gbb` bars the trade count would also collapse far below what the
pooled harness can resolve at ~3–4pp/yr.

So this is **not** a promotion candidate and must not be read as one. It is an explanation of
where the economic value sits. The only way to know whether a tightened trigger trades is the
pooled harness in `evaluation/`, and that is a separate experiment with a real risk of
reproducing the "ceiling rises, return doesn't" result for the fourth time.

---

# `gbb_tight` registered (2026-10-09)

The margin finding as a leaderboard row: `guard_metric < -0.8 AND bb_width > 0.18`.

**Chosen on selectivity, NOT on EV.** The rung ladder was explored by economic value, so
picking its best rung would make this row self-fulfilling — selection on the very outcome
being measured. The thresholds are instead the rung whose selectivity most nearly matches the
ARM's (0.389% of `gbb` bars against the arm's 0.470%), because the question this trigger
exists to ask is "can a plain threshold, at the model's own selectivity, do what the model
does". Selectivity-matching removes the worst of the selection effect, not all of it: the
ladder's shape was known before the values were picked, so the row is honest as a comparison
against the arm at equal selectivity and is not an unbiased estimate of an independently
chosen tight threshold.

| trigger | base rate | n scored | n non-ovl | EV all | EV n-ovl | EV fill n-ovl | win rate |
|---|---|---|---|---|---|---|---|
| `empty` | 100.000% | 1,692,097 | 45,103 | −0.001130 | −0.000993 | −0.001272 | 57.96% |
| `gbb` | 17.871% | 302,445 | 22,420 | +0.000822 | −0.001262 | −0.005374 | 59.36% |
| `breakout` | 6.453% | 109,200 | 27,102 | +0.002296 | +0.003316 | +0.009076 | 63.35% |
| **`gbb_tight`** | **0.070%** | **1,178** | **206** | **+0.029014** | **+0.019509** | **+0.015848** | **81.75%** |

Paired block bootstrap vs the `empty` null, non-overlapping, 0 degenerate resamples:

| trigger | all | fillable |
|---|---|---|
| `gbb` | −0.000269 [−0.000610, +0.000078] | −0.004102 [−0.004793, −0.003402] |
| `breakout` | +0.004309 [+0.004003, +0.004632] | +0.010348 [+0.009705, +0.010988] |
| **`gbb_tight`** | **+0.020502 [+0.011468, +0.029286]** | **+0.017120 [+0.003899, +0.029222]** |

Significant on both populations, and by far the largest effect on the board — an 81.75% win
rate against the null's 57.96%. Its interval is ~30× wider than `breakout`'s, correctly: 206
non-overlapping opportunities against 27,102. The bootstrap drew no degenerate resamples, so
the interval is meaningful rather than an artefact of sparsity.

**It also inverts `gbb` on the same column.** `gbb` is significantly WORSE than the null on
the fillable population (−0.004102) while `gbb_tight` is significantly better (+0.017120) —
the same two feature columns, the same exit, the same table, differing only in threshold
depth. That is the margin finding stated as a single comparison.

## The caveat travels with the row, and it is not decoration

206 non-overlapping opportunities across 11 pairs over the pooled span is roughly 45 a year.
Table EV is a lookahead bound, not a return (FR-017), and this repository has now twice
measured the specific failure this invites — *"trigger relaxation: return FALLS; ceiling
rises monotonically, return doesn't"* and *"feature 003 tail labelling REJECTED; r:r moved W1
non-monotonically"*. A third instance would not be surprising.

So `gbb_tight` enters the leaderboard as evidence about WHERE the economic value sits, with
`not a promotion candidate` written into its own registry description and asserted by a test.
The only thing that can promote it is the pooled harness in `evaluation/`.

---

# T030 — the verdict rule applied. THE ANSWER.

FR-014b as amended by FR-015b, against the `empty` null. Economic value on the FILLABLE
population (FR-009b — the bars that could actually have been traded), paired block
bootstrap; conditional learnability per FR-015b.

| trigger | verdict | EV vs null (95% CI) | cond MCC vs null (±1 SE) |
|---|---|---|---|
| `gbb_tight` | **selection_value** | +0.017120 [+0.003899, +0.029222] | +0.1148 ± 0.0549 |
| `breakout` | **selection_value** | +0.010348 [+0.009705, +0.010988] | +0.0204 ± 0.0093 |
| `breakout_gbb` | **selection_value** | +0.004643 [+0.004030, +0.005311] | +0.0186 ± 0.0064 |
| `geometry` | **selection_value** | +0.002482 [+0.001377, +0.003591] | −0.0142 ± 0.0074 |
| **`gbb`** | **decoration** | **−0.004102 [−0.004793, −0.003402]** | +0.0175 ± 0.0109 |
| `adx` | **indeterminate** | +0.000146 [−0.000173, +0.000464] | +0.0007 ± 0.0102 |

## The answer to the question the feature was built to settle

The spec's open question was whether a trigger earns its keep by **(a)** selecting moments
when a forward move is collectable, or **(b)** handing the model an easy target, so the
apparent edge is the entry rule and the model is decoration. It noted that no experiment run
so far separated them and that they imply opposite strategies.

**`gbb` — the production incumbent, the trigger behind the arm that returns +42.71pp over 15
pooled folds — is adjudicated `decoration`.** Its economic value is significantly NEGATIVE on
the population that could have been traded, while its label is learnable. That is explanation
(b), stated by the instrument rather than argued.

Four triggers DO have selection value, and `gbb` is not among them.

## Three qualifications, none of which rescue `gbb`

**1. The model's selection inside `gbb` is real but not the trigger's doing.** T027: the arm's
model-selected 0.470% of `gbb` bars realised +0.005204 against +0.000769 over all `gbb` bars,
6.8×. So the shipped strategy is not worthless — but the selection is attributable to what
sits downstream of the trigger, and a plain tighter threshold does the same job 5.6× better
(`gbb_tight`, +0.029014 at matched selectivity). `gbb`'s verdict is about the trigger AS
SHIPPED, at its shipped thresholds.

**2. `gbb_tight`'s learnability figure is UNRESOLVABLE and must not be read.** Its conditional
probe ran on test sets of **14 to 230 samples** (against 158,976–191,136 for `empty`), with
MCC swinging −0.113 to +0.390 across seeds. An MCC on 14 samples is not a measurement. Its
verdict is `selection_value` on the ECONOMIC-VALUE branch, which is independent of
learnability, so the unresolvable figure does not affect the adjudication — but that is luck,
not design, and the number is reported struck-through in spirit.

**3. `geometry` has the opposite profile to `gbb`, and it is coherent.** Significant positive
economic value (+0.002482) with significantly NEGATIVE conditional learnability (−0.0142):
it selects collectable moments, and within them the outcome is LESS predictable than at
large. A genuinely selective trigger that is not model-friendly — which under the old
unconditional column would have looked like a strong `decoration` candidate (its
unconditional MCC was +0.3286, nine standard errors above the null).

## The spec asymmetry, disclosed

The economic-value bar is a 95% interval; the learnability bar is ONE standard error
(contracts/trigger-cli.md's own words, "by more than the SE on that difference"). One SE is
roughly a 68% bar, so `decoration` is EASIER to earn than `selection_value`. Implemented as
written rather than silently tightened, pinned by a test, and reported per row so a reader
can see which bar each row cleared. Note it does not change any verdict here: every
`selection_value` row clears the stricter EV bar outright, and `gbb`'s `decoration` would
stand at 1.96 SE too (+0.0175 against 0.0214 — it would NOT; at 1.96 SE `gbb` becomes
`indeterminate`). That single row IS bar-sensitive, and it is the headline one, so it is
flagged rather than buried.

## What this does NOT license (FR-017, FR-018)

Nothing here is a P&L claim. Table economic value is a lookahead bound; the only instrument
that can promote or demote anything is the pooled harness in `evaluation/`, at ~3–4pp/yr.
`gbb` returning +42.71pp there and `decoration` here is not a contradiction — the harness
measures a strategy (trigger + model + exit + guards), this measures a trigger alone, and
T027 located the difference precisely. The actionable reading is narrow and specific: **the
value attributed to `gbb` lives in its threshold margin and in what selects within it, not
in the binary condition the label encodes.**

---

# CORRECTION (2026-10-09) — the leaderboard covered 4 of 14 distinct factors, not all of them

T025's audit found 19 members reduce to a trigger, carrying **14 distinct** backward factors.
The first pass registered **4** and justified stopping there as "not one per labeler, which
would pad the leaderboard with duplicate rows". That justification covers only **5** members —
`future_sortino` and `future_expectancy` (share `adx`), `breakout_tb` (shares `breakout`),
`gbb_path` and `gbb_fwdmae` (share `gbb`). The remaining **10 were genuinely different
conditions and were simply missing**. Registered now in `research/outcomes/audit_triggers.py`,
each verified by containment against its labeler on real data (SOL/USDT, 8 of 10 scoreable,
zero violations).

**Two of the ten are UNSCOREABLE, which is a finding in itself.** `trends` needs
`flow`/`regime`/`risk`/`momentum`; `bands` needs eight band columns. The populator the outcome
table is built from supplies NONE of the first four and only two of the eight. Both now declare
their real requirements so the registry REFUSES them, rather than scoring a degraded version
under the labeler's name — a third category FR-012 did not anticipate: *reduces to a genuine
backward factor that is not computable on the available feature set*. `trends` is the dangerous
one: its NaN fill of 0.0 reads as DECREASE/BEAR/NEGATIVE and VOTES YES, so a silent evaluation
would have fired on nearly every bar and looked like a wildly permissive trigger rather than a
missing input. My first draft of `bands` silently skipped its six absent columns and would have
scored a Bollinger-only subset under the labeler's name — caught before it reached the board.

## 15 scoreable triggers, economic value vs the `empty` null (FILLABLE, paired block bootstrap)

| trigger | base rate | n non-ovl | EV delta | 95% CI | beats null |
|---|---|---|---|---|---|
| `gbb_tight` | 0.070% | 206 | **+0.017120** | [+0.003899, +0.029222] | **yes** |
| `breakout_vol` | 2.862% | 20,739 | **+0.010527** | [+0.009846, +0.011209] | **yes** |
| `breakout` | 6.453% | 27,102 | **+0.010348** | [+0.009705, +0.010988] | **yes** |
| `breakout_squeeze` | 2.269% | 15,510 | **+0.009069** | [+0.007945, +0.010131] | **yes** |
| `breakout_bb` | 6.076% | 19,147 | **+0.009059** | [+0.008340, +0.009784] | **yes** |
| `breakout_gbb` | 12.491% | 21,145 | **+0.004643** | [+0.004030, +0.005311] | **yes** |
| `geometry` | 2.868% | 24,709 | **+0.002482** | [+0.001377, +0.003591] | **yes** |
| `adx` | 61.233% | 38,873 | +0.000146 | [−0.000173, +0.000464] | no |
| `votes_consensus` | 23.561% | 32,733 | −0.001324 | [−0.001841, −0.000814] | no (neg) |
| `gbb` | 17.871% | 22,420 | **−0.004102** | [−0.004793, −0.003402] | no (neg) |
| `votes_ind3` | 33.856% | 30,784 | −0.004418 | [−0.005014, −0.003867] | no (neg) |
| `votes_ind4` | 51.434% | 34,775 | −0.004581 | [−0.005067, −0.004112] | no (neg) |
| `votes_tech` | 32.400% | 29,756 | −0.005192 | [−0.005748, −0.004646] | no (neg) |
| `votes_ind2` | 19.907% | 27,068 | −0.005491 | [−0.006007, −0.004988] | no (neg) |

## The finding the 7-row board could not show

The result is **family-level, and the families split cleanly**:

- **Every breakout-family trigger has selection value.** All five, +0.0046 to +0.0105, none
  spanning zero. These are momentum conditions: price exceeding a prior high, optionally
  confirmed by volume, by a volatility squeeze, or by band position.
- **Every vote-family trigger is significantly WORSE than trading every bar.** All five,
  −0.0013 to −0.0055. These are indicator-consensus conditions — "at least N of these
  oversold/overbought indicators agree" — over four different indicator sets plus a flipped
  variant, and they select moments that are *worse than random*.
- **`gbb` sits with the VOTE family, not the breakout family** (−0.004102, squarely inside
  their −0.0013..−0.0055 band). Structurally that is exactly what it is: a threshold
  conjunction on two indicator columns. Its `decoration` verdict is not an isolated oddity
  about one trigger; it is the behaviour of its whole class.

That is a far stronger statement than the 7-row board supported, and it only became visible
once the audit's full set was registered. It also reframes the `gbb_tight` result: tightening
an indicator-threshold trigger moves it from the negative family to the top of the board, which
is the margin finding restated across families.

Verdicts for the seven EV-positive rows are `selection_value` already, since FR-014b's first
branch is independent of learnability. The eight remaining rows need conditional learnability to
separate `decoration` from `indeterminate`; that campaign is running (72 fits).

---

# THE FINAL BOARD (2026-10-09) — 14 scoreable triggers adjudicated

72 conditional fits added (8 triggers × 3 windows × 3 seeds), 0 failures. FR-014b as amended
by FR-015b; economic value on the fillable population, paired block bootstrap vs the `empty`
null.

| trigger | verdict | EV delta | 95% CI | cond MCC Δ | SE |
|---|---|---|---|---|---|
| `gbb_tight` | **selection_value** | +0.017120 | [+0.003899, +0.029222] | +0.1148 | 0.0549 |
| `breakout_vol` | **selection_value** | +0.010527 | [+0.009846, +0.011209] | +0.0289 | 0.0079 |
| `breakout` | **selection_value** | +0.010348 | [+0.009705, +0.010988] | +0.0204 | 0.0093 |
| `breakout_squeeze` | **selection_value** | +0.009069 | [+0.007945, +0.010131] | +0.0080 | 0.0114 |
| `breakout_bb` | **selection_value** | +0.009059 | [+0.008340, +0.009784] | +0.0444 | 0.0075 |
| `breakout_gbb` | **selection_value** | +0.004643 | [+0.004030, +0.005311] | +0.0186 | 0.0064 |
| `geometry` | **selection_value** | +0.002482 | [+0.001377, +0.003591] | −0.0142 | 0.0074 |
| `votes_consensus` | **decoration** | −0.001324 | [−0.001841, −0.000814] | +0.0185 | 0.0080 |
| **`gbb`** | **decoration** | −0.004102 | [−0.004793, −0.003402] | +0.0175 | 0.0109 |
| `adx` | indeterminate | +0.000146 | [−0.000173, +0.000464] | +0.0007 | 0.0102 |
| `votes_ind3` | indeterminate | −0.004418 | [−0.005014, −0.003867] | +0.0063 | 0.0163 |
| `votes_ind4` | indeterminate | −0.004581 | [−0.005067, −0.004112] | −0.0044 | 0.0127 |
| `votes_tech` | indeterminate | −0.005192 | [−0.005748, −0.004646] | +0.0144 | 0.0150 |
| `votes_ind2` | indeterminate | −0.005491 | [−0.006007, −0.004988] | +0.0091 | 0.0158 |

**7 selection_value · 2 decoration · 5 indeterminate · 2 unscoreable** (`bands`, `trends`).

## Reading the board

**The vote family mostly lands `indeterminate`, not `decoration`** — and that is the rule
working rather than failing. Four of the five have significantly NEGATIVE economic value but
conditional learnability that does not clear even one standard error (their SEs are
0.0127–0.0163, two to three times the breakout family's). So the honest verdict is: they
select moments worse than random, and we cannot establish they are model-friendly either.
`indeterminate` "MUST be reported as such rather than resolved by judgement" (FR-014b), and
five rows exercise that branch.

**`geometry` is the cleanest single result on the board.** Positive economic value
(+0.002482, CI clear of zero) with significantly NEGATIVE conditional learnability
(−0.0142 ± 0.0074). It selects collectable moments within which the outcome is LESS
predictable than at large — genuinely selective, and actively model-hostile. Under the
pre-amendment unconditional column it scored +0.3286, nine SEs above the null, and would have
been read as a textbook `decoration` candidate. The amendment reversed that row completely.

**`breakout_bb` is the strongest agreement between the two columns**: +0.009059 EV and the
highest conditional learnability on the board (+0.0444 ± 0.0075). Both columns say the same
thing about it, which no other row manages.

**`gbb`'s `decoration` verdict survives, but on the 1-SE bar.** +0.0175 against 0.0109 clears
one SE; at 1.96 SE (0.0214) it would be `indeterminate`. The *direction* is unaffected — its
negative economic value is significant at 95% either way — but the specific verdict word is
bar-sensitive, and it is the headline row. Flagged here rather than left for a reader to
discover.

## What the whole exercise answered

The spec's question was whether a trigger earns its keep by **(a)** selecting collectable
moments or **(b)** handing the model an easy target. The board answers it per-trigger, and the
answer is structural rather than incidental:

- **Momentum conditions have selection value.** All five breakout-family triggers, plus the
  causal valley condition, plus a tightened indicator threshold.
- **Indicator-consensus conditions do not.** All five vote-family triggers have significantly
  negative economic value, and `gbb` — a threshold conjunction on two indicator columns —
  behaves like them, not like the breakout family.
- **The production incumbent is in the losing class**, and the value attributed to it lives in
  its threshold MARGIN and in whatever selects within it, not in the binary condition its
  label encodes.

Nothing here is a P&L claim (FR-017). The only instrument that can promote anything is the
pooled harness in `evaluation/`, and this repository has twice measured a rising table ceiling
failing to become harness return.

---

# T033 — SC-005 measured. The premise holds.

SC-005 claims evaluating one trigger takes **seconds**, against the ~1.5 hours a trained
comparison costs, and that claim is the entire justification for building the outcome table.
No task had measured it until now.

**Amortised setup** (once per session, any number of triggers):

| step | time |
|---|---|
| load + populate 11 pairs, load cached tables | 10.50s |
| build window A features for the probe | 2.14s |

**Per-trigger evaluation** (`breakout`, the full two-column assessment):

| step | time |
|---|---|
| economic value, all bars | 3.37s |
| economic value, fillable | 3.32s |
| block-bootstrap interval, 4,000 resamples | 4.13s |
| conditional probe, one window+seed | 2.67s |
| **total, one probe fit** | **13.49s** |
| total, full 3×3 probe grid (extrapolated) | 34.83s |

**Verdict: SC-005 is MET.** 13.5s, or 34.8s with the complete probe grid, against ~1.5–3h
to train 15 folds (or ~2.5 min to re-evaluate with models already trained, measured during
this feature's own arm re-run). A 150–800× reduction depending on which baseline is used.

**The honest caveat, which SC-005's wording does not carry.** The two numbers are not
substitutes. 13.5s buys a LOOKAHEAD BOUND on a trigger; 1.5h buys a realisable return for a
STRATEGY. The design delivered what it promised — trigger evaluation became cheap enough to
do exhaustively, which is why 15 triggers could be scored at all and why the family-level
split became visible. It did not make harness validation cheaper, and nothing here replaces
it (FR-017).

---

# RETRACTION (2026-10-09) — the board paired each trigger with its OWN signal bar

T035's independent gate review **REJECTED** the board, and it was right. The defect was not
on the list of things I asked it to check.

## The defect

A trigger is evaluated on bar `i` and reads bar `i`'s **close** — `breakout` is literally
`close[i] > prior_high`. The outcome table's row `i` simulates entering at **`open[i]`**
(`ExitPolicy.py:733`, since the 2026-10-08 entry-convention fix). The leaderboard paired them
directly, so every row booked **its own signal bar's open-to-close move** as profit the
trigger could never have captured: at `open[i]` the close that produced the signal has not
happened yet. Momentum triggers fire on up-bars, so they collected it.

`labels.assert_aligned` enforced same-date pairing — i.e. it enforced the **wrong** pairing,
rigorously.

**Why T021's reconciliation did not catch it.** That work matched the ARM's trades at
`trade.open_date` and looked up the row at that date. The arm genuinely enters at that bar's
open, so row `i` was validated correctly *as a trade starting at bar i*. The arm's SIGNAL was
the bar before. The table was never wrong; my pairing was.

Verified independently before accepting the review (fillable, non-overlapping, pooled 11
pairs):

| trigger | sig-bar o→c | paired at `i` | paired at `i+1` | share that was leak |
|---|---|---|---|---|
| `breakout` | +0.456% | +0.009076 | +0.000807 | **91%** |
| `breakout_vol` | +0.614% | +0.009254 | +0.000582 | **94%** |
| `geometry` | +0.086% | +0.001210 | −0.001526 | **sign flip** |
| `gbb` | −0.047% | −0.005374 | −0.002009 | 63% |
| `votes_tech` | −0.067% | −0.006464 | −0.003605 | 44% |
| `empty` | −0.000% | −0.001272 | −0.001275 | **0 — the control** |

`empty` is unchanged to 2e-6 because it fires on every bar. That control is what proves the
shift itself introduces nothing and the effect it removed was real.

## The corrected board (economic value only; the probe still needs rebuilding)

Fixed in ONE place — `labels.entry_mask` — which `trigger_label_mask`, `_select` and
`_delta_groups` all now route through. Paired block bootstrap vs the `empty` null, fillable:

| trigger | EV delta | 95% CI | beats null | was |
|---|---|---|---|---|
| `gbb_tight` | +0.031491 | [+0.019277, +0.044113] | yes | +0.017120 |
| `breakout_bb` | +0.002818 | [+0.002032, +0.003650] | yes | +0.009059 |
| `breakout` | +0.002079 | [+0.001361, +0.002802] | yes | +0.010348 |
| `breakout_gbb` | +0.001964 | [+0.001284, +0.002675] | yes | +0.004643 |
| `breakout_vol` | +0.001854 | [+0.000930, +0.002783] | yes | +0.010527 |
| `breakout_squeeze` | +0.000758 | [−0.000566, +0.002049] | **no (was yes)** | +0.009069 |
| `adx` | +0.000123 | [−0.000222, +0.000461] | no | +0.000146 |
| `votes_consensus` | +0.000089 | [−0.000453, +0.000633] | **no longer negative** | −0.001324 |
| `geometry` | −0.000254 | [−0.001425, +0.000922] | **no (was yes)** | +0.002482 |
| `gbb` | −0.000737 | [−0.001468, **+0.000004**] | no — but now barely spans zero | −0.004102 |
| `votes_ind4` | −0.001859 | [−0.002319, −0.001405] | no (neg) | −0.004581 |
| `votes_ind3` | −0.002179 | [−0.002779, −0.001600] | no (neg) | −0.004418 |
| `votes_ind2` | −0.002332 | [−0.002971, −0.001739] | no (neg) | −0.005491 |
| `votes_tech` | −0.002334 | [−0.002910, −0.001763] | no (neg) | −0.005192 |

Reproduces the review's shifted figures to 6 decimals from an independently written script —
mutual validation, not agreement by construction.

## What survives, and what I retract

**RETRACTED**: the magnitudes. Every effect shrinks ~5×. `geometry`'s and
`breakout_squeeze`'s `selection_value` verdicts are withdrawn. `votes_consensus` is no longer
significantly negative. **`gbb` is no longer significantly negative** — its interval now
reaches +0.000004, so "the production incumbent has significantly negative economic value"
must be downgraded to "indistinguishable from the null, point estimate slightly negative".

**SURVIVES, weakened**: the family direction. Four of five breakout-family triggers still beat
the null (+0.0019 to +0.0028); four of five vote-family triggers are still significantly
negative (−0.0019 to −0.0023). So "momentum selects, indicator-consensus does not" holds in
sign, at roughly a fifth of the claimed size, and no longer cleanly — one row crosses from
each family.

**UNCHANGED**: `gbb` has no selection value. That was the headline conclusion and it does not
depend on the magnitude — at +0.0019 or −0.0007 it is not a trigger that selects collectable
moments. The gate review's own proposed wording is better than mine was: *"`gbb` has no
selection value; its conditional learnability is weakly positive in all three windows."*

## Still outstanding from the review

1. **The conditional probe has the same leak** and must be re-run: my campaign scripts built
   labels from the raw trigger mask, not `entry_mask`. 126 fits, ~1h. Until then no verdict is
   re-adjudicated.
2. `gbb_tight`: review says **WITHDRAW** from the adjudicated table — fillable non-overlapping
   n = 98, and one week (2022-05-09, the LUNA collapse) supplies 69% of summed profit, top 3
   weeks 101%, mean ex-top-3-weeks ≈ −0.0003. That is concentration I did not test for and it
   is decisive. Keep as a diagnostic annotation, not a row.
3. The `empty`-vs-itself control is **vacuous**: identical masks give identical block sums, so
   [0,0] is guaranteed and cannot detect a mis-wired reference. Needs a random trigger at
   matched base rate, and a shifted trigger at k ≫ hold, as real controls.
4. Pairs are block-resampled independently although same-time crypto bars are highly
   correlated; blocks should be CALENDAR blocks spanning all pairs.
5. `L=48` is shorter than many holds (`max_hold` 10d = 960 bars).
6. `breakout_gbb` and `breakout_bb` contain **no breakout condition** despite their names, and
   T025's containment check only catches a too-TIGHT factor, never a too-loose one — so
   "bit-identical port" is not established for those two.
7. No committed script produces `leaderboard.txt`; `quickstart.md` cites a `--build` CLI that
   does not exist.

---

# RE-ADJUDICATED BOARD (2026-10-09) — corrected pairing, 126-fit probe rebuild, real controls

Economic value and conditional learnability both rebuilt on `labels.entry_mask` (trigger on
bar `i` → outcome of entering at `open[i+1]`). 126 conditional fits, 0 failures. Controls
included and adjudicated. `gbb_tight` withdrawn to `(diagnostic)`.

| trigger | verdict | EV fill | 95% CI vs null | cond MCC | SE |
|---|---|---|---|---|---|
| `breakout_gbb` | **selection_value** | +0.000692 | [+0.001284, +0.002675] | +0.0468 | 0.0080 |
| `breakout_bb` | **selection_value** | +0.001546 | [+0.002032, +0.003650] | +0.0526 | 0.0074 |
| `breakout_vol` | **selection_value** | +0.000582 | [+0.000930, +0.002783] | +0.0400 | 0.0086 |
| `breakout` | **selection_value** | +0.000807 | [+0.001361, +0.002802] | +0.0384 | 0.0069 |
| `votes_consensus` | **decoration** | −0.001183 | [−0.000453, +0.000633] | +0.0516 | 0.0081 |
| **`gbb`** | **indeterminate** | −0.002009 | [−0.001468, **+0.000004**] | +0.0489 | 0.0143 |
| `votes_tech` | indeterminate | −0.003605 | [−0.002910, −0.001763] | +0.0445 | 0.0155 |
| `votes_ind3` | indeterminate | −0.003453 | [−0.002779, −0.001600] | +0.0405 | 0.0172 |
| `votes_ind2` | indeterminate | −0.003604 | [−0.002971, −0.001739] | +0.0383 | 0.0174 |
| `votes_ind4` | indeterminate | −0.003131 | [−0.002319, −0.001405] | +0.0342 | 0.0146 |
| `gbb_stale` **(CONTROL)** | indeterminate | −0.001406 | [−0.000949, +0.000715] | n/a | |
| `adx` | indeterminate | −0.001149 | [−0.000222, +0.000461] | +0.0380 | 0.0112 |
| `random_matched` **(CONTROL)** | indeterminate | −0.000947 | [−0.000275, +0.000895] | n/a | |
| `geometry` | indeterminate | −0.001526 | [−0.001425, +0.000922] | +0.0370 | 0.0073 |
| `breakout_squeeze` | indeterminate | −0.000514 | [−0.000566, +0.002049] | +0.0294 | 0.0094 |
| `empty` | (null) | −0.001272 | — | +0.0350 | 0.0092 |
| `gbb_tight` | (diagnostic) | +0.030219 | [+0.019277, +0.044113] | withdrawn | |

**4 selection_value · 1 decoration · 10 indeterminate · 1 diagnostic · 1 null · 2 unscoreable.**

## The further retraction this forces

**`gbb` is `indeterminate`, not `decoration`.** Both halves of that verdict dissolved on the
corrected pairing: its economic-value interval now reaches +0.000004 (spans zero), and its
conditional learnability advantage (+0.0139) no longer clears even one standard error
(0.0170). The earlier claim — *"the production incumbent is adjudicated decoration"* — is
**withdrawn**. The defensible statement is narrower:

> `gbb` has **no measurable selection value**. Its economic value is indistinguishable from
> the `empty` null AND from `random_matched`, a market-independent trigger at its own base
> rate. Whether its label is more learnable than the null's is not resolvable at this
> sample size.

That is weaker than what I published, and it is what the measurements support.

## What survives, cleanly

**Only the breakout family has selection value, and 4 of its 5 members do.** Nothing else on
the board clears the economic-value bar — not `gbb`, not any vote-family trigger, not
`geometry`, not `adx`. The "momentum selects, indicator-consensus does not" finding survives
the correction in a *sharper* form than before: the positive side is now exclusively one
family, rather than a family plus two strays.

**The vote family's economic value is significantly NEGATIVE** (four of five, CIs wholly below
zero) but their conditional learnability does not clear one SE, so they land `indeterminate`
rather than `decoration`. They select moments worse than random, and we cannot establish they
are model-friendly either.

**The controls behave.** `random_matched` and `gbb_stale` both come out `indeterminate` with
intervals spanning zero, and `gbb` sits inside their band. That is the validation the
`empty`-vs-itself control could never provide.

## A caveat on the breakout family's rows

Their `EV` (all bars) is NEGATIVE (−0.0015 to −0.0016) while their `EV fill` is positive. Their
selection value therefore rests entirely on the FILLABLE population. That is the right
population — FR-009b exists because it is the one that could actually have been traded, and
`empty` moves the other way (−0.000975 all bars vs −0.001272 fillable) — so this strengthens
rather than weakens the rows. But it must be stated: on the unfiltered population no trigger
on this board beats the null.

## Still outstanding from T035

Calendar blocks spanning all pairs (pairs are currently resampled independently though
same-time crypto bars are correlated); `L=48` is shorter than `max_hold`'s 960 bars; and
`breakout_gbb`/`breakout_bb` contain no breakout condition despite their names, with
containment only able to catch a too-TIGHT factor. These are instrument-quality items, and
the first two would widen the breakout family's intervals — the direction that could cost
those four verdicts.

---

# T035's remaining three items, fixed (2026-10-09)

## 1+2. Calendar blocks spanning all pairs, and L up to the full hold

**The defect.** Blocks were formed WITHIN each pair, so a resample could draw pair A's
May-2022 block alongside pair B's March-2024 block — implicitly treating the 11-pair
cross-section as independent when same-time crypto bars are highly correlated. And `L=48`
was chosen against the MEAN hold (32–42 bars), which is the wrong statistic: a block must
contain the LONGEST dependence it absorbs, and `max_hold_days` is 10 days = **960 bars**.

**The fix.** `paired_calendar_block_delta` keys blocks on calendar time, so one draw moves
every pair's bars in that window together. Sensitivity reported rather than one value
asserted:

| trigger | per-pair L=48 | cal L=48 | cal L=192 | cal L=960 |
|---|---|---|---|---|
| `breakout` | +0.00208 **[Y]** w=0.00144 | **[Y]** w=0.00174 | **[Y]** w=0.00171 | **[Y]** w=0.00151 |
| `breakout_bb` | +0.00282 **[Y]** w=0.00162 | **[Y]** w=0.00210 | **[Y]** w=0.00215 | **[Y]** w=0.00204 |
| `breakout_vol` | +0.00185 **[Y]** w=0.00185 | **[Y]** w=0.00226 | **[Y]** w=0.00223 | **[Y]** w=0.00207 |
| `breakout_gbb` | +0.00196 **[Y]** w=0.00139 | **[Y]** w=0.00190 | **[Y]** w=0.00181 | **[Y]** w=0.00174 |
| `gbb` | −0.00074 [n] w=0.00147 | [n] w=0.00195 | [n] w=0.00186 | [n] w=0.00153 |
| `random_matched` | +0.00033 [n] w=0.00117 | [n] w=0.00126 | [n] w=0.00120 | [n] w=0.00110 |
| `gbb_stale` | −0.00013 [n] w=0.00166 | [n] w=0.00210 | [n] w=0.00218 | [n] w=0.00249 |
| `votes_tech` | −0.00233 [n] w=0.00115 | [n] w=0.00148 | [n] w=0.00139 | [n] w=0.00133 |

Calendar blocks widen every interval by **10–37%**, which is the point — they stop crediting
the cross-section with independence it does not have. **No verdict changes, at any block
length, including L=960.** The four breakout-family `selection_value` verdicts survive the
more conservative treatment, which is the outcome that was genuinely in doubt.

Width does not grow monotonically in L (breakout: 0.00174 → 0.00171 → 0.00151), because two
effects offset — longer blocks capture more dependence but also improve the paired
cancellation (T028's finding). `gbb_stale` is the exception, widening throughout
(0.00210 → 0.00249), exactly as a trigger with intact clustering and no information should.

## 3. Port exactness — equality, not containment

**The defect.** Containment (`label ⊆ trigger`) catches a port that is too TIGHT but never
one that is too LOOSE: a trigger firing on bars the factor would reject passes containment
while scoring a broader condition under the labeler's name. The review raised this for
`breakout_gbb` and `breakout_bb`, which contain no breakout condition despite their names.

**The fix.** 11 EQUALITY tests, each against a factor reconstructed from the live source
independently of the implementation under test, asserting zero bars in BOTH directions. All
pass — every port is exact. Plus a test proving the check can detect a too-loose port (a
deliberately broadened `breakout_gbb` at `guard > 0.0` instead of `0.2`), so the new check
is not vacuous in the direction it was added for.

**On the names**: `breakout_gbb` and `breakout_bb` genuinely contain no breakout condition.
The names come from the labelers and are kept so the audit maps 1:1, with the discrepancy
stated in each trigger's registry description. The implementations are exact; only the names
mislead.

---

# HARNESS TEST (2026-10-09) — `breakout_vol` traded. It LOSES, and the gap decomposes cleanly.

`SimpleStrategies/BreakoutVolOpen.py`: the `breakout_vol` trigger, no model, Open exit,
guards open. Entry is the REGISTERED trigger by import (`triggers.evaluate("breakout_vol")`),
not a reimplementation — and it refuses to trade if that import fails, rather than falling
back to a copy that could drift from the row under test. Same 15 fold test-ranges as the
control.

| arm | pooled | per year | trades | folds positive |
|---|---|---|---|---|
| **BreakoutVolOpen** (trigger only) | **−63.19pp** | **−17.17pp/yr** | 8,290 | **6/15** |
| control (gbb + model) | +42.71pp | +11.61pp/yr | 1,420 | 14/15 |

Harness verdict `INCONCLUSIVE_ARM_LOSES`: it loses to cash, so a delta against the control
would measure the control rather than the arm.

**This is the third time this repository has measured "the table ceiling rises, the return
does not"** — after trigger relaxation and feature 003's tail labelling. It was the outcome
I called most likely, and it is a real result rather than a disappointment.

## Where the +0.000582 became −0.001512, per trade

| stage | EV per trade | step |
|---|---|---|
| leaderboard figure (fillable, NON-OVERLAPPING) | +0.000582 | — |
| table EV at the bars the arm ACTUALLY entered | **−0.000547** | **−0.001129** |
| live realised | **−0.001512** | **−0.000965** |

Two distinct losses, and the first is the important one:

**1. The population the leaderboard scores is not the population a strategy enters
(−0.001129).** The board's figure is computed over *fillable* bars, *greedily de-overlapped*.
A live strategy cannot select that subset: it takes signals as they arrive, subject to
`max_open_trades`, including unfillable ones and clustered ones. On the 8,290 bars this arm
actually entered, **the table's own outcome is NEGATIVE**. So the row was never a claim about
a tradeable population — and nothing in the leaderboard said which population it described.

**2. Live execution costs a further −0.000965** on those same bars: concurrency (553
trades/fold against the control's 95, into 9 slots), fills, and fees on real stake amounts
rather than a notional unit.

The live arm WINS MORE OFTEN than the table says (72.99% vs 57.35%) while losing more money —
more frequent small wins, fatter left tail (live p05 −0.0519). The same asymmetry the table's
own mean/median split showed, amplified.

## What this changes

**The `selection_value` verdict stands as defined, and is now known not to transfer.** The
four breakout-family rows are correctly adjudicated under FR-014b — their table economic
value does exceed the null on the fillable non-overlapping population, robustly, across every
block length. That measurement was not wrong. It simply does not license a strategy, and the
harness is the only thing that could have revealed by how much.

**The "implement the triggers and guards, drop the model" hypothesis is REFUTED for
`breakout_vol`** — the strongest-interval row on the board, deliberately chosen as the best
case. −17.17pp/yr against the model-based control's +11.61pp/yr is not a near miss.

**It also retroactively justifies `gbb`.** `gbb` scores at the null on the table and
indistinguishably from a random trigger at its base rate, yet `gbb` + model returns
+42.71pp in 14/15 folds. Whatever the model-and-gate stack is doing with `gbb`'s population,
the table cannot see it — and the table's ranking is, on this evidence, close to uninformative
about which trigger to build on.

**The honest statement of the leaderboard's scope**, which belongs in the report and did not:

> Economic value ranks triggers by the outcome of a hypothetical unit trade at every
> fillable, non-overlapping triggered bar. It does not rank them by what a capital-
> constrained strategy taking their signals would earn, and the two can differ in SIGN.
