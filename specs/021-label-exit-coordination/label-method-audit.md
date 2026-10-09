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
