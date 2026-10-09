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
