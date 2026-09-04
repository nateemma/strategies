# Should the breakout label target the TAIL rather than the average?

Conjecture (2026-09-04): the breakout label is tuned for "average" breakouts, the
payoff distribution is not Gaussian, and mean/median are therefore poor guides —
the label should be set to catch the BIG moves.

**Confirmed, with one blocker.** Scripts: `breakout_dist.py`, `breakout_tail.py`,
`tail_path.py`. No training required.

## 1. The distribution is emphatically fat-tailed

Forward MFE from breakout-condition bars (`guard_metric > 0.2 & bb_width > 0.035`),
pooled XRP/SOL/LINK, 15m:

| H | 6 | 12 | 24 | 48 | 96 |
|---|---|---|---|---|---|
| median % | 0.52 | 0.85 | 1.33 | 2.11 | 3.27 |
| p90 % | 2.43 | 3.62 | 5.25 | 7.87 | 11.54 |
| p99 % | 6.50 | 9.40 | 12.82 | 17.91 | 26.55 |
| **skew** | 3.78 | 3.44 | 7.85 | 5.28 | 4.30 |
| **excess kurtosis** | 64 | 35 | 225 | 81 | 43 |
| **% of gain in top 10%** | 43.5 | 41.4 | 39.4 | 38.0 | 37.3 |
| % of gain in top 25% | 70.8 | 68.2 | 65.9 | 64.2 | 63.1 |

Gaussian reference: skew 0, excess kurtosis 0. **The top 10% of breakout bars
carry ~40% of all upside; the top 25% carry ~65%.** Mean and median describe
almost none of the value.

## 2. The current threshold does not select for big moves at all

At H=48 the median forward MFE is **2.11%** while `min_gain` is **0.5%** — the
threshold accepts nearly everything. Worse, measured against the operative ATR
stop (−4% to −6%):

| setting | % bars | median gain | clears a 5% stop | reward:risk |
|---|---|---|---|---|
| **H=48, 0.005 (current)** | 9.07% | 2.83% | **26.0%** | **0.57** |
| H=48, 0.02 | 5.81% | 4.29% | 40.6% | 0.86 |
| H=48, 0.03 | 4.30% ⚠ | 5.34% | 54.9% | 1.07 |
| **H=96, 0.03** | **5.92%** | **6.41%** | **66.8%** | **1.28** |
| H=96, 0.05 | 3.95% ⚠ | 8.43% | 100% | 1.69 |

⚠ = below the ~5.5%-of-bars floor where `aug_risk` flags all-Hold collapse.

**Only 26% of currently-labelled bars ever clear the stop.** The label teaches the
model to find moves that mostly cannot pay for the risk of holding them. That is a
more fundamental defect than anything tested in the breakout study.

**H=96 / thr 0.03** is the best cell that stays above the density floor: median
gain 2.83% → 6.41%, reward:risk 0.57 → 1.28, stop-clearing 26% → 67%.

## 3. The blocker: time-to-peak, not path risk

Path-aware check (walk to the MFE bar, measure worst drawdown on the way):

| setting | median DD | p10 DD | stopped at −4% | median time-to-peak |
|---|---|---|---|---|
| H=48, 0.005 (current) | −0.62% | −2.77% | 4.5% | 6.8h |
| **H=96, 0.03** | −1.06% | −4.24% | **11.2%** | **17.0h** |

Path risk rises 2.5× but leaves 89% of labels reachable, so the improved
economics survive it.

**Time-to-peak is the problem.** The arm exits at a 4-hour median, and
`unclog_12h` force-exits at 12h whenever profit > 0.5% (hardcoded in
`BaseStrategy`). A label whose payoff peaks at 17h would be systematically
harvested early — trained on one horizon, traded on another.

## Verdict

The conjecture is right and the current label is mis-specified, but this **cannot
be adopted as a label change alone**. It is a package:

1. `MIN_BUY_GAIN_THRESHOLD` 0.005 → 0.03 and `HORIZON` 48 → 96;
2. extend the hold window to match — the `unclog_12h` threshold and
   `cexit_max_days` — or the exit machinery forbids what the label promises.

Changing (1) without (2) would train on 17-hour moves and cap them at 12 hours.

**Caveat before spending a retrain.** This is a model-side change, and the
category has produced nothing all day (five large metric shifts, no P&L). But it
differs from those in kind: it changes the label's ECONOMICS (reward:risk 0.57 →
1.28) rather than its density or the model's fit to it, and the breakout arm's
gate is loose enough that entries are not fully guard-determined. It is the
best-motivated untested idea remaining on this line.

---

# Same analysis on gbb (method 17) — the PRODUCTION label

Asked 2026-09-04: does gbb have the same issues, especially around the unclog
threshold? Script: `regime/gbb_dist.py`.

## The fat tail is generic, not a breakout quirk

`guard_metric < -0.2 & bb_width > 0.035`, forward MFE, pooled XRP/SOL/LINK:

| H | median | p90 | p99 | skew | excess kurt | top-10% share |
|---|---|---|---|---|---|---|
| 6 | 0.64 | 2.39 | 6.46 | 3.55 | 29.8 | 40.6% |
| 12 | 0.99 | 3.51 | 9.31 | 3.76 | 30.1 | 39.0% |
| 24 | 1.49 | 5.09 | 13.97 | 3.91 | 32.8 | 38.7% |
| 48 | 2.13 | 7.41 | 18.45 | 3.83 | 30.1 | 37.8% |
| 96 | 3.10 | 10.88 | 26.81 | 25.09 | 1001 ⚠ | 38.1% |

⚠ H=96 skew/kurtosis are outlier-driven — treat with caution.

Same shape as breakout: skew ~3.8, excess kurtosis ~30, top 10% of bars carrying
~38% of all upside.

## Production is NOT mismatched to the unclog window

| H | thr | % bars | median | clears TP 2.4% | clears 5% | median peak | stopped @−4% |
|---|---|---|---|---|---|---|---|
| 24 | 0.007 | 7.89% | 2.18% | 44.6% | 14.6% | 4.0h | 2.6% |
| **48** | **0.007 (prod)** | **8.75%** | **2.81%** | **57.8%** | 25.3% | **7.2h** | 5.5% |
| 48 | 0.010 | 8.01% | 3.07% | 63.1% | 27.6% | 7.8h | 5.7% |
| **48** | **0.020** | **5.83%** | **4.05%** | **86.7%** | 37.9% | **8.5h** | 6.4% |
| 96 | 0.007 | 9.49% | 3.79% | 69.4% | 38.5% | 14.0h ⚠ | 11.4% |
| 96 | 0.020 | 7.24% | 5.05% | 91.0% | 50.5% | 16.2h ⚠ | 12.3% |

**Median time-to-peak at production settings is 7.2h against a 12h unclog** —
comfortably inside. The mismatch that blocks the breakout candidate does not
apply here. Note H=96 WOULD introduce one (14–16h peaks), so any future horizon
increase for gbb carries the same caveat.

The label is also well matched to the exit that harvests it: **57.8% of labelled
bars clear the 2.4% take-profit**, which fires on 54 of 94 NNNC exits.

## The key observation: the label was never the differentiator

**gbb and breakout_gbb have essentially the SAME median forward gain** — 2.81% vs
2.83% at H=48, with the same tail shape. So the "label targets average moves"
critique applies equally to both, yet gbb makes +24% and breakout loses.

The difference is the GATE and the resulting hit rate: NNNC gbb runs 74.5% wins
against a 56.9% break-even (+17.5pp margin) on a tight seven-guard chain, while
the breakout arm manages 58% against 57% (+1pp) on a looser one. Same
distribution, same average move, opposite outcome — the day's gate-versus-model
theme once more.

Note the payoff SHAPE differs from what one might assume: NNNC's realised
reward:risk is only 0.76 (avgWin 3.73% vs avgLoss 4.93%). It is a high-hit-rate,
modest-payoff strategy. A mean-reversion label does not need big moves; it needs
to be right often. Tail-targeting is therefore much less obviously right for gbb
than for breakout.

## One untested opportunity for production

**`MIN_BUY_GAIN_THRESHOLD` 0.007 → 0.02 at H=48** lifts take-profit-clearing from
57.8% to **86.7%**, keeps density at 5.83% (just above the ~5.5% collapse floor),
and holds time-to-peak at 8.5h — still inside the unclog window. Unlike the
breakout package it requires NO exit changes.

Tempering: NNNC's gate is tight and absorbs model-side change — the peak-filter
fix added 77% more labels and produced byte-identical trades on two of three
windows. A threshold change alters label QUALITY rather than density, so it is
not the same intervention, but the prior is that the gate will absorb much of it.
One retrain to find out.

## thr 0.02 TESTED on NNNC — no out-of-sample effect

The one untested opportunity from the gbb analysis above: raise
`MIN_BUY_GAIN_THRESHOLD` 0.007 → 0.02 at H=48, predicted to lift
take-profit-clearing 57.8% → 86.7% while staying above the density floor and
inside the unclog window.

Trained on `20220101-20250831` (W1 out-of-sample) with its own
`NNNC_MLX_thr02.json` copied from production, so `exit_config_hash` matches and
the class-defaults confound is excluded.

**The label behaved exactly as predicted:** 6423 buy / 5829 sell labels (vs
10406 / 9529 at 0.007) — a 38% reduction, no all-Hold collapse, filter a no-op.

| arm | trades | W1 (OOS) | W2 | W3 | summed | PF | win% | worst DD |
|---|---|---|---|---|---|---|---|---|
| thr 0.007 (production) | 332 | **10.45** | 10.31 | **3.37** | 24.13 | 1.94 | 74.4 | 5.25 |
| thr 0.02 | 332 | **10.45** | **11.35** | **3.37** | 25.16 | 1.97 | 74.8 | 4.79 |

**W1 and W3 have IDENTICAL trade hashes.** The entire +1.03pp is W2, which is
in-sample. W1 payoff is unchanged to the decimal — win 74.7% vs 74.4%, avgWin
3.69% vs 3.73%, break-even margin +17.5pp in both.

**Not adopted.** There is no out-of-sample evidence for it.

Note the metric direction: `val_mcc` FELL 0.7590 → 0.6078 and precision 0.6422 →
0.4496 (a 2% gain requirement is a harder, sparser target), while summed P&L
nominally rose. Another decoupling, this time with the metric moving opposite to
the return — and neither reaching the out-of-sample trades.

This is the sixth model-side change in this study to leave OOS trading untouched,
across two families. See [[feedback_gate_and_mechanism_move_pnl_not_model]].
