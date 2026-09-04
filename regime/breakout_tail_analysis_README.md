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
