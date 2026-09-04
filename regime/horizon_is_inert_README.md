# `HORIZON` is inert above `PEAK_WINDOW` — the effective label horizon is 6 bars

Found 2026-09-03 while testing whether H=48 was too long for a breakout label.
The test was doomed by construction, which is how the defect surfaced: a variant
with `HORIZON = 24` trained to a **byte-identical model** (same SHA-256) as the
H=48 parent.

## Mechanism

`BaseNNStrategy.get_train_buy_signals` runs the labeler at `HORIZON`, then passes
every surviving peak through `filter_peaks_by_future_performance`, which
re-applies **the same gain test** over `self.PEAK_WINDOW` — and
`BaseStrategy.PEAK_WINDOW = 6`.

For any `HORIZON >= 6` the 6-bar test is strictly tighter (max over a shorter
window ≤ max over a longer one), so the filtered set is a SUBSET of the labeler's
and the intersection equals the filter alone. `HORIZON` cancels out.

| HORIZON | after labeler | after filter | effect |
|---|---|---|---|
| 3 | 11205 | 11205 | BINDS |
| 4 | 12706 | 12706 | BINDS |
| 6 | 14797 | 14797 | BINDS |
| 8 | 16172 | **14797** | INERT |
| 16 | 19193 | **14797** | INERT |
| 24 | 20757 | **14797** | INERT |
| 48 | 22830 | **14797** | INERT |

**`HORIZON` only does anything below 6.** 11th recorded inert-parameter instance.

## What this invalidates

`TrainingConfig.HORIZON = 48` carries a long comment citing a joint
H × threshold viability sweep, and warns the pair "MUST be retuned together".
That sweep was run with `DebugSignalLearnability`, which calls the labeler
**directly, without the peak filter** — so it measured an effect the training
pipeline does not have. Every H ≥ 6 trains an identical model.

It also explains a recorded oddity: the 2026-05-28 univariate H sweep found H=3
beating H=8 on backtest P&L (+21.68% vs +15.97%). H=3 is the only value tested
that is BELOW `PEAK_WINDOW`, so it is the only one that changed anything. "H=3 is
best" really meant "H=3 is the only one that does anything different".

## The horizon question, inverted

The original question was whether a 12-hour horizon is too long for a breakout.
The answer is that the documented horizon is irrelevant and **the effective one
is 6 bars = 90 minutes** — which, against a median trade duration of 16 bars
(4 hours), is arguably too SHORT rather than too long.

So the label teaches the model to find moves that pay within 90 minutes, while
the strategy holds for four hours and stops out at 19 bars. That is a real
mismatch, in the opposite direction from the one suspected.

**The lever is `PEAK_WINDOW`, not `HORIZON`.** It is a `BaseStrategy` class
attribute, so it cannot be swept via `run_arm --set`; use a subclass. Note it is
also aliased as `lookahead_window` in `BaseNNStrategy`, so check what else
consumes it before changing it globally.
