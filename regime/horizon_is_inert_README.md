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

## FIXED 2026-09-03

The filter is leftover scaffolding for label schemes that do not self-filter.
Confirmed by inspection: **18 of 27 labelers apply their own forward-gain test at
`horizon`** — including every one in current use (gbb, breakout_gbb,
indicators2/3/4, bands, breakout_bb) — so for those the filter was pure
redundancy that silently overrode the horizon. The other **9 have no gain test**
(`triple_barrier`, `multi_horizon_vote`, `optimal_signals`, `future_sharpe`,
`future_sortino`, `future_expectancy`, `local_extrema`, `geometry`,
`breakout_tb`) — the older/shape-based schemes the filter was written for.

**Fix:** `filter_peaks_by_future_performance` is now called with
`window=self.HORIZON` at both call sites. Strictly more correct in both cases —
a no-op for the 18 that self-filter, and the intended horizon for the 9 that
don't.

**`PEAK_WINDOW` is deliberately NOT changed.** It is aliased as
`BaseNNStrategy.lookahead_window`, which NNMT uses to shift its four aux-task
labels (`regime`, `risk`, `flow`, `momentum`) by `-lookahead_window + 1`. Setting
`PEAK_WINDOW = 48` would have moved those by 8x as a silent side effect.

Verified on breakout_gbb (min_gain 0.005): the filtered count now equals the
labeler's at every horizon — 11205 / 14797 / 19193 / 20757 / 22830 for
H = 3 / 6 / 16 / 24 / 48, against a flat 14797 before.

### Consequence: this changes labels for every NN strategy on the NEXT retrain

Existing trained models are unaffected — they are on disk and load unchanged, so
a running dry run is not touched. But any retrain now produces a materially
different model: at H=48 the breakout label set grows **14797 -> 22830 (+54%)**.

Every currently tuned parameter — NNNC's `entry_rvol_threshold` 2.0,
`prediction_threshold` 0.69, ROI 0.05, grace 6.5h — was fitted against models
trained on the *effective 6-bar* labels. **Retune after retraining, do not assume
they transfer.**

The open question this creates is worth measuring rather than assuming: NNNC has
been training on 90-minute labels while its configuration said 12 hours. Whether
the intended horizon is actually better is now testable for the first time —
retrain NNNC_MLX and compare against the current production model.

## A/B of the fix: correct, and P&L-NEUTRAL

Matched pair, identical config and training window (`20220101-20250831`, W1 OOS),
differing only in the filter window. Production `NNNC_MLX` was deliberately not
retrained — its own training window is unrecoverable, so comparing against it
would confound the fix with the window.

The fix did what it should to the labels and the model:

| | initial peaks | after filter | val_mcc | precision |
|---|---|---|---|---|
| pre-fix (window 6) | 11457 | 6456 (−44%) | 0.6045 | 0.4494 |
| FIXED (window 48) | 11457 | **11457** | **0.7590** | **0.6422** |

77% more training labels, `val_mcc` +0.155, precision +0.193 — a far larger
metric shift than any other intervention in this study.

And it changed the trading almost not at all:

| window | pre-fix | fixed | identical trades? |
|---|---|---|---|
| W1 (OOS) | 99 tr, 5.84% | 99 tr, 5.84% | **YES, byte-identical** |
| W2* | 194 tr, 6.30% | 197 tr, 5.35% | no, 95.5% overlap |
| W3* | 61 tr, 0.50% | 61 tr, 0.50% | **YES, byte-identical** |

Summed −0.95pp, entirely from W2 (in-sample, 3 trades' difference). W1 payoff
identical to the decimal: win 56.6%, avgWin 4.48%, avgLoss 3.81%.

**Keep the fix on correctness grounds** — `HORIZON` now means what it says, and
the 9 labelers that rely on the filter get it at the intended horizon. Do NOT
expect P&L from it.

**This also retracts the "retune after retraining" warning above.** The trades
barely move, so the existing tuning transfers.

## The finding that unifies the whole study

A bug fix that added 77% more labels and moved `val_mcc` by +0.155 produced
BYTE-IDENTICAL trades on two of three windows. The reason is the one measured
directly on NNMT: **the guard chain, not the model, selects the entries.**

That is the through-line for everything tested here:

| intervention | changes | P&L effect |
|---|---|---|
| native trailing stop off | mechanism | **+4.5pp NNNC, +12.0pp NNMT** |
| `entry_rvol_threshold` retune | GATE | **+7.0pp** |
| guards-off → quality guards kept | GATE | **+18.5pp** |
| aux head weights | model | none (99% entry overlap) |
| exit label level→crossing | model | none on economics |
| peak-filter fix | model | none (identical trades) |
| prediction_threshold sweep | model | win rate frozen at 58.2% |

**Gate and mechanism parameters move P&L. Model parameters do not.** Every
positive result in this study came from the first two categories, and every null
came from the third. That is worth more than any individual number here.

## Consequence for the GANs (and a validation hole it exposed)

GAN training labels come from the SAME filtered pipeline:
`CreateGANBase` calls `get_training_labels`, which calls
`get_train_buy_signals`/`get_train_sell_signals`, which apply the peak filter.
**Every saved GAN was therefore conditioned on the 6-bar-effective label
distribution**, and is stale against a post-fix strategy.

**The metadata validator would NOT have caught it.** `_gan_expected_metadata`
recorded `min_buy_gain_threshold`, `min_sell_loss_threshold`, `training_type` and
`horizon`. A pre-fix GAN saved `horizon=48`; a post-fix strategy expects
`horizon=48` — **match, validation passes** — while the label sets differ by ~77%.
That is precisely the silent label/GAN drift the validator exists to prevent, and
it slipped through because the effective window was never part of the config.

**Fix, wired through three sites:**
- `TrainingEngine._gan_expected_metadata` now includes `label_filter_window`
  (`LABEL_FILTER_WINDOW or HORIZON`).
- `CreateGANBase._master_save_kwargs` records it when a GAN is trained.
- `GANInterface._HISTORICAL_DEFAULTS = {"label_filter_window": 6}` — because we
  KNOW the historical value. Every NN strategy used `PEAK_WINDOW = 6` (only Debug
  classes overrode it), so a GAN with no recorded value was trained at an
  effective horizon of 6. Without this, the validator's missing-key leniency
  ("assumed to match") would have grandfathered exactly the GANs that are wrong.

Verified:

| saved metadata | result |
|---|---|
| pre-fix GAN, no key | **BLOCKED** — `label_filter_window: saved=6, expected=48` |
| pre-fix GAN, key = 6 | **BLOCKED** |
| post-fix GAN | passes |

**Practical impact.** `GANMetadataMismatchError` propagates as-is with a per-key
diff, so affected strategies fail loudly at GAN load with an actionable message
rather than training on mismatched conditioning. Production is unaffected —
`NNNC_MLX` and `NNMT_MLX` both inherit `gan_type = NONE`. The GAN variants
(`NNNC_DDPM_MLX`, `NNNC_CGP_MLX`, `NNNC_WGAN`, `NNMT_DDPM`, `NNMT_CGP_MLX`,
`NNMT_WGAN`, ...) need their GANs retrained before use.

Given the recorded verdict that GAN augmentation does not help P&L at any
fidelity, retraining them is not urgent — but they must not be used until it
happens, which is now enforced rather than trusted.

## Confirmed on NNMT — second family, same answer

Matched pair, same protocol. Isolation is cleaner here: NNMT's four AUX labels
shift by `lookahead_window` (still `PEAK_WINDOW = 6`, deliberately unchanged), so
ONLY the trading head's labels move.

Control verification worth noting: `NNMT_MLX_hzold` trained to
**`e99d73df48672179`** — byte-identical to `NNMT_MLX_s42` from the seed study,
trained on the same window *before the fix existed*. So `LABEL_FILTER_WINDOW = 6`
reproduces the pre-fix pipeline exactly rather than approximating it, and
training is fully deterministic given seed/window/labels.

| | buy labels | sell labels | trading_mcc | trading_prec | risk | mom | flow |
|---|---|---|---|---|---|---|---|
| pre-fix | 6092 | 5224 | 0.6057 | 0.4497 | 0.560 | 0.632 | 0.508 |
| FIXED | **10406** (+71%) | **9529** (+82%) | **0.7609** | **0.6458** | 0.562 | 0.632 | 0.509 |

Aux heads unchanged to three decimals — the isolation worked.

| arm | trades | W1 (OOS) | W2* | W3* | summed | PF | win% |
|---|---|---|---|---|---|---|---|
| pre-fix | 648 | **10.71** | 8.34 | 1.24 | 20.29 | 1.39 | 58.0 |
| FIXED | 649 | **10.71** | 8.33 | 1.24 | **20.28** | 1.39 | 57.9 |

**W1 byte-identical (176 trades), W1 payoff identical to the decimal, summed
−0.01pp.** One trade differs in W2.

### What this settles

**Production `NNMT_MLX` does not need retraining for performance.** Its live model
was trained under the bug, so it is inconsistent with the code that produced it —
worth regenerating for hygiene whenever convenient, but there is no P&L in it.

**The tuning transfers.** `entry_rvol_threshold 0.25` and the rest were fitted
against old-label models and are unaffected, exactly as on NNNC.

**Fifth large metric shift to convert to nothing, now across two families.**
`val_mcc` +0.155, precision +0.196, P&L −0.01pp. Combined with the NNNC result
(identical trades on 2 of 3 windows) this is about as clean a statement as the
data can make: on gate-dominated strategies, model-side improvements do not reach
the trades. See [[feedback_gate_and_mechanism_move_pnl_not_model]].
