# Follow-up study outline: per-fold guard tuning

**Proposed by**: feature 021, as the experiment it identified but could not run.
**Status**: outline only — input to the user's own planned study, not a spec.

---

## The question

**Is the quality-guard chain, tuned per fold on that fold's own training window and with NO
model, a tradeable strategy?**

And the one that matters more: **does the model add anything on top of it?**

## Why this is the right next experiment, in one paragraph

Feature 021 established that the `gbb` trigger alone is flat (+2.64pp, 7/15) and that nothing
within it is learnable on average (conditional MCC +0.05) or stably rankable in the tail (sign
inverts between windows, rank correlation −0.648). Yet `gbb` + model + a **per-fold** threshold
earns +39.98pp in 15/15 folds. The two facts are reconcilable only if the signal is real but
**regime-dependent and therefore only exploitable by something that re-tunes** — which is what
per-fold selection is and what a fixed-split probe cannot be. Separately, a one-off arm with
**globally** tuned guards and no model returned **+18.76pp, 12/15 folds, ~+5.1pp/yr** at a
per-trade EV of +0.0147 (95× the unguarded arm). That number is the reason to run this study
and is simultaneously untrustworthy — see Phase 0.

## Phase 0 — provenance audit, BEFORE any compute

The single most important task, and cheap.

1. **Where did the pre-2026-09-06 tuned guard values come from?** (`entry_adx_threshold` 10.0,
   `entry_atr_pct` 0.013, `entry_bb_width_threshold` 0.094, `entry_close_norm_threshold` 0.6,
   `entry_guard_threshold` 0.1, `entry_rvol_threshold` 1.9.) Git archaeology plus any `.fthypt`
   logs. **If their hyperopt window overlaps the 15 evaluation folds, the +18.76pp is
   in-sample and this study's motivating number evaporates.** Establish this first; it changes
   whether Phase 2's arm B means anything.
2. **Confirm the incumbent's configuration from artefacts, not prose.** Feature 021 twice
   trusted a provenance field that asserted rather than measured (`volume_filter_active` was a
   literal; `guards` recorded the CLI flag, not the effective value). Read the strategy class
   and the params file.
3. **Re-state the baselines** that already exist, so nothing is re-run needlessly:
   `GbbOpen` +2.64pp (7/15), `GbbGuardedOpen` +18.76pp (12/15), incumbent +39.98pp (15/15).

## Phase 1 — the per-fold tuning instrument

* **Tune on the fold's TRAINING window only; evaluate once on its test window.** Mirrors what
  the incumbent does for its model threshold, which is the mechanism under test.
* **`entry_enable_guards` stays `True`.** It is not the on/off switch it looks like: it also
  gates the `volume > 0` data-sanity check (`BaseStrategy.py:295`). Guards are opened by
  **neutralising thresholds**. 34.26% of bars in this dataset have `volume <= 0`.
* **Hold the exit fixed.** Prior sweeps show it is calibrated; a moving exit makes every arm
  incomparable and invalidates the shared `exit_config_hash` invariant.
* **Tune the guard space ONLY.** No model, no exit params, no entry-timing. The repo's own
  lesson: with thin data, joint search over many degrees of freedom fits the specific backtest.
* **Reduce the space before searching it.** Six thresholds on a ~90-day training window is too
  many. Use the existing guard-ablation evidence to pick the 2–3 that bind (`rvol` is recorded
  as dominant), and sweep those. Tune one side, lock it, tune the other.
* **Constrain the search to a POWERED operating point.** The globally-tuned guards admitted
  198 trades across 15 folds — 2.43% of the unguarded arm, with one fold trading **zero**
  times. Reject any candidate configuration below a pre-registered minimum trades/fold, or the
  study measures its own noise. This repo has recorded exactly this trap: *tight guards trade
  only the confident tail, so such changes look falsely inert.*
* **Loss function**: `WalletCalmarHyperOptLoss`, which reconstructs the equity curve from the
  order ledger — a final-amount proxy previously hid a 36% drawdown as 4%.

## Phase 2 — the arms

One exit, one fold plan, 15 folds each, so arms are comparable fold-for-fold.

| arm | guards | model | status |
|---|---|---|---|
| **A** control | open | none | **done** — +2.64pp, 7/15 |
| **B** global tuning | historical tuned values | none | **done** — +18.76pp, 12/15 *(suspect; Phase 0)* |
| **C** per-fold tuning | tuned on each fold's train window | none | **THE NEW ARM** |
| **D** incumbent | open | per-fold threshold | **done** — +39.98pp, 15/15 |
| **E** both | tuned per fold | per-fold threshold | optional, run only if C succeeds |

### The comparisons that decide it

* **C vs A** — do per-fold-tuned guards beat guards-open with no model at all? *Is the guard
  chain a strategy?*
* **C vs B** — if C ≪ B, then B's +18.76pp was the tuning window leaking, and that is the
  study's finding. **This is the falsification test and it must be stated before the run.**
* **C vs D** — does a model-free guard strategy match the incumbent? If yes, the model is
  ornamental and this repository's *"guard ablation: model adds NOTHING"* generalises from
  entry counts to return.
* **E vs C** — does the model add anything once the guards are tuned? The sharpest question in
  the set, and the only one that can rehabilitate model work.

## Phase 3 — power, and the checks 021 learned the hard way

* **Paired statistics on the symmetric difference.** These arms share most trades; treating
  them as independent overstated an oracle arm's resolution by 4.7× once already.
* **Persistence across temporally distant folds.** Report per-fold, and split the 15 into
  early/late halves — an edge must appear in **both**. Feature 021's tail signal looked real
  in one window and **inverted** in the next (ρ = −0.648, p = 0.017); that is the failure mode
  to design against, not to discover afterwards.
* **Verify every tuned parameter actually bit.** There are three documented ways a hyperopt
  parameter does nothing with no error. Diagnose from `.fthypt`, never the console, and hash
  the trade list per fold — if two configurations produce identical `trades_hash`, the
  parameter was inert.
* **Flag unpowered folds rather than averaging them in.** Any fold below the trades floor is
  reported as unpowered, not as a zero.
* **One trainer at a time.** Parallel runs caused OOM and Metal faults; `-j>1` hyperopt needs
  the pickle-by-value guard in the family base.

## Pre-registered decision rule

State before running, so the result cannot be re-read afterwards:

* **C beats A by more than the harness floor (>3–4pp/yr) with a paired CI excluding zero** →
  the guard chain is a genuine model-free lever, and per-fold tuning is the mechanism.
* **C ≈ A (within the floor)** → B's +18.76pp was contamination or a 3-fold accident, and the
  guard chain is not a lever either. A clean negative.
* **C ≈ D** → the model is ornamental; move all effort to gate/mechanism work.
* **E ≫ C** → the model adds real value *on top of* tuned guards, which would be the first
  positive model-side result in this repository for some time and would reopen model work.

## Out of scope, with the evidence for each

Changing the exit (calibrated; prior ladder and sweeps). Expanding the universe (*study 4: no
lever*). Adding features (*features are not the lever*: 24 indicators vs 8 raw OHLCV,
−1.23pp/yr, CI spans zero). Re-tuning the label — feature 021 established that labels in this
family are interchangeable, so label work is the one direction already excluded by evidence.
