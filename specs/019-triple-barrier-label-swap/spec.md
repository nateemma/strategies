# Feature Specification: Triple-Barrier Label Swap for the NNC Classifier Study

**Feature Branch**: `019-triple-barrier-label-swap` (spec directory only; commits to master)

**Created**: 2026-09-19

**Status**: Approved (author go, 2026-09-19)

**Input**: Feature 018 found that the gbb label's learnable content is its own two-threshold
trigger (guard_metric < 0.3 AND bb_width > 0.035): every architecture converges to that rule
and nothing more, so the architecture question is unanswerable under it. The author asked to
re-pose the question on a label that does not use indicators: the strict triple-barrier label
(profit barrier touched before the loss barrier within a horizon; no-touch is HOLD), whose
time-barrier fallback was a defect and has been fixed (Framework/TrainingSignals.py,
commit cf6abf5).

## Scope Statement

Classification metrics only, against the strict triple-barrier ternary label (SELL=0, HOLD=1,
BUY=2), on the same walk-forward split set, embargo, per-window scaler and class weights,
fresh-start harness, metrics and gate as feature 018. No backtests, no P&L, no trade-level
metrics. Constitution III applies: a learnable label is not an edge, and the recorded prior on
this label (MCC ≈ 0.007 on US spot, "information ceiling") makes a null the expected outcome;
the value of the study is a clean measurement of whether ANY model separates from the ridge
baseline and from the others under an instrument that has been verified end to end.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Six representatives on the strict triple-barrier label (Priority: P1)

As the strategy author, I run the six representative classifiers from 018 (RIDGE, MLP width
64, LSTM_FAM hidden 32 depth 2, Transformer d_model 32, TCN anchor, Wavenet) plus the trivial
baselines on the strict triple-barrier label at horizon 48 with ±2% barriers, three windows,
two seeds, and read a leaderboard that tells me (a) whether any model beats RIDGE by the gate
margin on a majority of windows, (b) whether any two models separate beyond seed noise, and
(c) the per-window class balance and the trivial-baseline levels.

**Independent Test**: the phase runs to completion with one fresh model per window (smoke
test passes first), the leaderboard shows all seven configs plus baselines with per-window and
pooled mean ± std, and the gate file records the decision.

**Acceptance Scenarios**:

1. **Given** the label config `{name: triple_barrier, horizon: 48, up: 0.02, down: 0.02}`,
   **When** the phase runs, **Then** every row records the label, a window-specific
   checkpoint name, `epochs_run ≥ 1`, and the class counts of its training and test rows.
2. **Given** the completed rows, **When** the gate is evaluated with the ridge reference taken
   from THIS label's RIDGE rows, **Then** the decision file lists per-config clause results
   and the viable set (possibly empty).

### Edge Cases

- A window where the label's BUY or SELL share is below 5%: flagged on the row; the gate
  treats a collapsed prediction (≥ 99.5% one class) as not viable, as in 018.
- The ATR-scaled barrier variant and other horizons are out of scope for this run; they are
  recorded as follow-ups conditional on this result.

## Requirements *(mandatory)*

- **FR-001**: The label MUST be the strict triple-barrier label as implemented in
  `Framework/TrainingSignals.py` after commit cf6abf5, selected through the study harness's
  `label` config field (horizon 48, up 0.02, down 0.02, conflict resolution on, peak filter
  off), and MUST NOT be modified during the study.
- **FR-002**: Splits (v1), embargo, per-window scaler and class weights, fresh-start harness,
  metrics and seed policy MUST be identical to feature 018.
- **FR-003**: The configuration set MUST be: MAJORITY, PERSISTENCE, RIDGE (weight decay
  1e-2), MLP_FAM width 64, LSTM_FAM hidden 32 depth 2, TRANSFORMER_FAM d_model 32, TCN_FAM
  channels 64 depth 3, Wavenet; all ind24 inputs, seq_len 16, lr 3e-4, stride 2, cap 40,
  patience 10, seeds {0, 1}; baselines and RIDGE 3 seeds.
- **FR-004**: The real-data smoke test MUST pass before the phase starts.
- **FR-005**: The gate MUST take its ridge reference from this phase's RIDGE rows only.
- **FR-006**: The report section MUST state the per-window class balance, the trivial
  baselines, the leaderboard, the gate decision, and whether any pair of models separates
  beyond 2× the seed std on a majority of windows.

## Success Criteria *(mandatory)*

- **SC-001**: 48 neural runs + baselines complete with zero failed rows.
- **SC-002**: The report answers, with numbers, whether any model beats RIDGE by ≥ 0.02 MCC
  and macro-F1 on a majority of windows, and whether any architecture ranking is resolvable.
- **SC-003**: A recorded recommendation on whether the ATR-scaled and other-horizon variants
  are worth running.

## Assumptions

- Expected outcome is a null (all models near RIDGE); the study is designed to make that
  null clean rather than to find a winner.
- Compute: about 2 hours on the single trainer.
