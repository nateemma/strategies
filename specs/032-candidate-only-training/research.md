# Research: Candidate-Only Training for NNNC Models (032)

All decisions below were checked against the code on 2026-10-13.

## R1. Where the filter goes in the training path

**Decision**: Inside `TrainingEngine.prepare_training_data`, per pair, after `df_to_tensor` and the
label `offset`, before the per-pair arrays are concatenated. The mask comes from the RAW pair frame
(`dataframe[i]`, before `scale_dataframe`), sliced exactly like the tensors:
- `[:train_end][offset:]` for train;
- `[test_start:][offset:]` for validation.

It is applied with one index array to the tensor, the one-hot labels and, if present, the P&L
weights.

**Rationale**:
- `populate_indicators` passes the full indicator frame to `maybe_train`, so `guard_metric` and
  `bb_width` are present on every NNNC class, raw8-feature GBR included, because its label needs
  them.
- With `tensor_method` 3 (MLX), `df_to_tensor` returns one zero-padded window per row, so row i of
  the tensor is the window ending at bar i. Masking after windowing keeps every window contiguous.
  Masking raw rows first would join bars that were not adjacent.
- `offset` is 0 for method 3 and `seq_len-1` for method 0. Slicing the mask the same way keeps
  both correct.

**Guard**: if `enhance_training_data` changes the train row count (GAN augmentation), the mask no
longer aligns. The filter refuses with an error, since no candidate-only arm uses augmentation.

**Alternatives rejected**:
- Filtering in `get_training_labels`, which would break windows.
- Sample weights of 0 on ineligible rows, which keeps the rows in batches and in the validation
  MCC.
- A new classifier type (constitution: one training path).

## R2. Class weights and the empty SELL class

**Decision**: No change to `get_training_class_weights`. It already gives weight 0.0 to a class
with no samples (`class_counts[i] > 0` check, single-task branch). NNNC sets
`apply_label_conflict_resolution = True`, so it uses the kept TRAIN labels.

**Rationale**: On eligible bars the SELL label is never set (`labels_gbb_path_sell` needs
`guard_metric > 0.3`), so SELL is empty. Its weight is 0, and the three-way head and every
prediction and threshold path stay as they are.

## R3. How the predicate is declared (strategy-agnostic)

**Decision**: The strategy declares `TRAIN_ROW_FILTER` as data: a tuple of `(column, op, value)`
conditions, ANDed, with `op` one of `lt`, `le`, `gt`, `ge`. A new generic helper,
`Framework/RowFilter.py`, has `row_filter_mask(df, spec) -> np.ndarray[bool]`. It validates the
spec, raises on an unknown column or op, and treats NaN as failing. The same helper is used at
three points:
1. training (R1);
2. the entry gate (R5);
3. threshold picking (R6).

So training, entry and threshold select exactly the same bars.

**Rationale**:
- Data, not a callable, so it hashes deterministically into the model key (R4).
- Framework holds no NNNC values. The gbb numbers (0.3, 0.035) live in the NNNC mixin.
- `TrainingEngine` reads `getattr(self, "TRAIN_ROW_FILTER", None)` and defines NO class default.
  So no existing class gains an attribute, and the 023 fingerprint and the 030 audit see nothing
  new on existing classes.

## R4. Model key: new key for the arms only

**Decision**: The NNNC mixin `CandidateTrainingMixin` (`NNNC/_candidate_training.py`) declares
`TRAIN_ROW_FILTER` and its own `MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER",)`.
`ModelKey.merged` unions that tuple along the MRO, so only classes with the mixin hash it.

**Checked**:
- `ModelKey.resolve` hashes attributes by name over `merged(cls, "MODEL_KEY_ATTRIBUTES")`.
- It hashes methods by `file:qualname` of the defining class (`_where`), not the source text. So
  editing the body of `prepare_training_data` / `train_model` does not re-key any model.
- `MODEL_CODE_VERSION` is NOT bumped, because existing training is unchanged when the filter is
  absent.
- The audit test (`Framework/tests/test_model_key_audit.py`) requires names defined in more than
  one class to be classified. `TRAIN_ROW_FILTER` is defined once (the mixin) and is listed in its
  `MODEL_KEY_ATTRIBUTES` anyway.
- The mixin's other names (`HELD_PAIR_ADD`, already classified) are checked by running the audit.

## R5. Entry gate

**Decision**: The mixin overrides `get_entry_conditions`: `mask = super(...) & row_filter_mask(df,
TRAIN_ROW_FILTER)`. It then recomputes `_st_tags = self._model_tags(mask, n)`, the same pattern as
`WidthBookMixin`. The model still predicts on all bars, because the exit side reads prediction
columns. `HELD_PAIR_ADD = None` on the mixin, matching the add-off controls and width books.

## R6. Threshold (Clarification Q1: A)

**Decision**: Use the existing tool and target rate. The controls' thresholds (e.g.
`evaluation/folds/thresholds_NNNC_A_Raw_Open.json`, 0.734–0.783) were picked by
`pick_threshold.py` at target rate **0.014735**: the incumbent's BUY rate at 0.80 over 1,291,078
bars (commit 63e046c).

**Changes**:
- `effective_buy_probs` gains an optional `keep` mask. Bars outside it are effective probability 0
  (counted, not dropped).
- `predicted_buy_probs` passes `row_filter_mask(trimmed, strat.TRAIN_ROW_FILTER)` when the strategy
  declares one. With no filter, the output is byte-identical.

**Fix needed first**: `cmd_pick` checks `run_folds.model_exists(cls)`, which looks in
`saved_data/<class>/`. Under 030 keys a fold's weights live at the key's location, so it would
SKIP every new fold. `cmd_pick` (and `cmd_control_rate`) must use
`run_folds.fold_has_model(fold, base, config, seed)`. This is a known open item.

## R7. Run order: train, then pick, then backtest

**Decision**: `run_folds` trains and evaluates in one loop and reads `thresholds` from the plan
before the loop. So a model's threshold can only be picked after it is trained. Per model:
1. a train-only driver (`research/candidate_training/train.py`) calls `run_folds.generate_fold` +
   `train_fold` + `fold_has_model` fold by fold. It is one process, one trainer, and resumable
   (skips folds with a model);
2. `pick_threshold.py --target-rate 0.014735 --out thresholds_co_<m>.json`;
3. write `thresholds` into the plan;
4. `run_folds --plan plan_co_<m>.json`, which finds every model, so it skips training and only
   evaluates.

ARO completes before GBR starts. Nothing else trains or backtests meanwhile.

## R8. Evaluation stages: real records, simulator as the instrument check

**Decision (refines the spec's US2)**:
- Stage 2/3 books come from the real 15-fold backtest records, not the simulator. Training
  dominates the cost (about 12 min a fold), a fold backtest adds a few minutes, and the comparators
  (`wf_aro_g24`, `fs_gbr_gr`, `w_aro`, `w_gbr`) are real records too. So real against real
  removes simulator error.
- The simulator (`research.nnnc_framework.entries`) is still run on each arm's own entries as the
  fidelity check (FR-011/SC-005, ≥ 95%), the instrument audit. It is also the basis for later
  screens on these models.

**Pre-registration**: One file, `research/candidate_training/preregistration.json`, holds all
three stage rules. It is committed BEFORE training starts, so every arm record postdates it and
the judge's refusal rule applies. The report evaluates the stages in order and marks later stages
"not reached" when one fails. Records for unreached stages exist but are not read.

**Stage rules (per model)**:
- **S1 mechanism**: the within-candidate AUC of the retrained model exceeds the current model's in
  BOTH halves. It is measured on identical eligible bars and labels, from fold signal tables built
  by `research.nnnc_framework.signals` for the new fold classes.
- **S2 early half (f00–f07)**: the arm's early pp > the width book's early pp AND > the control's
  early pp.
- **S3 late half (f08–f14, confirmatory)**:
  - `judge_one(arm, control)` = PASS;
  - AND the arm's late pp > the width book's late pp;
  - `judge_one(arm, width book)` is also reported.
- **Decision**: all three pass → ADOPTED as a candidate (promotion is the user's decision);
  otherwise NOT ADOPTED, naming the failing stage. Zero trades on any fold → INCONCLUSIVE for that
  model.

**Labels**: S2/S3 are PROVISIONAL (one era, 2023–26). There are no 2022-trained NNNC folds, so
constitution II's distant era is not available. This is the same limitation as 031's NNNC books
and is stated in the report.

## R9. Liveness and parity checks (constitution IV)

- **Liveness**: per fold, report:
  - the kept training-sample count and share (expected about 20%);
  - the train BUY share (expected about 50%);
  - the SELL count (expected 0).

  These are written by the filter to a sidecar in the model's folder (`row_filter.json`).
- **Parity (off)**:
  - a test builds training data for an existing NNNC class with and without the code change and
    compares the arrays;
  - every pinned model-key and fingerprint test passes unchanged;
  - `wf_aro` / `w_aro` per-fold trades are not re-run (nothing they load changed), but their model
    keys are asserted unchanged.
- **Entry parity**: on the arm's records, every entry bar satisfies the predicate (0 violations).
