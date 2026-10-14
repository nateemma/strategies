# Contracts: Candidate-Only Training (032)

## C1. `Framework/RowFilter.row_filter_mask(df, spec) -> np.ndarray`

- `spec` must be a non-empty tuple of `(column: str, op: "lt"|"le"|"gt"|"ge", value: float)`.
  Anything else raises `ValueError`.
- It returns a bool array with `len(df)` elements, the AND of the conditions. NaN gives False.
- It raises `KeyError` naming the column if a column is missing.
- It is pure: no I/O, and it does not mutate `df`.

## C2. `TrainingEngine.prepare_training_data` with a filter

- With `getattr(self, "TRAIN_ROW_FILTER", None)` absent or None, the returned arrays are identical
  (dtype, shape, values) to the pre-change code. A fixture test pins this on a synthetic frame for
  `tensor_method` 0 and 3, with and without `%pnl_weight`.
- With a filter, for each pair the train mask and validation mask are taken from the RAW frame,
  sliced like the tensors (R1), and applied to the tensor, labels and weights of that split.
- If the train row count after `enhance_training_data` differs from the pre-enhance count, it
  raises `ValueError` ("row filter cannot align with augmented training data").
- If the filter keeps zero train rows, or zero validation rows, across all pairs, it raises
  `ValueError`.
- It sets `self._row_filter_report` (the data-model RowFilterReport). `_train_keyed` writes it as
  `row_filter.json` at the model's location after a successful fresh training.

## C3. `CandidateTrainingMixin` (`NNNC/_candidate_training.py`)

- It sets the gbb `TRAIN_ROW_FILTER`, `MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER",)` and
  `HELD_PAIR_ADD = None`.
- `get_entry_conditions(df)` returns the parent's mask AND `row_filter_mask(df, TRAIN_ROW_FILTER)`.
  `_st_tags` is recomputed from the final mask.
- The model key differs from the same base without the mixin. Existing classes' keys are
  unchanged (pinned by the test).

## C4. `pick_threshold`

- `effective_buy_probs(df, keep=None)` with `keep=None` is byte-identical to today. With `keep`,
  rows where `keep` is False count as 0.0, and they are dropped only if unscored, as today.
- `predicted_buy_probs` passes `keep = row_filter_mask(trimmed, spec)` only when the strategy
  declares `TRAIN_ROW_FILTER`.
- `cmd_pick` / `cmd_control_rate` decide "trained?" with `run_folds.fold_has_model(fold,
  plan["base_strategy"], plan["config"], seed=1)`, not `model_exists(cls)`.

## C5. `research/candidate_training/` scripts

| script | does | refuses |
|---|---|---|
| `build_arms.py` | writes `NNNC/NNNC_CO_<M>.py` and `.json`, and `plan_co_<m>.json` from the control plan (same folds, snapshot and config; new classes and tags; `model_dir` = the fold class) | — |
| `train.py --model ARO` | trains the folds that lack a model, one at a time, then prints each RowFilterReport | runs if another freqtrade backtest or training process is running |
| `auc.py` | S1 from signal tables (`signals.py` on the new plans) | — |
| `judge.py` | evaluates S1→S2→S3 per model and writes `verdicts.json` and the fidelity and entry-violation checks | refuses (exit 2) if the pre-registration is uncommitted or dirty, or if any arm record predates its commit |
