# Research: Model Naming by Model Fingerprint (030)

These sources come from two code surveys done on 2026-10-12, reading only (file:line references are
in the survey notes summarised here). Every decision below resolves a design unknown in the plan.

## R1. What shapes the weights (the key's inputs)

**Finding.** The training path is:
1. `BaseNNStrategy.populate_indicators`;
2. indicators, then `add_additional_indicators`, then `get_training_labels`;
3. `TrainingEngine.maybe_train`, then `train_model`, which runs prepare, preprocess, class
   weights, markov and shuffle, then `classifier.train`.

What it reads, beyond the code itself:

| group | inputs |
|---|---|
| classifier | classifier class and `classifier_type`; `seq_len`; `batch_size`; classifier training knobs (`max_epochs`, `early_patience`, `weight_decay`, `learning_rate`); instance attributes set by mixins (`train_seed`, `pnl_loss_alpha`, `ponder_steps`); NNMT `_CLASSIFIER_*` overrides; `tensor_method` (3 under MLX, 0 otherwise; NNMT hard-codes 0) |
| labels | `TRAINING_TYPE`; `HORIZON`; `MIN_BUY_GAIN_THRESHOLD` / `MIN_SELL_LOSS_THRESHOLD`; `LABEL_RVOL_THRESHOLD`; `LABEL_MAX_DRAWDOWN`; `LABEL_FILTER_WINDOW`; `apply_label_conflict_resolution`; `LABEL_PARAMS(_SELL)`; `GUARD_T` / `BBW_T`; NNMT `PROFIT_*`, `lookahead_window`, `task_thresholds`, regime/risk/flow/momentum spans |
| features | `include_list`, `pre_normalized_columns`, `one_hot_columns`; the scaler and PCA *files* (by name attribute); `use_pca_reduction`, `pca_passthrough_columns`; `use_post_gan_scaling` (plus the hard-coded `main_tensor_scaler.pkl`); NNPredict/NNWavelet coefficient-transform settings |
| augmentation | `gan_type`, `gan_augment`, `gan_target_ratio`, `gan_augment_seed`, `gan_condition_tasks`, `gan_passthrough_columns`, every `gan_synth_*` and `gan_inference_*`, `gan_entropy_guidance*`; the GAN generator files; the entropy-guidance model (another model, loaded by class name) |
| data scope | `TRAIN_DATA_SPLIT` (the validation tail drives early stopping); `shuffle_train_data`; `aggregate_pairs`; the whitelist (membership and order); `timeframe`; the training timerange; **`startup_candle_count`** (R2); the candle data on disk |
| seed | `MLX_SEED` (read at import, default 42; the harness passes `--seed`, default 1); `train_seed` |

**Decision.** The key is built from four parts:
1. a **declared attribute list** — a generic one in `BaseNNStrategy`, merged with family
   extensions (NNMT, NNPredict and so on);
2. the **defining class of each training-path method** (R3);
3. **file identities** (R4);
4. **run inputs:** training window, whitelist, timeframe, startup, seed, and the model-code
   version.

**Rationale.** An explicit list is readable and keeps `Framework/` generic (FR-021). The risk is
an input left off the list, which would cause a collision. R5 guards against that.

**Alternatives considered.**
- *Hash every class attribute except a declared post-model list.* This fails safe, but any
  cosmetic attribute (`dbg_verbose`, a docstring constant) would rekey a model and force a
  retrain. It also can't tell a training-path method override from an inference one.
- *Hash method source code.* Any comment edit in `BaseNNStrategy` would rekey every model on disk.

## R2. `startup_candle_count` is a training input

**Finding.** Freqtrade loads `required_startup` extra bars and runs `populate_indicators` on the
full frame. It trims only afterwards (`freqtrade/optimize/backtesting.py:364-370, 1835-1839`).
Training happens inside `populate_indicators`, so the warm-up bars are in the training set:
- their NaN indicators become 0;
- they shift the 80/20 split point.

**Decision.**
- `startup_candle_count` is part of the key.
- A strategy that runs with a different warm-up from its model's training (the 4096-bar gate arms
  on 64-bar weights) declares **`MODEL_TRAINED_WITH = {"startup_candle_count": 64}`**. That makes
  the difference between runtime and training explicit, and it is recorded in the manifest's
  `users` list.

**Rationale.** This keeps the key honest. Today the gate arms borrow those weights silently
through `BorrowedModelMixin`; after this change the reuse is declared and visible.

**Rule.** `MODEL_TRAINED_WITH` may name only inputs that are *also* used at prediction time, such
as warm-up. At predict time the declared value is used for the key only, never for behaviour.

## R3. Training-path methods: the defining class, not the source

**Decision.** For each method on the declared training-path list, the key records the qualified
name of the class that defines it (via the MRO). A subclass that overrides a label, feature or
classifier method therefore gets its own key. An edit to an existing method does not rekey;
that is what the manual `MODEL_CODE_VERSION` is for (spec edge case). The list comes from the
survey:
- **Framework:** `populate_indicators`, `iteration_init`, `add_additional_indicators`,
  `get_training_labels`, `get_train_buy_signals`, `get_train_sell_signals`,
  `filter_peaks_by_future_performance`, `augment_training_signals`, `get_classifier_type`,
  `get_classifier`, `maybe_train`, `train_model`, `prepare_training_data`,
  `preprocess_training_data`, `enhance_training_data`, `get_training_class_weights`,
  `scale_dataframe`, `process_one_hot_columns`, `apply_pca`, `clean_for_tensor`,
  `normalise_for_gan`, `get_normalized_size`, `get_storage_location`.
- **Family extensions:**
  - NNMT: `get_*_target`, `get_trading_classes`, `get_class_weights`,
    `_apply_classifier_overrides`, `_balance_iteratively`, `_apply_post_gan_scaler`.
  - NNPredict and NNWavelet: their label and coefficient helpers, and `maybe_train`.
  - Sklearn: `train_model`.
  - Anomaly: `get_trading_classes`.

**Rationale.** In the survey, 4 NNNC classes override `add_additional_indicators` and 6 override
`augment_training_signals`. NNMT's Lbl02 arms override `get_profit_target`. All of these change
the weights without changing a single attribute.

## R4. File inputs are keyed by content

**Finding.** The scaler, PCA and tensor-scaler files are pre-built (`CreateScalers`) and shared
at the store root by *name*. They are re-read at predict time. The GAN scaler and the coefficient
scalers are fitted-if-missing during training. The GAN generators live in `GANs_PostScale/<type>/`.
The entropy-guidance model is loaded by class name.

**Decision.**
- **Pre-built files.** The key holds the SHA-256 (first 16 hex characters) of the main scaler and,
  when `use_pca_reduction` is on, the PCA file. It also holds the tensor scaler when
  `use_post_gan_scaling` is on, and the GAN generator files when `gan_augment` is on.
- **Fit-if-missing files** (GAN scaler `a`, coefficient scalers). The key holds their *name*. A
  missing file at training is recorded as `fitted-at-train` in the manifest, and the hash is taken
  after the fit.
- **Entropy-guidance model.** The key holds that model's own resolved key, or its legacy key.

Hashes are computed once per process and cached by (path, mtime, size).

**Rationale.** FR-001 asks for scaler identity by content. Regenerating a scaler must rekey every
model that depends on it, so a stale model is never served (constitution quality gate 7).

## R5. Completeness audit (the guard against collisions)

**Decision.** A test scans the class bodies of every in-scope family (NNNC, NNMT, NNPredict,
NNWavelet, Anomaly, Sklearn, plus Framework and utils mixins) for assigned class attributes and
`*_params` dictionary keys. Every name found must appear in exactly one of:
- `MODEL_KEY_ATTRIBUTES`: weight-affecting;
- `MODEL_KEY_IGNORED`: post-model, such as guards, thresholds, exits, stops, sizing, family exits,
  regime gate, trigger and entry mode, debug or plotting settings, and names declared but never
  read on the training path (`num_epochs`, `stride`, `refit_model`, `model_per_pair`,
  `combine_models`, `filter_signals`).

An unclassified name fails the test, so every new attribute has to be classified when it's added.

A second audit covers methods: any training-path-named method defined in a family class must be
on the training-path list.

**Rationale.** A key collision is worse than a retrain (spec). An explicit list without an audit
would decay the moment someone adds an attribute.

## R6. Training window and seed at load time

**Finding.** A model is evaluated on a different timerange from the one it was trained on. The
seed is read from the environment at import, and evaluation runs don't set it:
- fold models were trained with `MLX_SEED=1`;
- single models with 42 (the default) or 7/42 (`_sN` classes).

So neither the training window nor the seed can be read from the run that *loads* a model.

**Decision.**
- **Declared pins.**
  - `MODEL_TRAIN_TIMERANGE`: `run_folds` writes it into every generated fold class, from the plan's
    `train_timerange`.
  - `MODEL_SEED`: written by `run_folds` from `--seed`. On non-fold classes it is optional; the
    default is `MLX_SEED`, or 42 when that is unset.
  - The training step uses the pinned seed (`train_fold` already sets `MLX_SEED`; the resolver
    checks they agree).
- **Undeclared window.** When `MODEL_TRAIN_TIMERANGE` is undeclared (manual, non-fold runs), the
  window component is the literal `unpinned`, and the manifest records the range actually trained
  on. At load, an unpinned model prints that range, with a warning when it overlaps the run's
  timerange (a leakage warning).
- **Harness keeps its model checks.** `run_folds` still checks for a model by fold, through the
  key (FR-017).

**Rationale.** Declared pins make the key the same at train time and at evaluation. The
`unpinned` value keeps the semantics for manual runs as they are today (one model per class
setup), with the actual window now recorded.

## R7. Where models live; migration by index

**Decision.**
- **New models:** `saved_data/models/<prefix>__<hash>/<prefix>__<hash>.safetensors` (or the
  classifier's suffix), plus `manifest.json`.
  - The prefix is `<classifier>_<label>_g<gain>_h<horizon>`, lower-case.
  - The hash is SHA-256 over canonical JSON of the key, truncated to 16 hex characters.
  - Each model gets a unique file stem. This matters because MLX checkpoints to
    `/tmp/<stem>/`, so a shared stem would collide.
- **Index:** `saved_data/model_index.json` maps each hash to its weight file and kind (`keyed`,
  `migrated` or `legacy`). It is gitignored, like all of `saved_data/`. A copy is committed in the
  spec directory at migration time, as the audit record.
- **Migration** (`scripts/model_index.py migrate`) writes manifests and index entries. It never
  moves or rewrites weights.
  - **Walk-forward sets** (ARO, GBR, GFM, FMR, WF and the WF `_s7` sets): the key is resolved from
    the plan's fold class, with its window and seed. `manifest.provenance` is
    `reconstructed-from-plan`, recording the plan, run_id and params sha.
  - **Single models** (58 directories): a **legacy key** `legacy:<ClassName>` (Q3 → A), with a
    manifest marking unknown inputs. Only that class may resolve to it.
  - **Conflicts:** two directories resolving to one key with different weight bytes are reported
    and both left out of the shared key (FR-012).
- **Readable browsing:** `saved_data/models/<prefix>__<hash>` is a symlink to the legacy directory
  for migrated models (US4).

**Alternatives considered.**
- *Moving weights into the new layout.* Rejected: an interrupted move risks the per-fold sets,
  which cost hours each.
- *Symlinks only, no index.* Rejected: an index is easier to audit and test.

## R8. Lookup order (no behaviour change)

**Decision.** `get_model_path()` resolves in this order:
1. **A class-level `get_model_path` override.** These are the legacy redirects: NNMT (11),
   Oracle/Ora_Prec (37 inherit), NoModel, Noisy, Cap/Band/P0 and similar aliases. They are
   **kept unchanged** and recorded in a committed `legacy_redirects.json`. All of them point at
   single legacy models whose training inputs can't be reconstructed (the survey found NNMT_MLX
   trained at 0.007 while its class now reads 0.02). That meets the spec's "recorded as legacy
   redirect" clause.
2. **The index entry for the resolved key.**
3. **The legacy key `legacy:<ClassName>`,** only when a directory named for the class exists and
   is indexed as legacy.
4. **Otherwise:** the new keyed location. If nothing exists there, walk-forward evaluation fails
   loudly (FR-009); any other run trains into it and writes the manifest (FR-010).

**`BorrowedModelMixin` is removed from its users** (the twin, SIProbe, Live_*, SW_*):
- their fold classes resolve the same key as the source fold class, because only post-model
  attributes differ;
- the gate arms declare `MODEL_TRAINED_WITH` (R2);
- `NNNC_SW_MLX_*` resolve to the `NNNC_WF` keys, because `NNNC_WF_fNN` and `NNNC_SW_MLX_C2_fNN`
  share `NNNC_MLX`'s inputs.

The mixin file stays as a deprecation shim that raises if used (FR-016). Existing records
reference only class names, so no fingerprint changes.

**Detecting walk-forward evaluation.** `run_folds.evaluate_fold` sets
`FT_MODEL_EVAL_STRICT=1`, which the resolver reads. `train_fold` doesn't set it.

## R9. Verification for families without weights

**Finding.** Only NNNC and NNMT have weights on disk. The NNPredict, NNWavelet, Anomaly and
Sklearn directories are empty or absent. FR-018a's "representative strategy with existing
weights" therefore can't be met for them without training, and FR-020 forbids training.

**Decision (spec amendment, to confirm at analyze).**
- For NNPredict, NNWavelet, Anomaly and Sklearn, verification is:
  - key resolution for every class (no exceptions; distinct keys where inputs differ);
  - under strict mode, a run that stops with the missing-key message before any training starts.
- NNMT is verified by trades: `NNMT_MLX_Base` (legacy redirect) and `NNMT_Ctl_s42` (legacy own
  weights) on window W1, before and after, same day.

## R10. Training checkpoints and test_strat.sh

**Decision.**
- `test_strat.sh -f` should delete the *resolved* model path. A new
  `scripts/model_index.py path <Family> <Class>` prints that path. Without this, `-f` would leave
  keyed models in place and its post-check would fail.
- Checkpoint stems are unique per key (R7).
