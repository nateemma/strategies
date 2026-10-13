# Verification (030 T019–T033)

Run 2026-10-12. Baseline commit `a62fb23`. The after-chain ran on the switched loader with
`FT_MODEL_EVAL_STRICT=1` exported.

## Migration (T019, T020)

| | count |
|---|---|
| walk-forward models keyed from their plans | **79** (WF 19 incl. 4 seed-7, ARO 15, FMR 15, GBR 15, GFM 15) |
| single-model directories → `legacy:<Class>` | 51 |
| empty directories (skipped) | 3 (NNAnomaly_MLX, NNMT_CGP_MLX, NNPredict_MLX_LSTM) |
| conflicts | **0** |
| weight files changed (`verify-bytes`) | **0 / 152** |

**Structure of the keys:**
- Within each set, folds differ only in `run.train_timerange`, plus `run.seed` for WF's seed-7 folds.
- Across sets, the differences are the known ones:
  - ARO vs WF: `classifier_type` only;
  - ARO vs GBR: features (`include_list`, `main_scaler_name`, the scaler file,
    `add_additional_indicators`);
  - GBR vs GFM: adds the label (`TRAINING_TYPE`, `LABEL_MAX_DRAWDOWN`);
  - ARO vs FMR: label, horizon, gain/loss targets, `seq_len` and features.

Index snapshot: `model_index_migrated.json`.

## Former BorrowedModelMixin users resolve to their source key (T021, SC-002)

Each user was checked offline with the mixin removed (gate arms with their startup-64 pin) and
each fold's window and seed pinned:

| class | resolves to | folds |
|---|---|---|
| `NNNC_TriggerGate`, `NNNC_SIProbe`, `NNNC_Live_C2`, `NNNC_Live_TrigGate`, `NNNC_Live_Gate`, `NNNC_Live_GateOff`, `NNNC_SW_ARO_{C2,Gate,GateOff}` | `NNNC_ARO_fNN` | 15/15 each |
| `NNNC_SW_GBR_{C2,Gate,GateOff}` | `NNNC_GBR_fNN` | 15/15 each |
| `NNNC_SW_GFM_{C2,Gate,GateOff}` | `NNNC_GFM_fNN` | 15/15 each |
| `NNNC_SW_MLX_{C2,Gate,GateOff}` | `NNNC_WF_fNN` | 15/15 each |

## Live key = migrated key (T021a)

Arm D f00 (`wf_aro_lk030`) was run with the resolver in report mode:
- the key logged from inside the backtest = migrated `lstmraw_gbbpath_g0.02_h48__58c62d0df75807bd`;
- manifest OK;
- `trades_hash` = baseline;
- 0 training;
- the record's `model_key` filled.

**Deviation:** only arm D ran here. A report-mode run returns the pre-030 class-name path, so a
gate arm (no class directory of its own) would train in report mode. The gate arms are covered
by T021 offline and by the strict after-chain below.

## After vs baseline (T030, T031, SC-001, SC-002, SC-007)

| arm | folds equal | trades |
|---|---|---|
| `wf_aro` (arm D) | 15/15 | 1420 |
| `sw_fmr_ctl` | 15/15 | 946 |
| `sw_gbr_ctl` | 15/15 | 2395 |
| `sw_gfm_ctl` | 15/15 | 2051 |
| `sw_mlx_ctl` | 15/15 | 378 |
| `wf_trgate` (twin, **no mixin**) | 15/15 | 9056 |
| `sw_aro_gate` (**no mixin**, startup pin) | 15/15 | 848 |
| `sw_mlx_c2` (**no mixin**, NNNC_MLX model) | 15/15 | 374 |
| `NNMT_MLX_Base` W1 (legacy redirect) | 1/1 | 181 |
| `NNMT_Ctl_s42` W1 (legacy own weights) | 1/1 | 179 |

- The chain logged 0 × `training on`, 0 tracebacks, and no keyed training in any fold log.
- `verify-bytes`: 152 files, 0 differences.
- Fingerprints: 511 strategies, **0 changed**.
  - This is against `fingerprints_before_regen.json`, a scan of the baseline code taken after
    the baseline chain.
  - Against the pre-chain `fingerprints_before.json` the only differences are the 15 `NNNC_FMR_fNN`
    generated fold files. The baseline chain itself re-wrote those gitignored files from the
    ForwardMAE plan's snapshot (previously written by `plan_NNNC_FMR_incexit`); the baseline code
    shows the same 15 differences.
- Arm records carry `model_key`, e.g. `sw_aro_gate_a030` f03 →
  `NNNC_ARO_f03` (`afb96951699243c9`, migrated).

**FR-022.** The live bot runs `Basket/MomentumRegimeBasket15mFast`. It imports only `Basket/`
modules, and no `Basket/` file is in the 030 diff.

## Weightless families and strict mode (T032, SC-008, R9)

**Key resolution:**

| family | classes resolved | errors | distinct keys |
|---|---|---|---|
| NNPredict | 9 | 0 | 9 |
| NNWavelet | 7 | 0 | 6 (`NNWavelet` = `NNWaveletStrategy` + `pass`: same model, correctly one key) |
| Anomaly | 4 | 0 | 4 |
| Sklearn | 0 | 6 import errors | pre-existing: `utils.ClassifierKeras` is missing (baseline.md) |

**Strict-mode backtests** (2025-01-01 → 01-03, `FT_MODEL_EVAL_STRICT=1`):

| strategy | result |
|---|---|
| `NNPredict_MLX_LSTM` | rc 1, `ModelMissing`, no training |
| `NNWavelet` | rc 1, `ModelMissing`, no training |
| `NNAnomaly_MLX` | rc 1, `ModelMissing`, no training |
| arm D with gain 0.03 (temporary class, deleted) | rc 1, `ModelMissing: ... lstmraw_gbbpath_g0.03_h48__75a1c0fe9787eb5c (strict walk-forward evaluation refuses to train)` |

No new `saved_data/` entry was created by any of the four.

## Lookups (T028, T029, FR-007, FR-014)

`model_index.py key` resolves each of these to exactly its pre-030 directory:
- `NNNC_MLX_Open_s1` / `_s7` (legacy pair);
- `NNNC_MLX_L024` (legacy own weights);
- `NNMT_Ctl_s42` (legacy);
- `NNNC_MLX` at f04 seed 7 (`NNNC_WF_f04_s7`, migrated).

Two more cases:
- `NNNC_MLX_Open` reports `legacy-redirect` → `NNNC_MLX_Lbl02`.
- `NNNC_SW_GBR_Gate` f05 → `NNNC_GBR_f05`.

Manifests list key, kind, provenance (plan, run_id, params sha, fold, seed, window), users and
weight hashes. `saved_data/models/` holds 79 readable links, e.g.
`lstm_gbbpath_g0.02_h48__17782b9951401484 -> ../NNNC_WF_f05`.

## Tests

`pytest Framework utils/tests NNNC/tests evaluation SimpleTrigger/tests research/exit_coupling/tests`:
1023 passed. 4 failed, and all 4 fail identically at the baseline commit:
- the 3 training-pipeline goldens;
- `test_base_nn_strategy.py::TestAugmentTrainingSignals::test_buy_signals_extended_backwards`.

## Deviations from tasks.md

1. **T018:** the legacy-redirect register was written after T024, once the mixin users were gone,
   so the register test compares like with like. It records 96 classes; the plan estimated about 70.
2. **T021a:** arm D only (see above).
3. **T022:** the `.safetensors` addition to `BaseNNStrategy.model_exists`'s fallback was NOT made.
   It would flip `training_needed` before the classifier exists, a behaviour change with no
   benefit, because `maybe_train` re-checks through the classifier.
4. **R11 classifier identity:** the key uses the `classifier_type` attribute plus the class
   defining `get_classifier`/`get_classifier_type`, not a factory lookup. This is the same
   information, without importing classifier factories.
5. **`MODEL_TRAINED_WITH` pins are not hashed.** Hashing them would stop a pinned arm resolving to
   the trained model. The data model was amended.
6. **T026:** `test_strat.sh -f`'s path resolution was smoke-tested (fold, redirect and unpinned
   classes). A full `-f` run was not made, because it trains.
7. **`evaluation/pick_threshold.py`** still checks for a model by class-named directory. It is a
   separate tool, not the harness FR-017 names, and its fold classes have class-named directories
   (migrated). Noted as a follow-up.
