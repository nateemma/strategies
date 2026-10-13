# Implementation Plan: Model Naming by Model Fingerprint

**Branch**: `030-model-fingerprint-naming` (spec directory; commits go to master) | **Date**:
2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/030-model-fingerprint-naming/spec.md`

## Summary

Today `BaseNNStrategy.get_model_path()` returns `saved_data/<Class>/<Class>.keras`. This feature
makes it return the location of a **model key**: a hash of everything that shapes the weights,
resolved by a generic `Framework/ModelKey.py`. The key is built from:
- declared attribute lists (Framework generic, plus each family's extension);
- the defining class of each training-path method;
- file identities by content (scalers, PCA, GAN generators);
- run inputs: training window, whitelist, timeframe, warm-up, seed;
- a manual model-code version.

Existing models are **indexed, never moved or retrained**:
- the 79 walk-forward fold models get keys reconstructed from their plans;
- the 58 single-model directories get per-class legacy keys (Q3 → A);
- the existing class-level path overrides are kept as recorded legacy redirects.

`BorrowedModelMixin` users drop the mixin and resolve to the source fold's key. The gate arms
declare `MODEL_TRAINED_WITH = {"startup_candle_count": 64}`, because warm-up turned out to be a
training input (research R2).

Walk-forward evaluation runs strict: a missing model halts. Other runs may train, but only into
the resolved key, writing a manifest (Q2 → B).

Verification follows 029's pattern:
- a same-day baseline before any change, and the same arms after it;
- byte-identical trades and weight files;
- unchanged 029 fingerprints;
- zero training.

## Technical Context

**Language/Version**: Python 3.12 (repo `.venv`); freqtrade backtesting via `regime/run_arm.py` /
`evaluation/run_folds.py`.

**Primary Dependencies**:
- `Framework/BaseNNStrategy.py` (`get_model_path`, `model_exists`), `Framework/TrainingEngine.py`
  (`maybe_train`), `Framework/FeatureNormalizer.py`;
- classifier bases in `Predictors/`;
- families: `NNNC/NNNCStrategy.py`, `NNMT/NNMTStrategy.py`, `NNPredict/NNPredictStrategy.py`,
  `NNWavelet/NNWaveletStrategy.py`, `Anomaly/NNAnomalyStrategy.py`, `Sklearn/SklearnStrategy.py`;
- `NNNC/BorrowedModelMixin.py` and its users;
- `evaluation/run_folds.py` (TEMPLATE, `model_exists`, `model_key`, `train_fold`,
  `evaluate_fold`), `evaluation/foldplan.py`;
- `scripts/test_strat.sh`.

**Storage**:
- `saved_data/` (gitignored): `model_index.json`, `models/<prefix>__<hash>/`, and a
  `manifest.json` beside each indexed model;
- committed audit copies in the spec directory: the index snapshot, `legacy_redirects.json`,
  baseline/after JSONs and weight hashes.

**Testing**:
- pytest across `Framework/`, `utils/tests`, `NNNC/tests`, `evaluation/`, `SimpleTrigger/tests`;
- new tests: `Framework/tests/test_model_key.py` (determinism, sensitivity, ignore-list
  invariance, file-content keying, strict mode, manifest mismatch), `test_model_key_audit.py`
  (R5 classification, legacy-redirect register), and `NNNC/tests/test_model_key_nnnc.py`
  (borrowed users resolve to their source keys).

**Target Platform**: macOS / Apple Silicon (MLX).

**Project Type**: a strategy research repository; library code plus a CLI.

**Performance Goals**: resolving a key costs under 1 s per process (file hashes are cached), and
backtest time is unchanged.

**Constraints**:
- no behaviour change, and no training during migration or verification;
- weight bytes stay unchanged;
- `Framework/` stays generic;
- parameter dicts merge;
- one backtest chain at a time;
- the live bot is untouched.

**Scale/Scope**:
- about 500 BaseNNStrategy classes across 6 families;
- 144 `saved_data` entries;
- 79 fold models and 58 single models;
- about 70 legacy-redirect classes;
- 246 borrowed fold classes, generated from 16 base classes.

## Constitution Check

| principle / constraint | status |
|---|---|
| I–III, VIII, IX (evaluation principles) | Not engaged: no edge claim, no tuning. |
| IV Verify parameters are live | Engaged. Strict mode has a liveness check: a changed-input class must halt (quickstart 4). The key's sensitivity is tested per input (SC-003). |
| V Lookahead / realism | Unchanged. The new leakage warning for unpinned models whose training range overlaps the run is a small gain. |
| VI Diagnose before tuning | The surveys (research R1–R10) came before the design. |
| VII Validate cheaply | Migration has `--dry-run`, and the index is built before any loader change is switched on. |
| X No per-pair code | Respected. |
| XI No leakage | The key records the training window per model. Unpinned models print their training range. |
| Repository boundary | Only `user_data/strategies` changes; freqtrade itself is not touched. |
| Framework strategy-agnostic | `Framework/ModelKey.py` reads only declared lists. Family lists are declared in the family base classes. **PASS** |
| Parameter dicts merge | `MODEL_KEY_*` tuples merge along the MRO (union), never replace. **PASS** |
| Quality gate 7 (scalers regenerated) | Strengthened: a regenerated scaler rekeys every model that uses it. |
| Staging / commit to master | Followed. |

**Spec amendment needed (flagged for analyze).** FR-018a asks for a representative of each other
family *with existing weights*. NNPredict, NNWavelet, Anomaly and Sklearn have none (research R9),
so for them verification is key resolution plus a strict-mode halt. NNMT is verified by trades.

## Project Structure

### Documentation (this feature)

```text
specs/030-model-fingerprint-naming/
├── spec.md, plan.md, research.md, data-model.md, quickstart.md
├── contracts/resolver.md
├── baseline.md / baseline.json         (phase 1)
├── weights_before.json                 (sha256 of every weight file)
├── fingerprints_before.json / _after.json
├── legacy_redirects.json
├── model_index_migrated.json           (index snapshot)
├── verification.md, report.md
└── tasks.md                            (/speckit-tasks)
```

### Source Code

```text
Framework/
├── ModelKey.py                 NEW  resolve / locate / manifest / index / strict mode (generic)
├── BaseNNStrategy.py           get_model_path -> ModelKey.locate; MODEL_KEY_* generic lists;
│                               model_exists also checks .safetensors (fallback path)
├── TrainingEngine.py           record_training after a successful train
└── tests/test_model_key.py, tests/test_model_key_audit.py     NEW
NNNC/NNNCStrategy.py            family MODEL_KEY_* extension
NNMT/NNMTStrategy.py, NNPredict/NNPredictStrategy.py, NNWavelet/NNWaveletStrategy.py,
Anomaly/NNAnomalyStrategy.py, Sklearn/SklearnStrategy.py      family extensions
NNNC/BorrowedModelMixin.py      deprecation shim (raises if used)
NNNC/NNNC_TriggerGate.py, NNNC_SIProbe, NNNC_Live_*.py, NNNC_SW_*.py   drop the mixin;
                                gate arms gain MODEL_TRAINED_WITH
evaluation/run_folds.py         TEMPLATE writes MODEL_TRAIN_TIMERANGE / MODEL_SEED;
                                model_exists via key; evaluate_fold sets FT_MODEL_EVAL_STRICT=1
evaluation/foldplan.py          model_dir kept informational only
scripts/model_index.py          NEW  key / path / migrate / audit / verify-bytes
scripts/test_strat.sh           -f deletes the resolved path
NNNC/tests/test_model_key_nnnc.py   NEW
```

**Structure Decision**: the resolver goes in `Framework/` because it is generic. Every
family-specific input is a declaration in that family's base class.

## Phases

1. **Baseline**, on unchanged code, same day:
   - arm D 15 folds;
   - FMR control (GBR, GFM and MLX were run today as `sw_*_ctl`; if the day changes, re-run them);
   - twin `wf_trgate`; `sw_aro_gate` and `sw_mlx_c2`;
   - NNMT_MLX_Base and NNMT_Ctl_s42 on W1;
   - weight sha256 snapshot, fingerprint scan, and the AST list of path overrides.
2. **Resolver and declarations:**
   - `ModelKey.py`, the generic and family lists, and the audit tests;
   - **no loader change yet**: `get_model_path` is untouched.
3. **Migration:**
   - `migrate --dry-run`, then `migrate`;
   - snapshot committed;
   - conflicts gate the next phase (any conflict → stop and report);
   - borrowed users' keys checked equal to their sources' keys, offline.
4. **Switch the loader:**
   - `get_model_path` → `locate`, plus strict mode, the `run_folds` template pins, the
     `test_strat.sh` change, and the mixin removal.
5. **Verify:**
   - the phase-1 arms re-run: trades 15/15 (and W1 for NNMT);
   - weight bytes equal, fingerprints equal, 0 × training;
   - strict-mode halt checks;
   - weightless-family resolution;
   - tests.
6. **Report and memory.**

## Complexity Tracking

| complexity | why needed | simpler alternative rejected because |
|---|---|---|
| Declared key lists plus an audit test, rather than hashing everything | Cosmetic attributes must not rekey, and a forgotten input must fail CI | Hash-all rekeys on any edit; a plain list decays silently |
| Kept legacy redirects (about 70 classes) | Their weights' training inputs can't be reconstructed (NNMT_MLX trained at 0.007; the class reads 0.02) | Migrating them would assign keys that misdescribe the weights, which is the collision the spec forbids |
| `MODEL_TRAINED_WITH` pins | Warm-up is a training input, yet the gate arms must reuse 64-bar weights | Keeping BorrowedModelMixin defeats FR-016; excluding warm-up from the key is unsafe (R2) |
