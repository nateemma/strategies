# Report: Model Naming by Model Fingerprint (030)

*A no-behaviour-change feature: every result below is "identical" or "refused", not a performance
claim.*

## Summary

NN weights are now found by a **model key**, the hash of everything that shapes them. They are no
longer found by strategy class name.

| criterion | result |
|---|---|
| **SC-001** arm D and the reference NNNC arms reproduce their baseline | **8 arms 15/15**: arm D, FMR/GBR/GFM/MLX controls, twin, gate arm, NNNC_MLX C2 arm. 0 training, 0 weight bytes changed |
| **SC-002** former `BorrowedModelMixin` users load the same weights without it | **18/18 classes × 15 folds** resolve to their source key (offline); the twin, a gate arm and an NNNC_MLX arm reproduce their trades 15/15 |
| **SC-003** every weight-affecting input moves the key; every post-model setting doesn't | 152 key tests; the audit requires every multiply-defined attribute and method to be classified |
| **SC-004** migration indexes the walk-forward sets | 79/79 keyed from their plans, 51 legacy, **0 conflicts** |
| **SC-005** a renamed class finds its weights | tested (`test_a_renamed_copy_of_arm_d_resolves_to_the_same_key`) |
| **SC-006** NNNC fingerprints unchanged | **0 / 511** changed (vs a same-code re-scan; see verification.md) |
| **SC-007** NNMT identical; weightless families resolve; NNMT overrides recorded | NNMT_MLX_Base and NNMT_Ctl_s42 W1 identical; NNPredict/NNWavelet/Anomaly all resolve; 96 legacy redirects registered. Sklearn cannot be imported, and could not before 030 either |
| **SC-008** strict walk-forward refuses a missing model | 4/4 strict runs halt with `ModelMissing` naming the key; no directory created |

Commits:

| commit | step |
|---|---|
| `a62fb23` (+ sha `9bf23df`) | baseline |
| `b06f463` | resolver and key declarations (no loader change) |
| `8284207` | migration |
| `8d821b8` | loader switch, mixin retired, strict harness |
| (this commit) | verification and report |

## What changed

- **`Framework/ModelKey.py`** (new, generic). It resolves a key from four things:
  - the declared attributes;
  - which class defines each training-path method;
  - scaler/PCA/GAN file contents;
  - run inputs: training window, whitelist, timeframe, startup candles, seed.

  It also handles the index (`saved_data/model_index.json`), manifests, strict mode, the training
  lock and report mode.
- **Per-class declarations.** `MODEL_KEY_ATTRIBUTES/IGNORED/METHODS/METHODS_IGNORED` are declared
  in the 26 classes that own the names. These are Framework bases, the entry-framework and
  family-exit mixins in `utils/`, the NNNC mixins, and the family bases. `Framework/` names no
  family concept.
- **`BaseNNStrategy.get_model_path`** resolves the location in this order: the index, then a
  legacy class directory, then the keyed location. Class-level overrides keep their own path.
- **`TrainingEngine`** checks the manifest on load. Training into a new key holds a lock and writes
  a manifest and an index entry.
- **`evaluation/run_folds.py`:**
  - generated fold classes pin `MODEL_TRAIN_TIMERANGE`/`MODEL_SEED`;
  - model checks go by key;
  - evaluation is strict.
- **`BorrowedModelMixin`** is retired as a raising shim. Its 17 users drop it, and the gate arms
  declare `MODEL_TRAINED_WITH = {"startup_candle_count": 64}`.
- **Tooling:**
  - `regime/run_arm.py` records `model_key`;
  - `scripts/test_strat.sh -f` deletes the resolved path;
  - `scripts/model_index.py` provides `key`, `path`, `migrate`, `audit` and `verify-bytes`.
- **`AGENT_GUIDE.md`** has a "Model keys" section.

## Findings along the way

- **Warm-up candles are a training input.** freqtrade runs `populate_indicators`, where training
  happens, *before* it trims startup bars. So a 4096-bar arm on 64-bar weights was silently
  borrowing weights its own run would not have produced. The reuse is now declared and visible.
- **Shared scaler files were an unkeyed input.** A regenerated `main_scaler.pkl` would have been
  served silently to every model trained on the old one. It is now keyed by content.
- **Defaults drifted under existing weights.** For example, `NNMT_MLX` was trained at 0.007 but its
  class now reads 0.02. This is why pre-030 single models are per-class legacy entries and not
  re-keyed from today's attributes.

## Follow-ups

1. `evaluation/pick_threshold.py` still checks models by class-named directory (verification
   deviation 7).
2. The Sklearn family cannot be imported (`utils.ClassifierKeras` missing), which predates 030.
3. **Next feature, agreed 2026-10-12:** study entry, exit and stop features directly on NNNC.
   Variants now share their model's weights by declaration alone.
