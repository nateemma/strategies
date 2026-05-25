# v1 → v2 GAN Transition Prompt

Use this as a self-contained instruction set to finish migrating all remaining GAN backends and their strategies from the v1 (pre-normalized) pipeline to the v2 (post-GAN scaling) pipeline. The MT_WGAN MLX backend has already been migrated and is the reference implementation — point of truth for the patterns described here.

## Current State (as of 2026-05-21, MIGRATION COMPLETE)

The v1→v2 transition is complete across every backend that isn't CTAB-GAN
(which intentionally uses its own mode-specific normalization). Strategy-side
flags and Create-class flags are set. Backend internals (Tanh removal,
internal z-score, linear output) are applied to all MLX and TF WGAN/CGAN
backends. See the status tables below for per-file detail.

### Framework plumbing — DONE
- `Framework/BaseNNStrategy.py:155` defines `use_post_gan_scaling: bool = False` default
- `BaseNNStrategy.py:1933` (training-time) and `:2059` (prediction-time) branch on the flag and apply tensor-level scaling via the saved `main_tensor_scaler` instead of dataframe scaling
- `Framework/CreateScalers.py` already trains BOTH `main_scaler` (dataframe) and `main_tensor_scaler` (tensor) so the v2 path's scaler exists
- `GANs/paths.py:gan_save_path()` accepts `post_gan_scaling=` kwarg, routes to `saved_data/GANs_PostScale/<type>/` when True
- `Framework/TrainingConfig.py` — single source of truth for `TRAINING_TYPE`, `MIN_BUY_GAIN_THRESHOLD`, `MIN_SELL_LOSS_THRESHOLD`. All Create + Strategy classes read from this.

### Strategy-side migration status

| Strategy | gan_type | `use_post_gan_scaling` | Status |
|---|---|---|---|
| **NNMT_WGAN** (and `_MLX`, `_MLX_MultiLSTM`) | MT_WGAN | True | ✓ Migrated |
| **NNMT_DDPM** (and `_MLX`, `_MLX_MultiLSTM`) | MT_DDPM | True | ✓ Migrated |
| **NNNC_DDPM_MLX_LSTM_MT_v2** | MT_DDPM | True | ✓ Migrated (v2 variant) |
| **NNNC_DDPM_MLX_LSTM_MT** | MT_DDPM | True | ✓ Migrated 2026-05-21 |
| **NNNC_DDPM_MLX_LSTM** | TAB_DDPM | True | ✓ Migrated 2026-05-21 |
| **NNNC_WGAN** | WGAN | True | ✓ Migrated 2026-05-21 |
| `NNMT_CGP` | MT_CTAB_GAN | n/a | Skip — CTAB-GAN handles own normalization |
| `NNNC_CGP`, `NNNC_CGP_MLX` | CTAB_GAN | n/a | Skip — same reason |

### Create-side migration status

| Create class | gan_type | `use_post_gan_scaling` | Status |
|---|---|---|---|
| `CreateMTWGAN` | MT_WGAN | True | ✓ |
| `CreateMTDDPM` | MT_DDPM | True | ✓ |
| `CreateMTDDPM_v2` | MT_DDPM | True | ✓ |
| `CreateWGAN` | WGAN | True | ✓ Migrated 2026-05-21 |
| `CreateTabDDPM` | TAB_DDPM | True | ✓ Migrated 2026-05-21 |
| `CreateCtabGanPlus`, `CreateMTCtabGanPlus` | CTAB_GAN | n/a | Skip |

### Backend (df_*.py) v2 architecture status

| Backend | Architecture | Status |
|---|---|---|
| `df_mt_wgan_mlx.py` | conv1d + linear output + internal z-score + dropout | ✓ Reference implementation |
| `df_mt_ddpm_mlx.py` | conv1d backbone, EDM schedule, internal z-score | ✓ Naturally v2 + retrained 2026-05-21 |
| `df_tabddpm_mlx.py` | EDM schedule, internal z-score | ✓ Naturally v2 |
| `df_wgan_mlx.py` (single-task MLX WGAN) | linear output, internal z-score, persisted in metadata | ✓ Migrated 2026-05-21 |
| `df_wgan_gp.py` (TF single-task WGAN) | linear output, forward z-score on input, `_postprocess` inverts at generate | ✓ Migrated 2026-05-21 |
| `df_mt_wgan_gp.py` (TF MT WGAN) | Same pattern | ✓ Migrated 2026-05-21 |
| `df_cgan.py` (TF CGAN) | Linear output (Tanh dropped from `_postprocess_gen`), forward z-score on input | ✓ Migrated 2026-05-21 |
| `df_ctab_gan*.py`, `df_mt_ctab_gan*.py` | Mode-specific normalization (architectural — Tanh on continuous outputs is correct) | Skip |

### Open decisions resolved

- **TF backends**: migrate, don't delete (open-source surface for non-MLX users)
- **TrainingConfig**: refactored into `Framework/TrainingConfig.py` — single source of truth
- **EDM schedule** for MT_DDPM: turned on (2026-05-21) after bb_width lag-1 autocorr collapse with cosine-β
- **TRAINING_TYPE** default: changed from 19 → 17 (gbb) based on `DebugSignalLearnability` cross-pair analysis
- **MIN_*_THRESHOLD** default: changed from 0.008 → 0.003 based on same analysis

### Remaining work

**All scoped backends are now migrated.** What's left is operational, not code-side:

1. **Retrain** each GAN backend after the migration to produce v2-format saved models. Existing `saved_data/GANs/<type>/` models are stale: the strategies now look in `saved_data/GANs_PostScale/<type>/`. Trainable backends and their canonical Create classes:
   - `CreateMTWGAN` → `saved_data/GANs_PostScale/mt_wgan/`
   - `CreateMTDDPM` → `saved_data/GANs_PostScale/mt_ddpm/`
   - `CreateTabDDPM` → `saved_data/GANs_PostScale/tab_ddpm/`
   - `CreateWGAN` → `saved_data/GANs_PostScale/wgan/`

2. **Validate** each strategy loads from the new path. First end-to-end run of each strategy will surface a load error if the GAN wasn't retrained yet.

3. **Clean up legacy v1 model directories** once every backend has a verified v2 model:
   - `rm -rf saved_data/GANs/` after all backends are validated
   - Optionally simplify `GANs/paths.py`: remove `GAN_PARENT_DIR` and the legacy branch in `gan_save_subdir` once nothing references it.

4. **TF backends caveat:** all three TF migrations (`df_wgan_gp.py`, `df_mt_wgan_gp.py`, `df_cgan.py`) preserved their existing plateau/best-model checkpoint infrastructure. The training-time `_postprocess` calls were removed; the function itself remains as the generate-time inverse z-score. If an issue surfaces, check the gradient-penalty path — the math should still hold (GP operates on real and fake in the same space, now z-scored space) but it's worth a sanity check on first training run.

## Reference Implementation

Already done (verify these files before applying the pattern elsewhere):

- `GANs/df_mt_wgan_mlx.py` — Tanh removed (linear generator output), conv1d backbone via `_Conv1dResBlock`, hyperparameter kwargs plumbed (`hidden_dim`, `d_layers`, `dropout`, `n_critic`), defaults tuned for WGAN's 6x-per-step backbone cost (`hidden_dim=128`, `d_layers=1`, `n_critic=3`), internal z-score with `_ZSCORE_CLIP=4.0`.
- `GANs/CreateMTGAN.py` — `_run_mt_simple_training` gained `_post_training_fidelity_report` gated by `gan_run_diagnostics` class attribute.
- `GANs/CreateMTWGAN.py` — `use_post_gan_scaling = True`, `gan_run_diagnostics = True`.
- `NNMT/NNMT_WGAN.py` — `use_post_gan_scaling = True`, passes `post_gan_scaling=` kwarg into `gan_save_path()`.
- `NNMT/BaseNNMTStrategy.py` — second `gan_save_path()` call site also passes `post_gan_scaling=`.

Read these files as the canonical pattern before editing anything else.

## Scope

### Apply the full migration to:

1. **`GANs/df_wgan_mlx.py`** (MLX single-task WGAN). Currently OK under v1 because input is MinMax-scaled to [-1, 1]. **Becomes broken under v2** — Tanh-bounded output stops matching the input range. Must be migrated.
2. **`GANs/df_wgan_gp.py`** (Keras WGAN-GP, single-task) — has the Tanh+`x*std+mean` bug. Migrate to v2 even though it's also drifted behind its MLX counterpart (memory note `project_tf_mlx_gan_parity.md`). This is open-source code; keep TF backends working for users without Apple Silicon.
3. **`GANs/df_mt_wgan_gp.py`** (TF MT WGAN-GP) — same bug, same migration.
4. **`GANs/df_cgan.py`** (Keras CGAN) — same bug, same migration.

For TF backends, the conceptual pattern is identical to the MLX reference, but the implementation uses Keras layers / TensorFlow ops:
- Internal z-score: use `tf.constant` for mean/std, broadcast in the model's `_postprocess`/equivalent
- Linear output: remove `activation="tanh"` from the final `Dense`/`Conv1D` layer; drop the `_postprocess` `x*std + mean` rescale that was compensating for the Tanh bound (the inverse z-score replaces it)
- Conv1d backbone: use `tf.keras.layers.Conv1D` with `kernel_size=3, padding="same"` and residual connections built via the functional API
- LayerNorm: `tf.keras.layers.LayerNormalization`
- The hyperparameter kwargs (`hidden_dim`, `d_layers`, `dropout`) plumb through the same way

Existing TF backends already have plateau detection and best-model checkpointing (see `df_wgan_gp.py:1788` and `df_cgan.py:574-649`) — preserve that infrastructure; don't replace it with the simpler MLX pattern. The MLX side may eventually adopt these — see the "Open Questions" section.

### Out of scope (do NOT touch):

- `GANs/df_ctab_gan*.py` and `GANs/df_mt_ctab_gan*.py` (4 files) — CTAB-GAN's mode-specific normalization is part of the architecture; Tanh is correct under both pipelines.
- `GANs/df_mt_ddpm_mlx.py`, `GANs/df_tabddpm_mlx.py`, `GANs/diffusion_mlx.py`, `GANs/diffusion_edm_mlx.py` (4 files) — diffusion models predict noise (linear output by design), already z-score internally. They're pipeline-agnostic.

## Per-Backend Checklist

For each MLX backend in scope:

### 1. Architecture changes

- [ ] Generator: **remove Tanh** at output. Replace with linear (no activation) in both the seq path and the tabular path. Update docstring to record why: WGAN-GP enforces Lipschitz via gradient penalty on the critic — the generator does not need a bounded output.
- [ ] Generator: if `seq_len > 1`, use the **conv1d backbone** pattern from `df_mt_wgan_mlx.py` (lines 75-110-ish):
  ```python
  self.seed = nn.Linear(latent_dim + cond, seq_len * hidden_dim)
  self.blocks = [_Conv1dResBlock(hidden_dim, dropout) for _ in range(d_layers)]
  self.head = nn.Conv1d(hidden_dim, num_features, kernel_size=1)
  ```
  Import `_Conv1dResBlock` from `df_mt_ddpm_mlx.py` (already done for MT_WGAN — line 11 of `df_mt_wgan_mlx.py`).
- [ ] Critic: same — `1×1 Conv1d` for per-timestep in-projection, broadcast condition emb across T, N × `_Conv1dResBlock`, mean-pool over T, then linear heads. Mirror the structure in `MTCriticMLX` (`df_mt_wgan_mlx.py:126-208-ish`).
- [ ] Add architecture kwargs to the model `__init__`: `hidden_dim`, `d_layers`, `dropout`, `n_critic` (WGAN only) with sensible defaults. Mirror the values that worked for MT_WGAN: `hidden_dim=128`, `d_layers=1`, `dropout=0.0`, `n_critic=3`.
- [ ] Plumb the kwargs through the `balance_with_*` outer function so they're tunable from the strategy via `gan_kwargs`.

### 2. Internal z-score handling (v2 pattern)

- [ ] Add `_ZSCORE_CLIP: float = 4.0` class constant on the model.
- [ ] Compute `feature_mean`, `feature_std` from raw input at fit time. For 3D (B, T, F) data, mean/std across the (B, T) axes per feature.
- [ ] Apply forward z-score before training: `(x - mean) / std`, then `clip(x, -CLIP, +CLIP)`.
- [ ] In `_postprocess` (or wherever generation output is unscaled), apply the inverse: `x * std + mean`. **Drop the old `tanh-space` postprocess** — the generator output is already in z-scored space directly.
- [ ] Persist `feature_mean` and `feature_std` in the model metadata for load-time recovery.

Reference: `df_mt_wgan_mlx.py:439-459` for the forward z-score block, `_postprocess` method for inverse.

### 3. Dead-code cleanup

- [ ] Remove any unused generator/critic classes that no longer dispatch (e.g., the old `MTGenerator` we deleted from `df_mt_wgan_mlx.py`). Check for misleading "CNN-based" comments on dead MLP classes.
- [ ] Remove any commented-out v1 scaling logic.

### 4. Post-training fidelity diagnostic

In the corresponding `CreateXxx` class (e.g., `CreateWGAN`, `CreateCGAN`):

- [ ] Add `gan_run_diagnostics = True` class attribute.
- [ ] In the `_run_*_training` (or equivalent) method, after `interface.fit + save`, call a `_post_training_fidelity_report` helper that:
  - Samples ~50K rows from the training data.
  - Calls `interface.generate(n=50_000, task_labels=real_label_sample)` (or the single-task equivalent).
  - Wraps the call in try/except so diagnostic failures don't break training.
  - Calls `summarize_real_vs_synthetic(real_data, real_labels, synth_data, synth_labels, log=print)` from `GANs/diagnostics.py`.

Reference: `CreateMTGAN.py:_run_mt_simple_training` and `_post_training_fidelity_report` (~50 lines combined).

### 5. v2 path wiring

For the corresponding **strategy** class that consumes this GAN:

- [ ] Add `use_post_gan_scaling = True` as a class attribute.
- [ ] In any local `preprocess_training_data` or other method that calls `gan_save_path()`, pass the kwarg:
  ```python
  save_path = gan_save_path(
      self.get_storage_location(),
      self.gan_type,
      use_pca=bool(getattr(self, "use_pca_reduction", False)),
      post_gan_scaling=bool(getattr(self, "use_post_gan_scaling", False)),
  )
  ```
- [ ] Verify any custom `_balance_iteratively` or similar in the inheritance chain ALSO passes the kwarg. There were two call sites for MT_WGAN — one in `NNMT_WGAN.py` and one in `BaseNNMTStrategy.py`. Grep for `gan_save_path(` in your inheritance chain to find them all.

For the corresponding **Create** class:

- [ ] Add `use_post_gan_scaling = True` as a class attribute. The flag matters on both sides — strategy reads, Create writes; if they disagree, the load won't find a model.

### 6. Verification

For each migrated backend, run:

- [ ] Backend-specific test slice. For MLX:
  ```bash
  source .venv/bin/activate
  python -m pytest user_data/strategies/GANs/tests/test_mlx_suite.py \
                   user_data/strategies/GANs/tests/test_functional_suite.py \
                   user_data/strategies/GANs/tests/test_gan_output_contracts.py \
                   user_data/strategies/GANs/tests/test_gan_metadata_roundtrip.py \
                   -q -k "<gan_name>"
  ```
  Replace `<gan_name>` with the GAN family substring (e.g., `wgan and not mt_wgan` for single-task WGAN).

- [ ] Framework regression:
  ```bash
  python -m pytest user_data/strategies/Framework/test_base_nn_strategy.py -q
  ```

- [ ] End-to-end: run `CreateXxx` to train a v2 model. Verify:
  - The new model lands in `saved_data/GANs_PostScale/<gan_name>/`, not `saved_data/GANs/<gan_name>/`.
  - The fidelity diagnostic prints after training (σ_syn/σ_real, joint correlation, lag-1 autocorr).
  - Per-feature `σ_syn/σ_real` should now reach ~1.0 (was ~0.7 with the Tanh-bound bug). Mean shifts should also be smaller.

- [ ] Run the consuming strategy to confirm it loads from `GANs_PostScale/` and augmentation succeeds.

### 7. v1 saved model cleanup (after v2 validation)

Once a backend's v2 model is producing the expected diagnostic profile and a strategy run succeeds end-to-end:

- [ ] Delete the corresponding v1 model directory: `rm -rf user_data/strategies/saved_data/GANs/<gan_name>/`
- [ ] Keep `saved_data/GANs/` itself if other un-migrated backends still live under it.

When ALL backends have been migrated and validated:

- [ ] Delete the entire `saved_data/GANs/` parent directory.
- [ ] Consider also removing the `GAN_PARENT_DIR` constant and the legacy branch in `GANs/paths.py:gan_save_subdir` — the function becomes simpler if only `GAN_POST_SCALE_PARENT_DIR` and `GAN_PCA_PARENT_DIR` remain.

## Open Questions to Address

These were deferred during the MT_WGAN migration. Decide before or during this transition:

### Best-model selection / LR plateau for WGAN

MT_WGAN MLX still trains for N epochs at constant LR with no best-model checkpoint and no plateau detection. Other MLX GANs (TabDDPM, CTAB-GAN MLX, MT_CTAB_GAN MLX) have one or both. WGAN loss is not "lower is better" (Wasserstein distance, signed) so best-model selection is non-trivial. Options:

1. Match the others mechanically (ReduceLROnPlateau on D loss, save min-loss model). Wrong metric for WGAN but consistent with the codebase.
2. WGAN-appropriate: track moving average of `|D_loss + G_loss|` (Wasserstein-gap proxy), use that for both plateau and best-model. Marginally more correct.
3. Divergence guard only — no best-model; early-stop if `|G_loss|` exceeds threshold or grows unbounded. Saves final model. Simplest. Matches the saved memory note `feedback_framework_handles_training_callbacks.md` that says GAN `fit()` should train for N epochs at constant LR.

Ask the user which they want before implementing. Don't pick silently.

### TF parity decision

**Decided:** migrate, don't delete. The TF backends remain part of the public surface for users on non-Apple-Silicon hardware. Apply the same v2 pattern as the MLX backends.

## Final Sanity Checks

- [ ] After all backends are migrated, all tests pass: `python -m pytest user_data/strategies/ -q` (acknowledge that full GAN suite may segfault on Metal context — slice as needed).
- [ ] `grep -rn "use_post_gan_scaling = False" user_data/strategies/` should return only `BaseNNStrategy.py:151` (the default for non-migrated/legacy code) and the `getattr(..., False)` reads. No production strategy should explicitly set it to False.
- [ ] `grep -rn "mx.tanh\|nn.Tanh" user_data/strategies/GANs/` should match only CTAB-GAN variants (legitimate use).
- [ ] `grep -rn "saved_data/GANs/" user_data/strategies/` should match only test fixtures, paths.py defaults, and migration scripts — no production code paths.
