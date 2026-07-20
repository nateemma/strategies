# GAN pipeline — TODO / open threads

Living list of GAN-pipeline work. Context captured 2026-07-19/20 while debugging a
train/predict scaling mismatch. Previous v1→v2 migration plan:
`docs/v1_to_v2_gan_transition_prompt.md`.

---

## 1. Finish the GAN / no-GAN study

**Status: RUNNING.** Seed-robust A/B on the FIXED pipeline — GAN (`NNNC_DDPM_MLX`)
vs non-GAN (plain `NNNC_MLX`), identical LSTM + threshold, differing only in the GAN
aug, seeds {1,7,13}, pinned window `20240629-20260619`, both retrained on
freshly-regenerated scalers. Variants: `NNNC_DDPM_MLX_s{1,7,13}` /
`NNNC_MLX_s{1,7,13}` (via `TrainSeedStrategyMixin`).

Why it matters: every prior GAN-vs-non-GAN comparison (incl. the "no-GAN wins /
augmentation is net-noise" conclusion) was measured on a mis-scaled GAN predict path
(see #3). This is the first trustworthy read.
- GAN ≥ non-GAN across seeds → augmentation was helping; "no-GAN wins" was a
  scaling-bug artifact.
- GAN < non-GAN still → augmentation genuinely doesn't help even correctly scaled.

## 2. Review NNMT dataframe-vs-tensor scaling (likely NNMT-underperformance cause)

**Finding (confirmed in code):** the two scaling paths use the SAME method
(RobustScaler — `FeatureScaler` just wraps RobustScaler for 3-D), but scale
DIFFERENT columns:
- Dataframe path (`rolling_dataframe_normalise`, single-task/NNNC): RobustScales only
  `needs_norm_columns` and LEAVES `pre_normalized_columns` (+ passthrough) in their
  designed range — the tuned normalization.
- Tensor path (`clean_for_tensor` → `main_tensor_scaler=FeatureScaler`,
  multi-task/NNMT): `clean_for_tensor` SKIPS scaling; `FeatureScaler` RobustScales
  ALL columns including `pre_normalized_columns` → **re-scales / discards the
  pre-normalization work.** Plausible mechanism for **NNMT < NNNC**.

**Proposed fix (simple, from the user):** just run `scale_dataframe` on the dataframe
right before `df_to_tensor`, everywhere — retire the tensor-level `FeatureScaler`
step. Rationale: scaling before vs after windowing is equivalent for a per-feature
scaler (median/IQR invariant to window replication), AND `scale_dataframe` respects
`pre_normalized_columns`. **No inverse needed** (that's only for the row-level
`gan_scaler_a` MinMax round-trip). This unifies single+multi on the GOOD dataframe
normalization (the opposite unification from the original v2 intent, but the correct
one). Caveat: the multi-task GAN must then also train/generate in the
dataframe-normalized space (`CreateMTDDPM` on `scale_dataframe`'d data, no tensor
scaler).
**Next:** after #1, wire multi-task to normalize the dataframe up front, retrain an
NNMT variant, check the gap to NNNC closes.

## 3. Review current state of `post_gan_scaling` logic

Previous plan: `docs/v1_to_v2_gan_transition_prompt.md`.

**Intent (user):** `use_post_gan_scaling` was meant to scale the ASSEMBLED TENSOR
(via `main_tensor_scaler`) instead of the dataframe, to make single- and multi-task
pipelines uniform.

**Bugs found + FIXED this session:**
- Predict-time tensor scaling was applied for `use_post_gan_scaling=True` regardless
  of whether TRAINING applied it. `preprocess_training_data` (the tensor-scaler step)
  is MULTI-TASK-ONLY, so single-task `TAB_DDPM` trained in `scale_dataframe`
  (RobustScaler) space but predicted through `main_tensor_scaler` → mismatch.
  **Fix (commit d156c11):** gate predict on `gan_type in _MULTI_TASK_GAN_TYPES`.
  Verified NO-retrain: production DDPM **10.64% → 12.17% (+1.53pp)**.
- Earlier related fix for `gan_type==NONE` (commit 1f686fa, superseded by d156c11).
- `gan_scaler_a` (MinMax) round-trips cleanly (inverse IS applied) — not a bug.

**Still to review:** given #2's proposed direction (dataframe-normalize before
tensorize), reconsider whether `use_post_gan_scaling` / the tensor-scaler path should
exist at all, or be redefined as "normalize dataframe up front." Check the v2
transition doc against current behaviour; the multi-task tensor-scaler path is the
remaining place the pre-normalization is lost.

## 4. If GANs are not helping — investigate why

**Gated on #1.** Only pursue if the fixed-pipeline GAN/no-GAN study (#1) shows GAN ≤
non-GAN. Prior context: [[feedback_gan_ratio_sweep_no_gan_wins]] (no-GAN won a 9-run
sweep) and [[project_ae_filter_win]] (AE-filtered DDPM once beat no-GAN) were both on
the pre-fix pipeline, so re-establish the baseline first. Candidate angles if still
not helping: synth fidelity (regression-to-mean on heavy-tailed features — see
`gan_passthrough_columns`), quality-filter thresholds (AE/density/discriminator),
`gan_target_ratio`, and whether the classifier's information ceiling
([[feedback_triple_barrier_is_a_data_problem]]) leaves any room for synth to help at
all.
