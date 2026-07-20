# GAN pipeline — TODO / open threads

Living list of GAN-pipeline work. Context captured 2026-07-19/20 while debugging a
train/predict scaling mismatch. Previous v1→v2 migration plan:
`docs/v1_to_v2_gan_transition_prompt.md`.

---

## 1. Finish the GAN / no-GAN study — DONE (2026-07-20): non-GAN wins

Seed-robust paired A/B on the FIXED pipeline — GAN (`NNNC_DDPM_MLX`) vs non-GAN
(plain `NNNC_MLX`), identical LSTM + threshold, seeds {1,7,13}, pinned window, both
retrained on fresh scalers.

**Result: non-GAN wins.** profit GAN {12.89, 11.90, 11.97} (mean 12.25) vs non-GAN
{12.60, 13.53, 13.53} (mean 13.22); GAN−nonGAN = +0.29 / −1.63 / −1.56 (non-GAN wins
2/3, decisively at seeds 7/13). **Decoupling:** GAN val_mcc HIGHER at all 3 seeds
(~0.612 vs ~0.601) — augmentation improves classification but WORSENS P&L. Same
learnability≠edge wall as the P&L-weighted-loss probe.

**Conclusion: "no-GAN wins" survives on the correctly-scaled pipeline** (now
trustworthy, not a scaling artifact). The scaling fix (#3) was still worth it — it
took the GAN 10.64%→12.25% and narrowed the gap from ~2.6pp to ~1pp — but didn't flip
the verdict.

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
**Reframed via `v1_to_v2_gan_transition_prompt.md` (2026-07-20):** v2/post_gan_scaling
was a deliberate fix for a GAN VARIANCE bug (v1 MinMax[-1,1]+Tanh capped
σ_syn/σ_real≈0.7; v2 = internal z-score + LINEAR output → σ≈1.0). Crucial: that
variance fix lives in the GAN ARCHITECTURE (internal z-score + linear), NOT in the
tensor-level scaling. So the two are separable — this fix KEEPS the v2 GAN
architecture and only swaps the pipeline scaling tensor→dataframe. It's a refinement
of v2, not a revert.

**FULLY SPECED (2026-07-20, user-confirmed design). DO NOT feed the GAN scaled data
— it consumes RAW and self-z-scores; scaling it would break it + need a GAN retrain.**

Intended pipeline:
- **GAN:** raw (`clean_for_tensor`) in → raw synth out (internal z-score). UNCHANGED.
- **Classifier (train AND predict):** NORMALISED data with column processing —
  RobustScale `needs_norm` columns, PASS THROUGH `pre_normalized_columns` (exactly
  what `scale_dataframe`/`rolling_dataframe_normalise` does).
- **Order:** normalise AFTER the GAN, on the combined raw real+synth.

Three confirmed inconsistencies in the current MT path (all vs the above):
1. Real data is fed to the GAN PRE-SCALED — `prepare_training_data` runs
   `scale_dataframe` before the MT aug, but the GAN was trained on RAW
   (`CreateMTGANBase:67` uses `clean_for_tensor`). Classifier aug
   (`TrainingEngine.preprocess_training_data`, `_invoke_balance_multi_task`) feeds
   the GAN the `scale_dataframe`'d `tsr_train`.
2. Post-GAN `main_tensor_scaler` is a RobustScaler over ALL columns → steamrolls
   `pre_normalized_columns` (the NNMT<NNNC mechanism).
3. Classifier trains on `scale_dataframe→GAN→main_tensor_scaler` but predicts on
   `raw→main_tensor_scaler` (`BaseNNStrategy:994`) → train/predict mismatch.

**Fix (chosen approach A — column-aware tensor scaler, no df round-trip):**
- Feed the GAN RAW at MT aug time (`clean_for_tensor`), matching how the GAN was
  trained. → touch `prepare_training_data`/`preprocess` GATED ON MULTI-TASK (single-
  task is already correct on `scale_dataframe` after d156c11 — DO NOT break it).
- Make the post-GAN scaler (`main_tensor_scaler`/`FeatureScaler`) COLUMN-AWARE: fit +
  apply only on `needs_norm` columns, pass `pre_normalized` through (per-feature
  scaling is invariant to windowing, so this equals `scale_dataframe`). Requires
  `FeatureScaler` to know the pre_normalized column indices; `CreateScalers:65` fits
  it on `needs_norm` only. Apply post-GAN in preprocess AND at predict.
- **No GAN retrain** (GAN stays raw). Validate: `CreateScalers` (re-fit column-aware)
  + retrain NNMT; check σ_syn/σ_real ≈1.0 (unchanged) AND NNMT gap to NNNC closes.

**Diagnostic sub-step option:** fix #1 (train uses raw, matching predict+GAN) ALONE
first — if it closes the NNMT gap, the mismatch was the killer and #2 (column-aware
scaler) may be unnecessary. Cheap: just retrain NNMT, no scaler/GAN change.

Complexity: ~5 coordinated spots across shared single/multi-task code + a rebuild.
Execute as a focused effort with per-change verification; don't rush.

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

## 4. If GANs are not helping — investigate why  (#1 confirmed they don't)

**Partial answer from #1:** even correctly scaled, the GAN improves val_mcc but
worsens P&L (learnability≠edge). Synth generated from the same OHLCV distribution
fits the classification objective better but doesn't add tradeable information — the
information-ceiling wall ([[feedback_triple_barrier_is_a_data_problem]]). So a GAN on
the same features is structurally unlikely to help P&L, regardless of quality knobs.

Remaining angles ONLY if you want to squeeze it (low prior): synth fidelity /
regression-to-mean on heavy-tailed features (`gan_passthrough_columns`),
quality-filter thresholds (AE/density/discriminator), `gan_target_ratio`. But the #1
result suggests the ceiling is the target/features, not the GAN — new information
(order flow / funding / cross-asset) is the only lever that moves it, not synth.

Note: still worth doing #2 (NNMT scaling fix) — that's about NNMT not being crippled
by the tensor-scaler, independent of whether GAN aug helps.
