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

## 2. NNMT_DDPM/WGAN GAN train-vs-generate scaling mismatch — DONE (path A, P&L-neutral)

**VALIDATED 2026-07-20 (commits bc50253 + 7f16219).** Path A implemented (GAN on raw
everywhere; column-aware post-GAN tensor scaler; predict unchanged). Confirmed in code
that the MT_DDPM **self-scales**: `df_mt_ddpm_mlx.py` z-scores at fit (:356), samples
from noise + clips to ±4σ + de-z-scores to raw at generate (:600-618). So `generate()`
output is ALWAYS raw-scale regardless of input scale → the old
`normalise_for_gan`/`denormalise_from_gan` round-trip left real (MinMax) and synth
(raw) at DIFFERENT scales in the training mix. Path A co-scales them (both raw) then
normalises.

**A/B (old model load vs path-A retrain, same window -n720 -o30, 0-epoch load confirmed):**
pre-fix +3.75% / DD 0.70% / Calmar 9.87 vs path-A +3.98% / DD 0.84% / Calmar 9.19,
both 29 trades. **P&L-neutral (+0.23pp, within noise).** The bug was real but wasn't
costing P&L: the DDPM synth is low-quality (over-dispersed — z-scores saturate the ±4σ
clip → synth σ ~4-5× real, `OFF_DIST` on all 18 buckets in the fidelity report) whether
co-scaled correctly or not, so it adds similar noise either way. Same information-ceiling
pattern as NNNC ([[feedback_gan_ratio_sweep_no_gan_wins]]).

**Three-way confirmation (2026-07-20, same window, plain + path-A are fresh retrains):**
plain NNMT_MLX (no GAN) +3.84% / 35 tr / Calmar 10.72; NNMT_DDPM pre-fix +3.75% / 29 tr;
NNMT_DDPM path-A +3.98% / 29 tr. All within 0.23pp. **BUT this GAN-vs-non-GAN result is
statistically UNDER-POWERED:** the guards are deliberately tight right now (per prior
studies), so all three variants collapse to ~29-35 trades — too few to distinguish GAN
from non-GAN; a real effect could be masked. **To actually evaluate GAN vs non-GAN for
NNMT we probably need to LOOSEN the guards** (more trades → a discriminating test), then
re-run the paired comparison. The low trade count vs NNNC (~167) is that tight-guard +
3-of-4 task-filter regime (a chosen high-quality/low-quantity operating point,
[[feedback_sell_filter_is_capital_pacing]]) — NOT an architecture flaw.

**Two takeaways:** (a) KEEP path A — mechanically correct, removes the bug, makes the
fidelity diagnostic trustworthy; the within-strategy scaling A/B (+3.75 → +3.98) is
guard-independent, so "scaling fix is P&L-neutral" stands regardless. (b) The
GAN-vs-non-GAN verdict for NNMT is OPEN pending a loosened-guard re-run — the current
tight-guard tie is not conclusive. The synth lever, if chased, is DDPM quality (epochs
/ DDIM steps / tighter clip / backbone); #4's prior suggests better synth won't move
P&L, but that too was measured at tight guards. Original plan retained below.

---
### Original path A plan (kept for reference)

**Prior framing (in this doc + `project_ddpm_base_vs_nnnc_mlx_anomaly` memory) was
WRONG** — it described the *single-task + MT-GAN* path
(`TrainingEngine.preprocess_training_data:850`, which applies `main_tensor_scaler`).
The TRUE multi-task classifier (`NNMT_DDPM`/`NNMT_WGAN`) never touches
`main_tensor_scaler`. Traced end-to-end 2026-07-20:

- Classifier **train** (`NNMTStrategy.prepare_training_data:689`) → `scale_dataframe`.
- Classifier **predict** (`NNMTStrategy.get_predictions:918`) → `scale_dataframe`.
  → so the classifier's train/predict are ALREADY consistent + column-aware. No
  `main_tensor_scaler` steamroll, no train/predict mismatch on this path.

**ACTUAL root cause — GAN train-vs-generate input mismatch (half-applied v2):**
- `use_post_gan_scaling=True` (set on `NNMT_DDPM:48`, `NNMT_WGAN`) flips the GAN
  TRAINING side to feed RAW: `CreateMTGANBase:60-67` → `df_ready=clean_for_tensor(df)`
  (GAN self-z-scores internally, v2 linear output).
- But the GENERATE side (`NNMT_DDPM/WGAN.preprocess_training_data:146,198`) still does
  the v1 round-trip: `normalise_for_gan` (MinMax[-1,1] of `scale_dataframe` space) in,
  `denormalise_from_gan` out. So the GAN is QUERIED on a distribution it never saw →
  OOD → corrupted synth. `use_post_gan_scaling=True` there only changes the model
  LOAD path (`gan_save_path`, line 138), NOT the scaling. The comment at `NNMT_DDPM:44`
  describes the intended v2 behaviour the code never implemented.
- Best explanation for the `adx`/`vwap` "6σ wrong-direction mode collapse" noted in
  `NNMT_DDPM:73-83` — likely an artifact of the OOD input, not an inherent GAN limit,
  so `gan_passthrough_columns` may be papering over this (re-evaluate after the fix).

**Path A (user-chosen 2026-07-20): complete v2 — GAN on raw everywhere.**
Intended pipeline: GAN raw in → raw synth out (unchanged, already trained this way);
classifier consumes column-aware NORMALISED data; normalise AFTER the GAN.
- **No GAN retrain** — `CreateMTGANBase` already trains on raw. Only the generate side
  + classifier scaling change to match.
- Equivalence established (so predict need NOT change):
  `column_aware_tensor_scaler(clean_for_tensor(df))` == `scale_dataframe(df)`, because
  `rolling_dataframe_normalise` is a GLOBAL per-feature `RobustScaler` on `needs_norm`
  (skips `pre_normalized`) + `np.clip(±10)` — invariant to windowing. So predict stays
  on `scale_dataframe:918`; only train's post-GAN normalise must use the same op.

**Concrete edits (gated on `use_post_gan_scaling` so single-task/plain-NNMT untouched):**
1. `FeatureScaler` → column-aware: RobustScale `needs_norm` indices, pass
   `pre_normalized` through, clip ±10 (replicates `scale_dataframe` on a tensor).
   Needs the passthrough column indices at fit time.
2. `CreateScalers:63-67` → fit `main_tensor_scaler` column-aware (compute `needs_norm`
   indices from `clean_for_tensor` columns vs `pre_normalized_columns`).
3. `NNMTStrategy.prepare_training_data:689` → for `use_post_gan_scaling`, build the
   tensor from `clean_for_tensor` (RAW) instead of `scale_dataframe`.
4. `NNMT_DDPM` + `NNMT_WGAN.preprocess_training_data` → drop `normalise_for_gan` /
   `denormalise_from_gan`; feed the RAW tensor to `_balance_iteratively`; take raw
   synth out; then apply the column-aware `main_tensor_scaler` to the combined
   real+synth AND to `test_data`. Cover the non-aug / GAN-load-fail branch too (raw
   must still get normalised before the classifier).
- **Validate:** `CreateScalers` re-fit + retrain NNMT_DDPM; check the
  `gan_run_diagnostics` fidelity report (`adx`/`vwap` mode-collapse should shrink) AND
  the NNMT_DDPM-vs-NNNC gap. Then re-test whether `gan_passthrough_columns` is still
  needed.

## 2b. Clean up `use_post_gan_scaling` — make it the ONLY path (AFTER #2)

Once #2 lands and path A is validated, `use_post_gan_scaling=True` (GAN-on-raw +
column-aware tensor normalise) should become the single pipeline. Remove the
`use_post_gan_scaling=False` branches: the `scale_dataframe→normalise_for_gan` GAN
training path (`CreateMTGANBase:68-71`), the flag itself + its `gan_save_path`
branching, and the now-dead v1 `normalise_for_gan`/`denormalise_from_gan` round-trip
in the MT preprocess. Retire `main_scaler`-vs-`main_tensor_scaler` duplication where
they now coincide. Goal: one scaling story, no `getattr(self,"use_post_gan_scaling")`
gates. Do NOT start until #2 is confirmed (keep the fallback until the new path is
proven).

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
