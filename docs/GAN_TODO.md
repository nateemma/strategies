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

## 5. GAN sample-quality improvement plan (staged) — STARTED 2026-07-20

Motivation: MT_DDPM synth is over-dispersed (z-scores saturate the ±4σ clip →
σ_syn ≈ 4-5× σ_real, OFF_DIST on all 18 fidelity buckets). User ideas: (1) stricter
acceptance filters, (2) discriminator/critic rejection, (3) loss that punishes bad
samples. Reframed with a 4th (fix the sampler root cause) and ordered cheapest-first.
**Rules:** measure before build, one lever at a time, P&L-gate every fidelity gain
(fidelity ≠ edge is what we're TESTING, not assuming — [[feedback_triple_barrier_is_a_data_problem]]).
Note MT_DDPM is DIFFUSION (no discriminator) — idea (2) applies to MT_WGAN's critic;
for DDPM a learned rejector = a stronger AE filter (folds into Phase 2).

**Phase 0 — powered, trustworthy baseline (precondition, no model changes).** The
current GAN-vs-non-GAN test is under-powered (~29-35 trades at tight guards). Loosen
guards for trade volume; retrain the DDPM variant + non-GAN control; paired seeds,
pinned window. Output = powered P&L delta (with seed spread) + current fidelity report
(OFF_DIST/σ baseline). DECISIONS: (a) target family — NNMT MT_DDPM (defect diagnosed,
richest fidelity report) vs NNNC TabDDPM (already trade-powered, AE-filter win
[[project_ae_filter_win]]); (b) which guards to loosen (NNMT: apply_task_filters is the
dominant lever, disable → ~4.5× trades [[feedback_sell_filter_is_capital_pacing]]).

**Phase 1 — root-cause sampler tweaks (idea 4), NO GAN retrain.** DDIM steps and
generate-time clip are inference-time (decoupled from training) → re-augment + retrain
classifier only. Grid `num_sample_steps` 50→100/250, generate `_ZSCORE_CLIP` 4→3/2.5,
sampling-noise scale (via `_apply_gan_inference_overrides`). Fidelity gate: σ_syn/σ_real
→~1, OFF_DIST shrinks. Then P&L-gate on Phase-0 A/B. Highest value-per-effort.

**Phase 2 — moment/range acceptance filter (idea 1), NO GAN retrain.** In
`balance_multi_task` accept loop: reject windows whose per-feature CLASS-CONDITIONAL
μ/σ deviate > N MADs from real, or with features pinned at the clip band. Tune N via
fidelity report; compose with the existing AE filter. Fidelity gate → P&L-gate.

**Phase 3 — training-time work (idea 3; idea 2 for WGAN), ONLY if Phase 1-2 P&L moves.**
#3: distribution/moment-matching aux term on DDPM denoising loss, OR tune MT_CTAB_GAN
(already carries this objective) before new loss code. #2: MT_WGAN critic-score accept
threshold. Expensive (GAN retrains) → last + conditional. Watch the P&L-loss prior:
shaping loss raised val_mcc not P&L ([[feedback_pnl_weighted_loss_raises_mcc_not_pnl]]);
a fidelity loss differs but the "metric-not-trade" pattern is the risk.

**Every phase reports fidelity AND powered-A/B P&L.** Two clean outcomes: synth moves
P&L → ceiling broken for this target, Phase 3 justified; or fidelity climbs / P&L flat
across cheap Phases 1-2 → high-confidence cheap reading that the lever is elsewhere.

**Phase 0 DONE (2026-07-20) — target NNNC TabDDPM, loosen via prediction_threshold.**
prediction_threshold is inference-time, so the baseline reuses the existing
NNNC_MLX / NNNC_DDPM_MLX weights (0-epoch, NO retrain) via isolated subclasses
`NNNC_MLX_P0` / `NNNC_DDPM_MLX_P0` (own class → never touch NNNC_MLX.json, which the
hyperopt owns). At **prediction_threshold 0.45** (pinned 20240629-20260619):
- **Power achieved:** ~580 trades (vs ~166 at 0.6, ~3.5×) — discriminating.
- **Baseline A/B (single-seed):** non-GAN +21.24% / 599 tr / DD 4.09% vs GAN +20.72% /
  573 tr / DD 3.84% / Calmar 10.54 → **non-GAN marginally ahead (−0.52pp)**, GAN slightly
  fewer trades. Consistent with all priors (non-GAN ≥ GAN).
- **Fidelity baseline:** deferred to Phase 1's control run (unchanged sampler) — single-
  task `balance_single_task` emits a per-class fidelity report and NNNC_DDPM_MLX has
  `gan_run_diagnostics=True`.
- **Seed spread:** deferred — add paired seeds {1,7,13} to get the noise band WHEN a
  Phase-1/2 variant shows a P&L delta worth testing for significance (−0.52pp is almost
  certainly within noise at this trade count).

**Phase 1 DONE (2026-07-20) — REFRAMED: NNNC TabDDPM synth is ALREADY good; ceiling
confirmed.** The control run's fidelity report was the key diagnostic: single-task
TabDDPM synth is high-fidelity — both signal classes flag `ok`, σ_syn/σ_real ~0.5-1.0,
worst-feature |Δμ| < 0.5σ, AE-filtered (kept ~70% @ thr 0.005). **The over-dispersion
(σ 4-5×, OFF_DIST all buckets, 10σ shifts) was MT_DDPM-SPECIFIC (NNMT), NOT TabDDPM.**
So Phase 1's sampler tweaks aimed at an absent defect. Ran the grid anyway to confirm
(fresh 44-epoch trains, pred_thr 0.45, own class names → production untouched):
| variant | class0/2 σ_ratio | P&L | tr |
| non-GAN | — | 21.24% | 599 |
| control (steps50/clip4) | 0.47/0.55 | 20.48% | 575 |
| clip25 (clip2.5) | 0.40/0.45 | 20.28% | 574 |
| steps250 (250 steps) | 0.48/0.54 | 20.36% | 575 |
clip25 WORSENED the (already slight) under-dispersion as predicted; steps250 left
fidelity unchanged; both P&L flat-to-worse. **All GAN variants ~0.8pp UNDER non-GAN.**
Cleanest information-ceiling evidence yet: high-fidelity, AE-filtered GAN synth at a
POWERED operating point (~580 tr) still adds no edge → quality is NOT the NNNC
bottleneck. Phase 2/3 on NNNC would solve a non-problem. `gan_inference_zscore_clip`
override added to `_apply_gan_inference_overrides` (TrainingEngine).
**Where quality DOES have traction: NNMT MT_DDPM (bad synth). To test "does fixing bad
synth quality move P&L", re-point the plan at NNMT + loosen its guards (disable
apply_task_filters). Open decision.**

**Phase 2 DONE (2026-07-21) — the premise was WRONG; GAN BEATS non-GAN at the powered
operating point.** Chased "why is the GAN consistently worse" and found it isn't.
Journey: (a) diagnosed conservatism — GAN takes 24 fewer trades, dropping net-winning
marginals (−0.76pp); traced to under-dispersed synth (σ_ratio ~0.5). (b) Tried to widen:
post-hoc z-space scale broke joints (clip truncation); DDIM η hit the wrong sampler
(model uses EDM/Heun); EDM churn joint-safe but INEFFECTIVE (denoiser denoises noise back
to modes — under-dispersion is baked into the score); post-hoc OUTPUT scale widens σ +
preserves correlations EXACTLY (validated) BUT is off the nonlinear manifold → AE rejects
~98% → Metal ~500K crash (→ added 400K draw cap in balance.py). **Dispersion widening =
DEAD END** (off-manifold, P&L-negative). (c) Turned the AE filter OFF (draw cap makes it
safe) — and the GAN jumped. **Paired seeds {default,1,7,13} @ pred_thr 0.45 (guards on,
config confirmed consistent):** non-GAN mean 20.29 vs AE-on 21.40 vs AE-off **21.96**.
**AE-off > non-GAN 4/4 (mean +1.68pp); AE-on > non-GAN 3/4 (+1.11); AE-off > AE-on 3/4
(+0.57, seed13 flips).** So BOTH GAN variants beat non-GAN on average — the "consistent
underperformance" ([[feedback_gan_ratio_sweep_no_gan_wins]]) was tight-guard + single-seed.
AE filter is neutral-to-slightly-negative at powered guards (helpful at tight guards,
[[project_ae_filter_win]]) — same operating-point flip as everything else
([[feedback_evaluate_interventions_at_powered_operating_point]]). **CAVEAT:** the edge
(~1.5pp) shows at the LOOSE operating point (0.45, ~580 tr), NOT the tight deployment
(0.6, ~166 tr, where non-GAN won). Opens: is loose-guards+GAN > tight-guards+non-GAN? —
a deployment-config question. Repro: `NNNC_DDPM_MLX_P2_aeoff` (+ seed variants via
TrainSeedStrategyMixin). Kills: LONG background batches get SIGKILLed (external, not OOM —
mem 94% free, no traceback); single runs survive → run seed sweeps one-at-a-time.
