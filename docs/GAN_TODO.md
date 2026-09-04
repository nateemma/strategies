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

---

# Fidelity / parity backlog (opened 2026-08-29, after the CTAB-TF fix)

Context: [[project_ctab_tf_joint_structure_fixed]] closed CTAB-TF's joint-structure
defect (mean|corr| 0.043 → 0.240, 95% of MLX). That fixed ONE defect in ONE family.
The scorecard's D1 table still shows the rest:

| variant | sigma ratio | dcorr | state |
|---|---|---|---|
| CTAB-TF | 0.83 | 1.03 -> **0.12** | joint structure FIXED |
| MT_CTAB-TF | 0.95 | 1.08 -> **~0.23** | FIXED but PARTIAL -> task 1 |
| MT_WGAN-TF | **0.17** | 0.57 | worst under-dispersion -> task 4 |
| WGAN-MLX | 0.36 | **1.14** | widest joint gap -> task 3 |
| MT_DDPM-MLX | **3.96** | 1.12 | over-dispersed ~4x |
| WGAN-TF | 0.78 | 0.21 | best joints, still under-dispersed |

**Under-dispersion is the dominant unsolved defect class** — 5 of 6 variants emit
too little variance, one emits ~4x too much. D1 proved the filters cannot repair
it (rejection cannot add variance that was never generated).

Note on priority: GAN augmentation does not move P&L on this pipeline regardless
of fidelity (the information ceiling, §4/§5 above). These tasks are FIDELITY /
PARITY work — justified because TF is the only backend for users not on Apple
silicon, not because they are expected to show up in a backtest.

## Task 1 — sweep `corr_loss_weight` for MT_CTAB-TF  [DONE 2026-08-29: cw 15 -> 30]

MT reached 0.126 / +0.834, only ~51% of MT-MLX, where single-task reached 95%.
`corr_loss_weight=15` was tuned on the SINGLE-TASK objective; MT's generator loss
also carries info and downstream terms, so its optimum is very likely different.
Sweep, then 3-seed the winner. Cheapest known headroom.
Gate: `ctab_corr_probe --gan-type MT_CTAB_GAN`.

**RESULT:** optimum is 40 (0.155 / +0.862), shipped 30 (0.146 / +0.858) for
margin — the cliff at 40-50 fails SILENTLY (restores epoch 2, worse than no fix).
MT 51% -> 62% of MT-MLX. Full tables in GAN_SCORECARD.md.
**Key finding for task 2: the cliff is a SHARE of the generator loss, not an
absolute scale** (single-task ~15-20, MT ~40-50, MT carries more non-adversarial
loss). Residual MT-vs-single-task gap is unexplained and still open.
Side observation: selection restores epoch 14/40 at high cw — TESTED, selection
HELPS (turning it off is worse); not a defect.

**MT residual gap chased separately: SEVEN hypotheses refuted, cause NOT found.**
See GAN_SCORECARD.md. Excluded: architecture (measured identical), downstream
loss, selection, LR schedule, the cw confound, per-batch corr-target noise, and
Training-By-Sampling (tested on both a balanced and a 6.7:1 imbalanced fixture).
Remaining untested surface is the DISCRIMINATOR side (D updates, GP term,
n_critic). Recommend task 2 ahead of resuming this.

## Task 2 — re-test CGAN-TF dispersion penalties with the G clip REMOVED  [DONE 2026-08-30: REFUTED]

The scorecard refuted CGAN's data-space std-matching penalty as "flat at
w=1/5/20 — the adversarial signal out-muscles it even at w=20". **But CGAN-TF
clips its generator at `clipnorm=1.0`** (`df_cgan.py:433,547`, via Adam), and
under a norm clip raising the weight CANNOT increase update magnitude — it only
rotates the direction, and past a threshold destabilises. That is exactly the
behaviour measured on CTAB today (monotone gain to cw=15, collapse at 20, plus a
further gain from removing the clip that no weight could buy).

So the null was measured under the very constraint that plausibly caused it.
Add a G-clip knob to CGAN and re-run the std-matching sweep with it off.
Same "re-test nulls after a mechanism fix" lesson as BatchNorm-vs-LayerNorm.

**RESULT: REFUTED.** No interaction — the penalty gains +0.026 clipped and
+0.033 unclipped. The clip is NOT what suppressed it, and the CTAB G-clip win is
CTAB-SPECIFIC rather than family-wide. **Do not sweep the clip everywhere.**
Weight escalation also refuted: sigma saturates at ~0.58 (w 20->500 buys 0.007)
while joint structure collapses (+0.982 -> +0.262). The scorecard's original
"moment penalties do not fix CGAN dispersion" stands on stronger evidence.
Kept `std_match_weight` (default 0.0) + `g_clip_norm`/`d_clip_norm` knobs.
Next lead for CGAN dispersion is the GENERATOR OUTPUT PARAMETERISATION, not the
loss. Full tables in GAN_SCORECARD.md.

CAVEAT: CGAN-TF is NOT run-to-run reproducible (same config/seed gave sigma
0.438 then 0.391). Any CGAN comparison needs repeats; single-run deltas mean
nothing.

## Task 3 — port the correlation term to WGAN-MLX  [DONE 2026-08-30: default cw=1.0]

WGAN-MLX holds the widest joint gap left (dcorr 1.14). The mechanism is generic —
supply the gradient the critic does not — and now lives in `CTABGANPlusBase`.
WGAN has no VGM mode heads, so there is no soft-decode step: match correlation on
the raw generator output directly. Note WGAN-MLX already got the class-mean fix
(D2) and its dcorr stayed 1.14, so the joint defect is untouched by that.

**RESULT: WORKS — the mechanism ports.** corr-of-corr +0.805 -> +0.923,
mean_dcorr 0.211 -> 0.135 (3 seeds, no overlap with baseline). Shipped cw=1.0,
the sigma knee: correlation keeps improving to cw=40 but sigma inflates past 1
(1.10 -> 1.71), trading dispersion for correlation. First PORTABLE fix in this
backlog. CAVEAT: WGAN-MLX training is unseeded (`mx.random.seed` never called),
so every run is an independent draw — the sigma gain at cw=1 is within noise and
is NOT claimed; only "does not degrade sigma". Full tables in GAN_SCORECARD.md.

## Task 4 — diagnose MT_WGAN-TF sigma = 0.17  [VOID 2026-08-30: sigma is 0.699, not 0.17]

Emits 17% of the required variance — the most severe single defect in the table,
and worse than CGAN's 0.44-0.55. Diagnose BEFORE attempting a fix (the D2/CTAB
playbook: budget ladder, then probe the component the divergence implicates).
Do not assume it is the same cause as CGAN's under-dispersion.

**VOID.** Re-measured first: MT_WGAN-TF is sigma 0.699 with EXCELLENT joint
structure (dcorr 0.087, coc +0.965, 3 runs) — one of the healthier variants.
D1's 0.17 predates fixes that have since landed. **D1's sigma column is retired**
and with it this backlog's "under-dispersion is the dominant unsolved defect
class" framing: on re-measurement most variants are sigma 0.70-1.09, coc +0.86
to +0.99. See the re-measured inventory in GAN_SCORECARD.md.

## Task 5 — TAB_DDPM / MT_DDPM  [CLOSED 2026-08-30: NO production defect; harness artifact]

Surfaced by task 4's re-measurement. TAB_DDPM-MLX has BOTH failure modes:
sigma 3.946 (4x over-dispersed) AND corr-of-corr -0.026 — ZERO joint structure,
indistinguishable from CTAB-TF pre-fix. MT_DDPM-MLX likewise (3.964 / +0.284).
Every other variant is now +0.86 or better. This is also the production
augmentation path.

**FIRST STEP IS NOT A FIX.** GAN_TODO §5 Phase 1 recorded the opposite for this
model ("high-fidelity, sigma_syn/sigma_real ~0.5-1.0") from the PRODUCTION
fidelity report. Raw-generator-on-fixture vs post-scaling-production-report
cannot both be right. Run both on the same model instance and find where they
diverge before designing anything. Task 4 is a worked example of what building
on an unverified baseline costs.

Note the correlation term does not port here as-is: diffusion has no adversarial
generator step to attach it to, so it would need a different attachment point.


### Task 5 update (2026-08-30)

**"Zero joint structure" RETRACTED** — a probe artifact. The scorecard adapter
runs a toy config (d_model 32 vs 256, timesteps 100 vs 1000). At production shape
+ 300 epochs corr-of-corr is +0.496, not -0.026. Still the family's worst, but
not the CTAB-TF disease.

**sigma ~3.9 STANDS** — invariant to 8x width, 10x timesteps, 7.5x budget. The
+/-4 sigma clip-saturation signature.

**BLOCKED on reconciliation.** Production's fidelity report says sigma ~0.5-1.0
for this model; capacity does not explain the gap. Run BOTH paths on the SAME
model instance before designing any fix. Not yet run.


## Task 6 — MT_CTAB-TF residual gap: the DISCRIMINATOR side  [DONE 2026-08-30: partial — D sets the LEVEL, not the stall]

MT stalls after ~epoch 15 while single-task keeps climbing, at MATCHED corr
weight and byte-identical networks (G 366,600 params, D 749,825). MT is even
AHEAD at epoch 10. The defect is a PLATEAU, not a deficit.

**Seven hypotheses refuted** (full detail in GAN_SCORECARD.md): architecture
(measured identical), the downstream BCE-vs-CE loss, model selection (it HELPS),
the LR schedule (fires after the selected epoch), a cw confound in the first
ladder, per-batch corr-target noise (global target inert), and
Training-By-Sampling (inert on balanced AND 6.7:1 imbalanced fixtures).

**Untested surface: the discriminator.** D's update rule, the gradient-penalty
term, `n_critic`. Everything on the generator side, the losses, the sampler,
selection and the schedule is excluded. Use the budget-ladder-then-probe method
that root-caused CTAB-TF: instrument both implementations, find where they
diverge, then probe the component the divergence implicates.

Related: MT_CTAB's correlation sits at +0.858 vs MT-MLX's +0.985. Likely the same
stall rather than a separate defect — do not chase it independently until task 6
resolves.

## Task 7 — CGAN dispersion: the GENERATOR OUTPUT PARAMETERISATION  [DONE 2026-08-30: REFUTED — adversarial equilibrium]

sigma 0.547 against a target of 1.0. **The loss-side lever is EXHAUSTED** (task
2): data-space std penalties saturate at sigma ~0.58 and destroy joint structure
when pushed (corr-of-corr +0.982 -> +0.262 at w=500), and the gradient clip is
not the limiter.

Remaining lead is the generator's own output: the layer is LINEAR with no
activation ceiling, and only 1.9% of values pin at the [min,max] bound, so the
generator simply emits ~46% of the required variance from its own score. Same
shape as the MT_DDPM "under-dispersion is baked into the denoiser" finding, where
post-hoc and loss-side corrections both failed.

NOTE CGAN's joint structure is EXCELLENT (+0.982, better than the fixed CTAB-TF).
Any dispersion fix must not cost correlation — the penalty route fails precisely
because it does.

## Parked — the corr-weight divergence cliff  [CHARACTERISED, NOT EXPLAINED]

Above cw ~15-20 (single-task) / ~40-50 (MT) training diverges, the guard fires
and restores an untrained epoch. **Characterised:** it is a SHARE of the
generator loss, not an absolute scale — which correctly predicted MT's higher
cliff from its larger non-adversarial loss mass. **Not explained:** it is neither
gradient clip and not a startup transient (a warm-up ramp only delays it).

Bounds achievable magnitude but blocks nothing; both families ship safely below
it. Deliberately parked — do not spend runs guessing at it without a new lead.


### Task 5 mechanism (2026-08-30)

sigma ~4 is **CLIP SATURATION**, not a training failure: **99.8% of generated
values sit at `_ZSCORE_CLIP = ±4`** (mean |z| 3.997, p99 4.000) against real
0.1%. sigma ~3.9 is arithmetic. Eliminated first, on one model instance:
aggregation (per-class 3.320 vs pooled 3.280), filtering (|z|<=4 keeps ZERO
synth, so it cannot narrow anything), data volume (3.4-3.9 at 4k/12k/40k rows),
and capacity/timesteps/epochs. Repro: `GANs/quality/ddpm_clip_probe.py`.

**Open:** does production saturate too? It feeds PRE-SCALED data where the probe
feeds raw, and its report reads 0.5-1.0 — so pre-scaling probably avoids it.
Run the clip probe against a production-path instance and compare the boundary
fraction. Do NOT build a sampler fix first.


### Task 5 CLOSED (2026-08-30)

Production-trained TabDDPM (`saved_data/GANs_PostScale/tab_ddpm`, same
architecture as the ladder) does **NOT** saturate: sigma 1.025 in z-space,
|z| mean 0.878, **0.2%** at the ±4 boundary — versus the probe's 99.8%. Real
data reference is 0.1%. **The production model is healthy and GAN_TODO §5
Phase 1 was right.**

"TAB_DDPM is the family's worst defect" is FULLY RETRACTED — a harness artifact
in two layers: the adapter's toy config (killed joint structure) and training on
the RAW fixture rather than production's pre-scaled data (saturates the clip).

**Adapter fix attempted — the pre-scaling idea was a NO-OP.** The fixture is
already in production-like space (std median 0.580 vs production 0.609);
`build_real_fixture` normalises. Also eliminated: dropout (the only config
difference), training steps up to production's own 58,500, 10x rows, and the
degenerate `bb_width` feature. The fixture path still pins 95.9-99.8% of values
at the clip where production hits 0.2%. **Cause unresolved — the remaining
difference is the DATA itself.**

**Shipped instead: a clip-saturation guard.** `ctab_corr_probe` reports
`clipsat=` and warns loudly above 20%, so the harness declares its own results
invalid rather than emitting garbage (TAB_DDPM 100.0% warns; WGAN 0.1% silent).
**Standing rule: evaluate DDPM against the production-trained model on disk, not
a fixture-trained one.**


### Task 7 result (2026-08-30) — REFUTED, and it closes the generator side

A learnable per-feature output gain is INERT (sigma 0.553 vs baseline 0.558) and
adds nothing to the std penalty (0.573 vs 0.575), 3 seeds.

**Diagnostic:** the learned gain stays at ~0.97-0.98 (init 1.0) EVEN WITH the std
penalty at w=20 demanding more variance. Scaling one parameter is the cheapest
route to higher sigma and the generator declines it — so the adversarial loss
actively opposes dispersion harder than the penalty pushes.

**CGAN under-dispersion is an ADVERSARIAL EQUILIBRIUM, not a generator defect.**
Generator-side work is CLOSED: loss (task 2), weight to 500, gradient clip, and
output parameterisation all eliminated. Any fix must change D's objective or use
a divergence that penalises under-dispersion. Retro-explains task 2's saturation
at ~0.58 — an equilibrium, not a weak lever.


### Task 6 result (2026-08-30) — partial; stall still unexplained (8th hypothesis)

D-side code is functionally identical to single-task's, so tested behaviour:
weakening D should let the corr term settle higher if the plateau is an
adversarial equilibrium.

**Confirmed for the LEVEL.** `n_critic=2`, 3 seeds, tight (0.172/0.171/0.170, no
overlap with baseline): |corr| 0.146 -> 0.171, dcorr 0.211 -> 0.186, coc +0.858
-> +0.880. Clean differential — weakening D HURTS single-task (0.232 -> 0.219),
so it relieves an MT-specific constraint. `gp_weight` is not a lever.

**NOT confirmed for the STALL.** Trajectory (selection off) still peaks ~ep20 and
declines by ep40 — the plateau sits higher, it does not lift. D pressure sets
where MT settles, not why it stalls. **Stall unexplained after 8 hypotheses.**

**Default unchanged:** n_critic=2 trades sigma (1.001 -> 0.952) for correlation.
Not strictly dominant, unlike every default changed this session. Use it if
correlation matters more than dispersion.


---

# Research backlog (opened 2026-08-31, after tasks 1-7 closed)

Tasks 1-7 exhausted the *debugging* seam: every remaining item needs a new
mechanism or a new instrument, not another tuning hypothesis. Ranked by how
well-grounded each is in what tasks 1-7 actually measured.

**The finding these are built on:** three independent instances showed the
generators cannot satisfy BOTH moments at once — CGAN's std penalty bought sigma
and collapsed correlation (+0.982 -> +0.262); WGAN-MLX's corr weight bought
correlation and inflated sigma (1.10 -> 1.71); MT_CTAB's `n_critic=2` buys
correlation with sigma (1.001 -> 0.952). Every default shipped was chosen on
STRICT DOMINANCE, never on the best single metric.

## Task 8 — match COVARIANCE, not correlation  [DONE 2026-08-31: CONFIRMED on WGAN-MLX; the tradeoff was an artifact]

`_correlation_loss` is **scale-invariant by construction**: it normalises each
feature before comparing, so it constrains the correlation structure and says
NOTHING about sigma. That is exactly why raising it inflates dispersion on
WGAN-MLX, and the mirror of why CGAN's sigma penalty destroys correlation — each
term pins one moment and leaves the other free.

Covariance = correlation x the two sigmas. **Matching the covariance matrix
constrains both jointly, in one term.**

- Small edit: drop the per-feature normalisation in `_corr_offdiag` /
  `_correlation_loss` (`GANs/ctab_gan_base.py`), reusing all existing plumbing,
  probes and the `--set` passthrough.
- **Falsifiable prediction:** the sigma inflation at high corr weight disappears.
- Either outcome is informative — if the tradeoff is an artifact of scale-free
  targets it dissolves; if covariance matching hits the same wall, the tradeoff
  is intrinsic to these generators, which is worth knowing.
- Gate on BOTH axes (dcorr and sigma) and 3 seeds, as everything else was.
- Applies to CTAB-TF, MT_CTAB-TF and WGAN-MLX — all three already carry the term.

## Task 9 — repair D's joint blindness instead of bypassing it  [OPEN]

The ORIGINAL root cause is still unfixed. CTAB-TF's discriminator scores
class-conditionally column-shuffled data as real: `joint_sensitivity` 0.045 vs
CTAB-MLX's 0.67, 3 seeds, no overlap. The corr term routes AROUND this rather
than repairing it.

Unexplained: why MLX's critic acquires joint sensitivity at epoch 10 (the same
epoch its generator's correlation onset happens) and TF's never does, on
near-identical architectures. The instrument already exists
(`GANs/quality/ctab_disc_probe.py`).

Most principled remaining question in the family — and task 7 pointed at D's
objective independently, from the other direction. **Worst track record though:**
nine causes refuted before this session, plus the gradient-clip hypothesis
refuted in task 2. Do not start without a new idea about D specifically.

## Task 10 — build a gradient-balance instrument  [OPEN — highest leverage per hour]

Two open items are the same shape: "the adversarial term and an auxiliary term
interact in a way I cannot see."

- **MT's stall** — 8 hypotheses refuted. Task 6 showed D pressure sets the LEVEL
  but not the stall; the trajectory still peaks ~ep20 and declines.
- **The cw cliff** — characterised as a SHARE of the generator loss (which
  correctly predicted MT's higher cliff) but never explained.

Log per-epoch gradient norms of each loss component (adversarial, corr,
downstream) for MT vs single-task. That shows directly whether the corr gradient
vanishes, gets overwhelmed, or starts OPPOSING around epoch 15.

**One instrument, two parked questions.** Eight hypotheses died on the stall
because they guessed at causes instead of measuring the interaction — this is the
measurement.

## Task 11 — G/D balance beyond `n_critic`  [OPEN]

Task 6 found a real differential — weakening D HELPS MT (0.146 -> 0.171) and
HURTS single-task (0.232 -> 0.219) — probed with only one crude knob.

Standard levers for exactly the equilibrium pathology tasks 6 and 7 both landed
on, none tried here:
- `critic_lr_ratio` — already exists on WGAN-MLX, never swept.
- TTUR (separate G and D learning rates).
- Spectral normalisation on D in place of the gradient penalty (known to affect
  mode coverage, which is what an under-dispersion equilibrium is).

Cheap, and it attacks the equilibrium directly rather than through a penalty
(which task 7 proved cannot win against it).

## Task 12 — does working synth move P&L?  [OPEN — UNBLOCKED 2026-08-31 by task 8]

Prior work concluded GAN augmentation does not move P&L at any fidelity (the
information ceiling). **That was measured on generators that emitted essentially
independent features.** CTAB-TF went from corr-of-corr -0.047 to +0.945.

A powered A/B would test whether that conclusion survives working synth.
**Blocker:** the fixes are TF and production runs MLX, so the correlation term
must reach the production path first (CTAB-MLX already has good joint structure,
so the honest test is TabDDPM or the MLX WGAN path).

Use the established protocol — pinned timerange, paired seeds, loosened guards
for trade volume ([[feedback_evaluate_interventions_at_powered_operating_point]]).

## Task 13 — make the scorecard able to measure diffusion  [OPEN — small]

The DDPM adapters train on the raw fixture and produce a degenerate model
(95.9-99.8% of values pinned at the +/-4 clip vs a production-trained model's
0.2%). Cause unresolved after eliminating scaling, dropout, steps to production's
own count, 10x rows, capacity, timesteps and the degenerate `bb_width` feature.

The `clipsat=` guard now makes this LOUD rather than silent, so nothing is
blocked. Either make the harness train DDPM the way production does, or drop the
DDPM rows and evaluate them only against the production-trained model on disk.


### Task 8 result (2026-08-31) — CONFIRMED, and it retracts the tradeoff finding

`_corr_offdiag` is blind to dispersion BY CONSTRUCTION — a 2x sigma error scores
**0.000000** under it. The "generators cannot satisfy both moments" conclusion
was an artifact of that, not a property of the models.

**WGAN-MLX: covariance mode is strictly dominant at EVERY weight.** Sigma pinned
at 1.01-1.05 across a ladder where correlation mode ran to 1.706, with BETTER
correlation throughout, and NO cliff to cw=200. New default
`corr_loss_mode="covariance"`, `corr_loss_weight=100`: mean|corr| 0.303 (real
0.306), coc +0.998, dcorr 0.021, sigma 1.020 — 6x better dcorr than the previous
default with sigma essentially at 1. Weight is MODE-DEPENDENT (covariance ~100,
correlation ~1); change both together.

**CTAB-TF: covariance is WORSE (dcorr 0.167 vs 0.122) at every weight, and still
has a cliff. Default unchanged — OPEN question why the same term dominates on one
family and loses on the other.**


### Task 8 follow-up (2026-08-31) — WHY covariance loses on CTAB-TF: ANSWERED

Decomposed via a third mode `covariance_offdiag` (scale-sensitive, no diagonal),
CTAB-TF cw=15: correlation dcorr 0.122 -> covariance_offdiag 0.152 -> covariance
0.167. **Scale-sensitivity is the bigger cost; the diagonal adds less.**

**Mechanism:** covariance off-diagonal is `corr * sigma_i * sigma_j`, so when
sigma != 1 matching it forces correlation to compensate the WRONG way (an
over-dispersed generator must UNDER-correlate). Measured: |corr| 0.196 vs 0.232,
real 0.306. The diagonal is what drives sigma -> 1 and makes the coupling
harmless — **it works on WGAN-MLX (1.706 -> 1.008) and does NOT on CTAB-TF
(1.101 -> 1.126, slightly worse).**

**PREDICTIVE RULE: use covariance where dispersion is BROKEN and the diagonal can
move it; use correlation where dispersion is already near-correct.** CTAB's sigma
is held near 1.10 by something the diagonal cannot overcome — same shape as
CGAN's adversarial-equilibrium dispersion (task 7). Check sigma FIRST, then pick
the term.

Also refuted: soft-decode variance bias (fake mode-block max prob 0.870 vs real
1.000, but only a ~1.4% sigma bias — too small to matter).


### Task 12 UNBLOCKED (2026-08-31)

Its stated blocker was "the fixes are TF and production runs MLX, so the corr
term must reach the production path first". **Task 8 did exactly that.**
WGAN-MLX now ships covariance/100 and produces mean|corr| 0.303 (real 0.306),
corr-of-corr +0.998, mean_dcorr **0.021**, sigma **1.020** — roughly a 10x
fidelity improvement over the state in which the information-ceiling conclusion
was measured (dcorr ~0.21 with sigma drifting).

`saved_data/NNNC_WGAN_MLX/` and `NNMT_WGAN_MLX/` exist, so WGAN-MLX-augmented
strategies are real and trained — the A/B needs a retrain, not new plumbing.

**The premise of "GAN augmentation does not move P&L at any fidelity" no longer
holds:** it was established on generators emitting badly wrong joint structure.
This is now a clean test of a conclusion whose premise has changed.

Protocol (unchanged): pinned timerange, paired seeds, guards loosened for trade
volume ([[feedback_evaluate_interventions_at_powered_operating_point]]), and
judge on P&L not val_mcc ([[feedback_learnability_mcc_doesnt_equal_edge]]).


---

# COVERAGE GAP — the term reaches only 3 of 9 variants (opened 2026-08-31)

Tasks 3 and 8 built and validated the joint-structure term but only ever wired it
into `df_ctab_gan`, `df_mt_ctab_gan` and `df_wgan_mlx`. Six variants never got
it. Cross-referenced against the re-measured inventory, ranked by defect size.
**Apply the task-8 rule: check sigma FIRST, then pick the mode** — covariance
where dispersion is broken and the diagonal can move it, correlation where
dispersion is already near-correct.

| variant | sigma | dcorr | has term | action |
|---|---|---|---|---|
| WGAN-MLX | 1.020 | 0.021 | covariance | done (task 8) |
| CTAB-TF | 1.091 | 0.122 | correlation | done (sigma fine -> correlation) |
| MT_CTAB-TF | 1.001 | 0.211 | correlation | done; dcorr still worst of the three |
| **MT_WGAN-MLX** | 0.939 | **0.143** | NO | **task 14 — port; worst dcorr in the WGAN family** |
| **MT_WGAN-TF** | **0.699** | 0.087 | NO | **task 15 — port; sigma broken, so covariance** |
| **CGAN-TF** | **0.547** | — | NO | **task 16 — worst sigma in family; see caveat** |
| WGAN-TF | 0.887 | 0.036 | NO | task 17 — already good, modest upside |
| TabDDPM / MT_DDPM | — | — | NO | blocked: fixture harness invalid for diffusion (task 13) |

## Task 14 — port the term to MT_WGAN-MLX  [DONE 2026-08-31: default correlation/15]

Worst dcorr in the WGAN family (0.143 vs single-task's 0.021 after task 8) and
sigma 0.939. Same MLX code pattern that already works in `df_wgan_mlx`
(precomputed constant target, `loss_gen` sees no real data), so this is a proven
port rather than new design. Sigma is near 1, so start with `correlation`, but
test both — 0.939 is far enough off that the diagonal may have real work.

## Task 15 — port the term to MT_WGAN-TF  [DONE 2026-08-31: default covariance/40; RULE CONFIRMED]

Sigma 0.699 is the most broken dispersion outside CGAN, and dcorr 0.087 is
already decent. **By the task-8 rule this is the strongest covariance
candidate**: the diagonal has genuine work to do. TF/pac code pattern, closer to
`df_ctab_gan` than to the MLX ports.

## Task 16 — covariance on CGAN-TF  [DONE 2026-08-31: REFUTED, as predicted]

Sigma 0.547 is the worst dispersion in the family and correlation is EXCELLENT
(+0.982), so the profile is exactly what covariance is for.
**CAVEAT — expect it to fail.** Task 7 established CGAN's under-dispersion is an
ADVERSARIAL EQUILIBRIUM: a learnable per-feature gain never moved off 1.0 even
with an explicit std penalty at w=20 demanding more variance. A covariance term
is still a generator-side penalty and will probably hit the same wall. Worth ONE
test because the defect is the largest left; do not iterate if it behaves like
task 7.

## Task 17 — port the term to WGAN-TF  [DONE 2026-08-31: INERT, as predicted — default off]

Already the best joint structure in the family (dcorr 0.036) with sigma 0.887.
Little headroom; do last, if at all.


### Task 14 result (2026-08-31)

dcorr 0.148 -> **0.066** (3 runs, tight) with sigma statistically unchanged
(0.951 vs 0.958, overlapping). Shipped `correlation` / cw=15. cw=40 reaches dcorr
0.039 but drifts sigma to 0.923 — a trade, not a dominance; available as an
option. Stock run: mean|corr| 0.299 (real 0.306), coc +0.976, sigma 0.945.

**The task-8 predictive rule survived**: sigma was near-correct (0.917-0.958) and
correlation was indeed competitive; covariance offered no advantage. Correlation
mode did NOT inflate sigma here because this variant starts UNDER-dispersed —
scale-freedom only runs away when sigma has room to grow.

**The rule is now trustworthy enough to pick the MODE for tasks 15-17 without
sweeping both.** Weight still needs sweeping — it has never transferred (15, 30,
100, 15 across four variants so far).


### Task 15 result (2026-08-31) — port + the predictive rule CONFIRMED

3 runs, no overlap on either axis: dcorr 0.069 -> **0.047**, sigma 0.723 ->
**0.853**. Shipped `covariance` / cw=40. Stock: mean|corr| 0.304 (real 0.306),
coc +0.991, dcorr 0.043, sigma 0.847.

**RULE CONFIRMED in the positive direction, on a variant not used to build it.**
Correlation-mode control at cw=15 reaches the SAME dcorr (0.056 vs 0.055) but
leaves sigma at 0.666 — unchanged from the 0.673 baseline, exactly as a
scale-blind term must — while covariance moves it to 0.860. **Mode can now be
picked from measured sigma without sweeping both** (weight still needs sweeping:
15, 30, 100, 15, 40 across five variants).

Limit: sigma 0.853 is not 1.0. The diagonal's authority VARIES by variant
(WGAN-MLX reached 1.008). That is the rule's "and the diagonal can move it"
clause; CGAN is the extreme test.


### Task 16 result (2026-08-31) — REFUTED; task 7 beats the task-8 rule

sigma FLAT at 0.556-0.585 across a 100x weight range (cw 5 -> 500), the same
~0.58 saturation the std penalty hit in task 7.

**Three distinct generator-side mechanisms now fail identically on CGAN:** a free
per-feature gain (task 7), a direct std penalty (task 2), and the second-moment
diagonal (here). **CGAN's under-dispersion cannot be moved from the generator
side** — any fix must change D's objective.

**The rule gains its qualifier:** covariance helps where sigma is broken AND THE
DIAGONAL CAN MOVE IT. Authority varies: WGAN-MLX full (1.706 -> 1.008),
MT_WGAN-TF partial (0.723 -> 0.853), CGAN none (0.571 -> 0.563).

Default stays 0.0. A marginal correlation gain exists (dcorr 0.053 -> 0.034) but
is single-seed on a noisy variant, and correlation was never CGAN's problem.


### Task 17 result + COVERAGE WORK COMPLETE (2026-08-31)

WGAN-TF: inert (dcorr 0.034 baseline vs 0.030/0.038/0.041 across weights —
non-monotonic, spread ~ the difference from baseline). Nothing to fix; the
variant already had the family's best joint structure. Default stays 0.0; the
term is wired and available.

**Coverage complete — the term now reaches every variant where it can help:**

| variant | dcorr | sigma | mode / weight |
|---|---|---|---|
| WGAN-MLX | 0.021 | 1.020 | covariance / 100 |
| MT_WGAN-TF | 0.043 | 0.847 | covariance / 40 |
| MT_WGAN-MLX | 0.074 | 0.945 | correlation / 15 |
| CTAB-TF | 0.122 | 1.091 | correlation / 15 |
| MT_CTAB-TF | 0.211 | 1.001 | correlation / 30 |
| WGAN-TF | 0.034 | 0.891 | off — nothing to fix |
| CGAN-TF | 0.064 | 0.547 | off — dispersion is an EQUILIBRIUM |

**Remaining fidelity items are all low-yield or blocked:** tasks 9 (D joint
blindness, 10 refutations behind it), 10 (gradient-balance instrument, serves two
parked puzzles), 11 (G/D balance), 13 (DDPM harness). None has a specific
untested mechanism of the kind tasks 8/14/15 had.

**NEXT: task 12 — the P&L A/B.** The fidelity list is complete.

---

## Task 18 — consolidate GAN storage and the scaling-pipeline split  [OPEN]

**Problem.** GAN artifacts are spread across three parent directories chosen by
two flags, and the flags do not always mean what the path implies.

`GANs/paths.py` selects the parent from `use_pca` / `post_gan_scaling`:

| flags | parent |
|---|---|
| neither | `GANs/` |
| `post_gan_scaling=True` | `GANs_PostScale/` |
| `use_pca=True` (wins over post-scale) | `GANs_PCA/` |

Current state on disk is already split, with no single type present in both:

```
saved_data/GANs/mt_ctab_gan
saved_data/GANs_PostScale/{ctab_gan, wgan, mt_wgan}
```

**The sharp edge.** `BaseNNStrategy.use_post_gan_scaling` is documented as
"only supported for MT_DDPM with single-task labels currently; other gan_type
values fall back to the pre-GAN scaling pipeline regardless of this flag" — but
the flag still changes the SAVE PATH unconditionally. So for any non-MT_DDPM
type, setting it trains a **pre-scale GAN into the PostScale directory**: the
path claims one pipeline, the weights are the other. Nothing detects this,
because the pipeline choice is not recorded in the GAN metadata.

Costs already paid: time lost looking for `mt_ctab_gan` under the wrong parent,
and a real risk of loading a GAN trained under a different scaling pipeline than
the consumer assumes — the same class of silent mismatch as the
`label_filter_window` bug (fixed 2026-09-03), which was invisible for exactly the
same reason: the distinguishing property was not in the metadata.

**Proposal.**
1. **One approach.** Decide whether post-GAN scaling is the pipeline or is
   dropped. It is currently supported for one gan_type out of ~8, and the
   recorded verdict is that GAN augmentation does not help P&L at any fidelity,
   so "drop it" is a legitimate answer and the cheapest one.
2. **One directory.** Collapse to a single `GANs/<gan_type>/` layout. PCA and
   scaling variants become *metadata*, not path segments.
3. **Record the pipeline in metadata**, alongside `training_type`, `horizon` and
   `label_filter_window`, and add it to `_gan_expected_metadata` so a mismatch
   fails loudly instead of resolving by directory name.
4. **Migrate or delete** the existing `GANs_PostScale/` artifacts. Given every
   GAN is already stale against the peak-filter fix, deleting and retraining is
   simpler than migrating and loses nothing.

**Sequencing note.** Point 3 is the durable part and is cheap. Points 1–2 touch
`paths.py`, every `Create*` script and the loaders, so do them together and only
once the "is post-GAN scaling worth keeping at all" question is answered.

**Priority: LOW for P&L, MEDIUM for correctness.** No GAN is in production —
`NNNC_MLX` and `NNMT_MLX` both run `gan_type = NONE`. This is cleanup that
prevents a future silent mismatch, not a performance item.
