# GAN Parity & Quality Assessment — Implementation Plan

Spec: `docs/superpowers/specs/2026-08-26-gan-parity-design.md`
Status: Phase A/B/C are actionable now. Phase D is intentionally empty until the
Phase-B scorecard has run.

## Preflight

Run from the freqtrade root with:

    PYTHONPATH=. .venv/bin/python -m pytest user_data/strategies/GANs/tests/... -v

Verify before starting:

    # registry resolves for all types (registry populates on importing GANs.backends —
    # importing GANBackend alone leaves it EMPTY and every lookup fails misleadingly)
    PYTHONPATH=.:user_data/strategies .venv/bin/python -c "
    import GANs.backends
    from GANs.GANType import GANType
    from GANs.GANBackend import resolve_backend
    for t in GANType:
        if t.name in ('NONE','BOTH'): continue
        for m in (True, False):
            try: print(t.name, m, resolve_backend(t, prefer_mlx=m).__name__)
            except Exception as e: print(t.name, m, 'NONE')
    "

Expected: WGAN/MT_WGAN/CTAB_GAN/MT_CTAB_GAN resolve on both; CGAN TF-only;
TAB_DDPM/MT_DDPM MLX-only.

## Task A1: `GANs/quality/` package + utility probe

Create `GANs/quality/__init__.py` and `GANs/quality/utility_probe.py`.

`utility_probe.delta_val_mcc(real_X, real_y, synth_X, synth_y, *, seed) -> dict`

- Fixed stratified split of `real` into train/val (val never sees synth).
- Fit the classifier twice: on `train` and on `train + synth`.
- Return `{"mcc_real": float, "mcc_aug": float, "delta": float, "n_synth": int}`.
- Classifier must be cheap and deterministic — this runs 11 times. Use the
  repo's existing sklearn-side classifier, NOT an MLX/TF net: the probe measures
  *synth usefulness*, and a heavyweight model adds variance and runtime without
  changing the ranking.
- Guard: if `synth` is empty or non-finite, return `delta=None` and a reason
  string rather than raising — a broken variant must still produce a row.

Acceptance: unit test with synth drawn from the same distribution as real
(delta ≈ 0 within noise) and synth drawn from pure noise (delta clearly negative).

## Task A2: manifold metrics

Add to `GANs/quality/scorecard.py`:

- `clip_band_fraction(synth_z, clip)` — fraction of values within 1e-6 of ±clip.
  This is the metric that would have caught the MT_DDPM over-dispersion
  (σ saturating the ±4σ band); RMSE-style metrics do not show it.
- `nn_distance_ratio(real, synth)` — median nearest-neighbour distance
  synth→real over median NN distance real→real. ~1 is on-manifold; >>1 is off.

Acceptance: unit tests on synthetic fixtures with known answers (a deliberately
clipped array reports a high band fraction; noise reports a high NN ratio).

## Task A3: scorecard driver

`GANs/quality/scorecard.py::build_scorecard(fixture, *, types=None) -> DataFrame`

For each registered `(type, backend)`:
1. resolve backend; if unavailable, emit a row with `available=False` and skip.
2. fit on the fixture with a small epoch budget; time it.
3. generate `n_real` samples with a fixed seed.
4. contract checks: finite, label coverage, save/load round-trip.
5. fidelity via `diagnostics.summarize_real_vs_synthetic`.
6. manifold metrics (A2), utility probe (A1).
7. never raise — a failing variant becomes a row with the failure recorded.

That last point matters: the goal is a complete table including the broken ones.

`render_markdown(df)` writes `docs/GAN_SCORECARD.md`.

Acceptance: `tests/test_scorecard.py` runs the driver over every registered pair
on a tiny fixture (few hundred rows, 1–2 epochs) and asserts every pair produces
a row, no exception escapes, and the columns are populated.

## Task B1: run and commit the baseline

Run on a realistic fixture (real feature matrix from the NNNC pipeline, not
synthetic noise) and commit `docs/GAN_SCORECARD.md`.

**This is the deliverable that answers the original question.** Everything after
is driven by it.

## Task C1: CTAB z-band output clipping — CANCELLED

The plan required verifying against `mlx_ctab_helpers.transform_vgm` before
transplanting the WGAN call site. That check killed the task: CTAB already clips
per column to the training-time `[min, max]` (`mlx_ctab_helpers.py:35`), which is
a stronger bound than ±4σ. No work needed. See spec G3 (retracted).

Replacement work, done: `manifold.bound_saturation` — saturation measured in
decoded-value space so it detects BOTH bounding mechanisms. `clip_band_fraction`
alone would report ~0 for every CTAB row regardless of how saturated it was.

## Task D — driven by Phase B (now populated)

Baseline: `docs/GAN_SCORECARD.md`. Ordered by evidence strength.

**D0 — STRENGTHEN THE UTILITY PROBE (blocks any utility-driven work).** 3-seed
replication showed 10 of 11 variants FLIP SIGN on delta_val_mcc. The probe cannot
currently resolve the effect. Raise n_synth well above 300, average over repeats
inside the probe, enlarge the fixture, and re-verify that a known-bad synth
(random labels) still reads clearly negative while the seed spread on a
known-good one closes. Until this lands, NO decision may cite the utility column.

**D1 — measure POST-FILTER fidelity alongside raw.** Highest value. The scorecard
measures raw `generate()`; production consumes output filtered by density (:322),
discriminator (:342) and autoencoder in `balance.py`, and GAN_TODO §5's numbers
are post-filter. Until both are in the table, scorecard rows cannot be compared
against any historical finding. This also reframes D2.

**D2 — WGAN-MLX joint structure. DIAGNOSED 2026-08-26: NOT a hyperparameter.**
Two controlled within-variant sweeps on the real fixture, both REFUTED:

    gp_weight   σ_ratio   max Δcorr   NN     Δμ/σ
            2     0.380       0.982  1.588  1.378   <- WGAN-TF's value
           10     0.400       1.151  1.712  1.320
           50     0.428       1.016  1.713  2.204   <- current

    critic_lr_ratio   σ_ratio   max Δcorr   NN     Δμ/σ
               0.25     0.386       0.999  1.894  1.703   <- current (TTUR)
               0.50     0.465       0.848  2.071  2.930
               1.00     0.448       0.973  2.118  2.618   <- TF-equivalent

Δcorr is flat-to-noisy across both; NN gets WORSE without TTUR. The gp_weight
hypothesis came from a CROSS-VARIANT correlation (TF 2.0 / MT-TF 10.0 / MLX 50.0
mapping to best/worse/worst) with n=3 and no controls -- the controlled test kills
it. `df_wgan_gp.py:933` does carry the comment "Reduced from 10.0 to prevent
gradient penalty from dominating", so the fix was real FOR TF; it just does not
transfer.

ARCHITECTURE ALSO REFUTED (2026-08-26). Forcing TF to the same MLP generator
leaves it just as good, so an MLP can clearly learn this joint structure:

    config                      σ_ratio   max Δcorr    NN    Δμ/σ
    WGAN-TF default (CNN)         0.788       0.214  1.055   0.258
    WGAN-TF architecture="mlp"    0.814       0.239  1.048   0.169
    WGAN-MLX (MLP-only)           0.385       1.270  1.320   1.281

MLX at the SAME architecture is ~5x worse on Δcorr and half the dispersion.
THREE hypotheses now refuted (gp_weight, TTUR, architecture), every one formed by
READING CODE and comparing implementations. STOP HYPOTHESISING FROM SOURCE.

NEXT STEP IS INSTRUMENTATION, not another guess: log per-epoch generator output
std and max Δcorr for TF-MLP and MLX-MLP on identical data, and find WHERE they
diverge -- early (init / input scaling) or late (collapse during training). That
localises the fault instead of proposing another candidate. Only then propose a
fix.

Superseded note -- ARCHITECTURE INVENTORY (still true, just not the cause). WGAN-TF offers four generator
architectures (BASELINE / CNN / DCGAN / MLP) and defaults to a Conv1D residual
CNN (`wgangp_gen_cnn`). WGAN-MLX has ZERO architecture options -- grep for
`architecture|Conv` returns 0; it is a pure nn.Linear MLP. So the WGAN parity gap
is an absent architecture on the MLX side, which is a port rather than a tweak.
NOTE this inverts the framing the work began with: here MLX is missing what TF
has. Confirm by forcing TF to architecture="mlp" and checking its Δcorr degrades
toward MLX's before committing to the port.

ALSO FOUND: `gp_weight` is not reachable through the MLX path at all --
`balance_with_wgan_mlx` has no such parameter, so WGAN-MLX is hardwired to 50.0.
A configurability gap independent of whether 50.0 is the right value.

**D2 (original text) — WGAN-MLX joint structure.** WGAN-MLX max dcorr 1.141 and worst dmu 2.212
against TF's 0.211 / 0.229 on identical data. MLX is the WEAKER implementation
here, inverting the premise that TF is the laggard. Diagnose before fixing:
TTUR / critic-LR ratio and the absence of EMA in `df_wgan_mlx.py` are the
candidates. Same check for MT_WGAN-MLX (dcorr 1.030 vs TF 0.559).

**D3 — CGAN MLX backend.** The only type with no MLX path. Whether it is worth
building depends on whether CGAN is still used; decide before implementing.

**D4 — coverage holes in the threshold suite** (WGAN-MLX, MT_WGAN-MLX, MT_DDPM,
CGAN). Mechanical once D1/D2 settle what the thresholds should be.

NOT on the list: DDPM raw-output dispersion. Established as neither
under-training nor under-capacity, and the AE filter exists downstream to handle
it — revisit only if D1 shows it survives filtering.

## Task E — deferred

E1 lifecycle unification (WGAN/MT_WGAN gain `fit/generate/save/load`;
`load_from` aliased to `load`). E2 the TF-DDPM decision. Both need their own
spec section; neither blocks A–D.

## Risks

- **R1 — fixture realism.** Fidelity metrics on synthetic-noise fixtures are
  meaningless. Task B1 must use a real feature matrix.
- **R2 — runtime.** 11 pairs × fit is slow. Keep the epoch budget minimal for
  the contract test; the committed baseline can afford a longer run.
- **R3 — the utility probe is a proxy.** It ranks; it does not settle P&L.
  Anything it flags still needs the GAN_TODO §5 powered-A/B protocol before a
  production claim.
- **R4 — silent no-ops.** This repo has a documented history of changes that run
  cleanly while doing nothing (inert hyperopt params, stale JSON overrides, cache
  keys). Every task above has an acceptance check that would fail if the change
  had no effect.
