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

## Task C1: CTAB z-band output clipping

`df_ctab_gan_mlx.py`, `df_mt_ctab_gan_mlx.py`, `df_ctab_gan.py`,
`df_mt_ctab_gan.py`: add `_ZSCORE_CLIP: float = 4.0` and clip generator output
before the inverse transform, mirroring `WGANMLX.generate()` (`df_wgan_mlx.py`
:172-178) and `TabDDPMMLX.generate()`.

Note the CTAB path de-normalises through the VGM/BGM inverse rather than a plain
z-score, so the clip applies to the *scalar* component in z-space before
`inverse_transform`. Verify against `mlx_ctab_helpers.transform_vgm` — do not
assume the WGAN call site transplants unchanged.

Acceptance: scorecard `clip_band_fraction` becomes reportable for CTAB rows (it
is undefined today); existing CTAB quality tests unchanged; a regression test
that a generator emitting extreme values yields finite de-normalised output.

## Task D — driven by Phase B

Intentionally empty. Populated after the scorecard runs, one variant at a time,
each fix landing in BOTH backends (spec D5) and re-verified by re-running the
scorecard.

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
