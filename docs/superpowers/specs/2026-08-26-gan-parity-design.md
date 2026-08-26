# GAN Family Parity & Quality Assessment (Design)

Status: draft — 2026-08-26
Related: `docs/GAN_TODO.md` §5 (staged sample-quality plan, Phases 0–2 DONE for
TabDDPM/MT_DDPM), `docs/superpowers/specs/2026-05-11-tabddpm-design.md`.

## 0. Problem

The TabDDPM line has been through a full measure → diagnose → fix → re-measure
cycle (GAN_TODO §5): fidelity reported per class (σ_syn/σ_real, OFF_DIST buckets,
worst-feature |Δμ|), then P&L-gated with paired seeds at a *powered* operating
point. That process produced real conclusions — NNNC TabDDPM synth is already
high-fidelity and the ceiling is the target, not the GAN; MT_DDPM synth is
over-dispersed; dispersion-widening is a dead end; AE-off beats non-GAN 4/4.

**That process has never been run on the other nine implementations.** WGAN,
MT_WGAN, CTAB-GAN, MT-CTAB-GAN (each TF + MLX) and CGAN (TF) have never been
assessed for whether their synth is on-manifold or whether it helps predictions.
We do not know if they are fine, broken, or somewhere between.

This spec covers building that assessment and running it — not guessing at fixes.

## 1. Goals and non-goals

### Goals

- **G1.** One comparative **scorecard** covering every registered `(GANType,
  backend)` pair on a shared fixture: fidelity *and* downstream utility in the
  same table, so variants can be ranked and regressions caught.
- **G2.** Close the assessment coverage holes: WGAN-MLX, MT_WGAN-MLX, MT_DDPM,
  CGAN are absent from the quality suite today (7 of 11 covered).
- **G3.** Fix the one unambiguous correctness gap found in review: no CTAB
  variant (TF or MLX, single or multi-task) clips generator output to the
  z-score band, which is the crash risk `e44662f` fixed for WGAN.
- **G4.** Bring TF and MLX to functional parity **as peers**. TF is the
  supported path for non-Mac users of this open-source repo; it is not legacy.
- **G5.** Unify the lifecycle so the interface layer stops special-casing:
  WGAN/MT_WGAN are function-based (`balance_with_*`, no `fit()`); everything
  else is model-based (`fit/generate/save/load`). Naming also drifted
  (`load` vs `load_from`).

### Non-goals

- **NG1.** Porting EDM / Min-SNR-γ to the GAN families. These are
  *diffusion-specific* parameterisations of a noise schedule; WGAN and CTAB have
  no schedule. The portable half of the TabDDPM work is output clipping, EMA
  weights and best-loss checkpoint restore.
- **NG2.** Deleting TF implementations, or treating the TF/MLX line-count gap
  (`df_ctab_gan.py` 2283 vs `df_ctab_gan_mlx.py` 724) as bloat to prune before
  auditing what it is.
- **NG3.** A TF DDPM. Deferred to an explicit decision after the scorecard.
- **NG4.** Chasing P&L. GAN_TODO §4/§5 establish that for NNNC the ceiling is
  the target/features, not synth quality. This work is justified as correctness,
  coverage and maintainability. Any P&L gain is a bonus, not the thesis.

## 2. Decisions locked in

- **D1 — comparative, not pass/fail.** The existing suite asserts thresholds
  (`test_mean_rmse_below_threshold`). That answers "did this clear a bar", never
  "which variants are good and where does each fail". The scorecard emits ranked
  numbers; threshold tests stay as regression guards.
- **D2 — both axes in one table.** Fidelity alone is the mistake this codebase
  already made once: TabDDPM fidelity was fixed and augmentation still matched
  no-GAN. Every scorecard row carries a fidelity block *and* a utility number.
- **D3 — utility proxy is cheap and comparable, not a backtest.** Δval_mcc on a
  fixed split, real vs real+synth, same classifier and seed. Full powered-A/B
  P&L (GAN_TODO §5 protocol, paired seeds, loosened guards) is reserved for
  variants the scorecard flags as worth it — it is far too slow for 11 cells.
- **D4 — reuse, don't rebuild.** `diagnostics.summarize_real_vs_synthetic`
  already computes marginal/joint/temporal fidelity with flags;
  `tests/quality_base.GANQualityMixin` already builds fixtures and metrics. The
  scorecard is a driver over those, not a second metrics engine.
- **D5 — TF and MLX are peers.** Every type that has both backends gets both
  rows. A fix landing in one backend and not the other is an incomplete fix.
- **D6 — the scorecard is an artifact, not just a test.** Committed baseline
  output, so drift is a reviewable diff rather than a rerun-and-squint.

## 3. Architecture and integration points

### Files created

- `GANs/quality/scorecard.py` — driver: enumerate registered backends, fit each
  on the shared fixture, generate, compute fidelity via `diagnostics`, compute
  the utility proxy, emit a table (markdown + JSON).
- `GANs/quality/utility_probe.py` — the Δval_mcc proxy; classifier-agnostic,
  takes `(real_X, real_y, synth_X, synth_y)` and returns the delta.
- `GANs/quality/__init__.py`
- `GANs/tests/test_scorecard.py` — the driver runs end-to-end on a tiny fixture
  for every registered pair (contract test, not a quality assertion).
- `docs/GAN_SCORECARD.md` — committed baseline output (D6).

### Files modified

- `GANs/df_ctab_gan_mlx.py`, `df_mt_ctab_gan_mlx.py`, `df_ctab_gan.py`,
  `df_mt_ctab_gan.py` — add `_ZSCORE_CLIP` output clipping (G3).
- `GANs/tests/test_quality_suite.py` — add the four missing configs (G2).
- `GANs/backends/*.py` — only if the scorecard needs a uniform accessor.

### Files unchanged

- `diffusion_mlx.py`, `diffusion_edm_mlx.py`, `df_tabddpm_mlx.py`,
  `df_mt_ddpm_mlx.py` — the reference line is not touched by this work.
- `balance.py`, `GANInterface.py` — lifecycle unification (G5) is a later phase
  with its own spec section; not in the first implementation pass.

### No new pip dependencies

The utility probe uses the classifier already present in the repo.

## 4. Scorecard contents

Per `(GANType, backend)` row:

| block | fields |
|---|---|
| identity | type, backend, available?, fit seconds |
| fidelity — marginal | worst-feature \|Δμ\|/σ, median σ_syn/σ_real, OFF_DIST bucket count |
| fidelity — joint | max \|Δcorr\|, mean \|Δcorr\| |
| fidelity — temporal | max \|Δautocorr\| (3-D types only; blank for tabular) |
| manifold | fraction of synth pinned at the clip band; NN-distance ratio |
| utility | Δval_mcc (real+synth − real), same seed and split |
| contract | finite output, label coverage, save/load round-trip |

The manifold block is the piece the TabDDPM work needed and the generic suite
lacks: "σ saturating the ±4σ clip" was the MT_DDPM diagnosis and is invisible in
RMSE-style metrics.

## 5. Tests

- `test_scorecard.py` — every registered pair completes the driver on a tiny
  fixture. This is a *contract* test: it catches a backend that silently
  produces nothing, which review found is a live failure mode in this repo.
- Existing `test_quality_suite.py` threshold tests remain, extended to the four
  uncovered pairs.
- `test_mlx_tf_parity.py` currently covers CTAB preprocessing/eval helpers only.
  Extend with a generator-behaviour parity check once the scorecard shows what
  differs.

## 6. Phasing

- **Phase A** — scorecard driver + utility probe + contract test (G1, G2).
- **Phase B** — run it; commit `GAN_SCORECARD.md` baseline. **Deliverable: the
  first answer to "do these GANs produce good output".**
- **Phase C** — CTAB z-clip fix (G3), verified by the scorecard's manifold block.
- **Phase D** — fixes driven by what Phase B shows, per variant, both backends.
- **Phase E** — lifecycle unification (G5) and the TF-DDPM decision (NG3).

Phases A–C are independent of what the scorecard finds and can be implemented
now. Phase D deliberately has no content until Phase B has run.
