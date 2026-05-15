"""
MT_WGAN Normalization Audit — Smoke Test
2026-05-15

Empirically verifies whether MT_WGAN's _postprocess output is aligned with
real data statistics, and compares against MT_DDPM as a known-correct control.

Run from the user_data/strategies/ directory:
    python -m GANs.tests.audit_mt_wgan_normalization

Or directly:
    cd user_data/strategies && python GANs/tests/audit_mt_wgan_normalization.py
"""

from __future__ import annotations

import sys
import os

# Allow running directly from strategies/ or from GANs/tests/
_HERE = os.path.dirname(os.path.abspath(__file__))
_STRATEGIES = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _STRATEGIES not in sys.path:
    sys.path.insert(0, _STRATEGIES)

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data: intentionally varied feature scales.
#
# Feature 0: near-zero scale, σ ≈ 1.0,  μ ≈ 0.0      (unit-scale)
# Feature 1: large scale,    σ ≈ 1e6,   μ ≈ 5e6       (millions)
# Feature 2: bounded,        σ ≈ 0.5,   μ ≈ 0.0       ([-1,1]-ish)
# Feature 3: positive,       σ ≈ 20,    μ ≈ 100       (volume-like)
# Feature 4: tiny,           σ ≈ 0.001, μ ≈ 0.002     (bps-like)
# ─────────────────────────────────────────────────────────────────────────────
_N_ROWS = 10_000
_SEQ_LEN = 4
_NUM_FEATURES = 5
_EPOCHS_WGAN = 30    # enough to train but quick
_EPOCHS_DDPM = 50


def make_synthetic_data(n: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Return float32 array of shape (n, seq_len, 5) with varied scales."""
    f0 = rng.normal(0.0, 1.0, (n, seq_len, 1)).astype(np.float32)
    f1 = rng.normal(5e6, 1e6, (n, seq_len, 1)).astype(np.float32)
    f2 = np.clip(rng.normal(0.0, 0.5, (n, seq_len, 1)), -1.0, 1.0).astype(np.float32)
    f3 = rng.normal(100.0, 20.0, (n, seq_len, 1)).astype(np.float32)
    f4 = rng.normal(0.002, 0.001, (n, seq_len, 1)).astype(np.float32)
    return np.concatenate([f0, f1, f2, f3, f4], axis=-1)


def make_labels(n: int, rng: np.random.Generator) -> dict:
    """Return a simple single-task label dict (trading: 3 classes, roughly balanced)."""
    raw = rng.integers(0, 3, n)
    onehot = np.zeros((n, 3), dtype=np.float32)
    onehot[np.arange(n), raw] = 1.0
    return {"trading": onehot}


def stats_table(name: str, data: np.ndarray) -> str:
    """Return a formatted per-feature stats string for (N, seq_len, F) or (N, F) data."""
    if data.ndim == 3:
        flat = data.reshape(-1, data.shape[-1])
    else:
        flat = data
    F = flat.shape[1]
    lines = [f"\n  [{name}]  shape={data.shape}"]
    lines.append(f"  {'Feature':>10}  {'mean':>12}  {'std':>12}  {'min':>12}  {'max':>12}")
    lines.append("  " + "-" * 64)
    for f in range(F):
        col = flat[:, f]
        lines.append(
            f"  {f:>10d}  {col.mean():>12.4g}  {col.std():>12.4g}  "
            f"{col.min():>12.4g}  {col.max():>12.4g}"
        )
    return "\n".join(lines)


def ratio_table(real: np.ndarray, synth: np.ndarray, label: str) -> str:
    """Print mean/std ratio (synth/real) per feature — 1.0 = perfect alignment."""
    if real.ndim == 3:
        real_flat = real.reshape(-1, real.shape[-1])
        synth_flat = synth.reshape(-1, synth.shape[-1])
    else:
        real_flat, synth_flat = real, synth
    F = real_flat.shape[1]
    lines = [f"\n  [{label}] synth/real mean and std ratios  (1.0 = perfect)"]
    lines.append(f"  {'Feature':>10}  {'mean_real':>12}  {'mean_synth':>12}  "
                 f"{'ratio_mean':>12}  {'std_real':>12}  {'std_synth':>12}  {'ratio_std':>12}")
    lines.append("  " + "-" * 90)
    for f in range(F):
        rm, rs = real_flat[:, f].mean(), real_flat[:, f].std()
        sm, ss = synth_flat[:, f].mean(), synth_flat[:, f].std()
        ratio_m = sm / rm if abs(rm) > 1e-12 else float("nan")
        ratio_s = ss / rs if abs(rs) > 1e-12 else float("nan")
        lines.append(
            f"  {f:>10d}  {rm:>12.4g}  {sm:>12.4g}  {ratio_m:>12.4f}  "
            f"{rs:>12.4g}  {ss:>12.4g}  {ratio_s:>12.4f}"
        )
    return "\n".join(lines)


def run_mt_wgan_experiment(data: np.ndarray, labels: dict, epochs: int):
    """
    Trains MT_WGAN on data, returns:
      (raw_gen_output, postprocess_output)
    raw_gen_output is what self.gen(z, c) produces BEFORE _postprocess.
    """
    from GANs.df_mt_wgan_mlx import MTWGANMLX, balance_with_mt_wgan_mlx
    import mlx.core as mx

    n, seq_len, num_features = data.shape
    task_label_dims = {task: lab.shape[1] for task, lab in labels.items()}

    print(f"\n{'='*70}")
    print(f"  Training MT_WGAN ({epochs} epochs, seq_len={seq_len}, F={num_features})")
    print(f"{'='*70}")

    # Use _return_model to get the trained GAN object back
    _, _, gan = balance_with_mt_wgan_mlx(
        data,
        labels,
        epochs=epochs,
        batch_size=256,
        task_target_ratios=None,
        verbose=True,
        _return_model=True,
    )

    # Generate a sample matching the label distribution of real data
    n_gen = 2000
    # Sample label rows from real distribution
    rng_idx = np.random.default_rng(99)
    idx = rng_idx.integers(0, n, n_gen)
    sampled_labels = {task: labels[task][idx] for task in sorted(labels.keys())}

    sorted_tasks = sorted(sampled_labels.keys())
    import mlx.core as mx
    c_parts = [mx.array(sampled_labels[t], dtype=mx.float32) for t in sorted_tasks]
    c = mx.concatenate(c_parts, axis=-1)
    z = mx.random.normal((n_gen, gan.latent_dim))

    # Raw generator output BEFORE _postprocess
    raw_gen = gan.gen(z, c)
    mx.eval(raw_gen)
    raw_gen_np = np.array(raw_gen)

    # Apply _postprocess
    post_gen = gan._postprocess(mx.array(raw_gen_np))
    mx.eval(post_gen)
    post_gen_np = np.array(post_gen)

    return raw_gen_np, post_gen_np, gan


def run_mt_ddpm_experiment(data: np.ndarray, labels: dict, epochs: int):
    """
    Trains MT_DDPM on data (known-correct control), returns generated samples.
    """
    from GANs.df_mt_ddpm_mlx import MTDDPMMLX

    n, seq_len, num_features = data.shape
    task_label_dims = {k: int(v.shape[1]) for k, v in labels.items()}

    print(f"\n{'='*70}")
    print(f"  Training MT_DDPM ({epochs} epochs, seq_len={seq_len}, F={num_features})")
    print(f"{'='*70}")

    model = MTDDPMMLX(
        seq_len=seq_len,
        num_features=num_features,
        task_label_dims=task_label_dims,
        epochs=epochs,
        batch_size=256,
        d_model=128,
        d_layers=2,
        verbose=True,
    )
    model.fit(data, labels)

    # Generate using matching labels
    n_gen = 2000
    rng_idx = np.random.default_rng(99)
    idx = rng_idx.integers(0, n, n_gen)
    sampled_labels = {task: labels[task][idx] for task in sorted(labels.keys())}

    synth = model.generate(n_gen, sampled_labels)
    return synth


def main():
    rng = np.random.default_rng(42)
    data = make_synthetic_data(_N_ROWS, _SEQ_LEN, rng)
    labels = make_labels(_N_ROWS, rng)

    print("\n" + "=" * 70)
    print("  AUDIT: MT_WGAN Normalization Smoke Test")
    print("  2026-05-15")
    print("=" * 70)

    print("\n[A] REAL DATA DISTRIBUTION")
    print(stats_table("real_data", data))

    # ── MT_WGAN ──────────────────────────────────────────────────────────────
    raw_wgan, post_wgan, gan = run_mt_wgan_experiment(data, labels, _EPOCHS_WGAN)

    print("\n[B] MT_WGAN RESULTS")
    print(stats_table("mt_wgan: raw generator output (before _postprocess)", raw_wgan))
    print(stats_table("mt_wgan: _postprocess output (what callers receive)", post_wgan))

    print("\n[B1] MT_WGAN alignment: _postprocess vs real")
    print(ratio_table(data, post_wgan, "MT_WGAN _postprocess vs real"))

    print("\n[B2] feature_mean / feature_std stored in model:")
    print(f"  feature_mean: {gan.feature_mean}")
    print(f"  feature_std:  {gan.feature_std}")
    print(f"  feature_min:  {gan.feature_min}")
    print(f"  feature_max:  {gan.feature_max}")

    print("\n[B3] raw generator output stats (expect ~ N(0,1) if Tanh limits range):")
    raw_flat = raw_wgan.reshape(-1, _NUM_FEATURES)
    for f in range(_NUM_FEATURES):
        col = raw_flat[:, f]
        print(f"  Feature {f}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}")

    # ── MT_DDPM (control) ─────────────────────────────────────────────────────
    synth_ddpm = run_mt_ddpm_experiment(data, labels, _EPOCHS_DDPM)

    print("\n[C] MT_DDPM (CONTROL) RESULTS")
    print(stats_table("mt_ddpm: generate() output (what callers receive)", synth_ddpm))
    print(ratio_table(data, synth_ddpm, "MT_DDPM vs real (control — expect close to 1.0)"))

    # ── Summary comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: Scale alignment — synth mean / real mean per feature")
    print("  (values near 1.0 = good alignment; huge deviations = bug)")
    print("=" * 70)
    print(f"\n  {'Feature':>10}  {'real_mean':>12}  {'wgan_post_mean':>16}  "
          f"{'wgan_ratio':>12}  {'ddpm_mean':>12}  {'ddpm_ratio':>12}")
    print("  " + "-" * 80)

    data_flat = data.reshape(-1, _NUM_FEATURES)
    wgan_flat = post_wgan.reshape(-1, _NUM_FEATURES)
    ddpm_flat = synth_ddpm.reshape(-1, _NUM_FEATURES)

    wgan_buggy = False
    for f in range(_NUM_FEATURES):
        rm = data_flat[:, f].mean()
        wm = wgan_flat[:, f].mean()
        dm = ddpm_flat[:, f].mean()
        wr = wm / rm if abs(rm) > 1e-12 else float("nan")
        dr = dm / rm if abs(rm) > 1e-12 else float("nan")
        flag = ""
        if not (0.5 < abs(wr) < 2.0):
            flag = " *** SCALE MISMATCH ***"
            wgan_buggy = True
        print(f"  {f:>10d}  {rm:>12.4g}  {wm:>16.4g}  {wr:>12.4f}  {dm:>12.4g}  {dr:>12.4f}{flag}")

    print("\n" + "=" * 70)
    if wgan_buggy:
        print("  VERDICT: MT_WGAN _postprocess output is MISALIGNED with real data.")
        print("  The ratio(s) far from 1.0 confirm a normalization bug.")
        print("  MT_WGAN applies inverse-z-score in _postprocess but never applies")
        print("  the forward z-score in fit(). The generator trained on raw data,")
        print("  so its Tanh output is in [-1, 1], but _postprocess multiplies by")
        print("  feature_std and adds feature_mean — mapping the [-1,1] range")
        print("  to an incorrect scale.")
    else:
        print("  VERDICT: MT_WGAN output appears scale-aligned with real data.")
        print("  (Normalization may be OK, or this test did not expose the issue.)")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Pytest regression tests for the forward z-score normalisation fix.
#
# These tests verify that the fix is applied correctly at the algebraic level
# and that postprocessed output is on the correct scale.  They do NOT assert
# full distributional convergence — that requires longer training and is
# limited by the generator's Tanh activation (see note below).
#
# Key regression: before the fix, _postprocess output for large-scale features
# (e.g. σ_real ≈ 1e6) would be bounded in [-1, 1] with near-zero std.  After
# the fix, postprocessed output must be in the correct real-data scale range
# regardless of generator convergence.
#
# KNOWN STRUCTURAL LIMIT (not addressed in this patch):
#   The generator's Tanh activation bounds output to (-1, 1) in z-scored
#   space, which corresponds to approximately ±1σ in real-data space.
#   Generator mode collapse (constant -1 or +1) for specific features is
#   a pre-existing architectural instability separate from the z-score fix.
#   The tests below are tolerant of mode collapse while still catching the
#   original pre-fix pathology (wrong scale by factor of std).
#   See 2026-05-15-mt-wgan-audit-findings.md for full details.
# ─────────────────────────────────────────────────────────────────────────────

import pytest

_EPOCHS_PYTEST = 60   # more epochs than the visual demo for stable convergence
_N_GEN = 2000


@pytest.fixture(scope="module")
def trained_mt_wgan():
    """Train a small MT_WGAN and return (trained_gan, real_data, real_labels)."""
    rng = np.random.default_rng(42)
    data = make_synthetic_data(_N_ROWS, _SEQ_LEN, rng)
    labels = make_labels(_N_ROWS, rng)

    from GANs.df_mt_wgan_mlx import balance_with_mt_wgan_mlx

    _, _, gan = balance_with_mt_wgan_mlx(
        data,
        labels,
        epochs=_EPOCHS_PYTEST,
        batch_size=256,
        task_target_ratios=None,
        verbose=False,
        _return_model=True,
    )
    return gan, data, labels


def _generate_synth(gan, real_data, real_labels, n_gen=_N_GEN):
    """Generate n_gen samples from a trained MT_WGAN using the real label distribution."""
    import mlx.core as mx

    rng_idx = np.random.default_rng(99)
    n = real_data.shape[0]
    idx = rng_idx.integers(0, n, n_gen)
    sampled_labels = {task: real_labels[task][idx] for task in sorted(real_labels.keys())}

    sorted_tasks = sorted(sampled_labels.keys())
    c_parts = [mx.array(sampled_labels[t], dtype=mx.float32) for t in sorted_tasks]
    c = mx.concatenate(c_parts, axis=-1)
    z = mx.random.normal((n_gen, gan.latent_dim))

    synth = gan.gen(z, c)
    synth = gan._postprocess(synth)
    mx.eval(synth)
    return np.array(synth)


def test_feature_stats_stored_on_model(trained_mt_wgan):
    """After training, feature_mean and feature_std must be set and have correct shapes.

    This is the minimal sanity check that the stats computation path ran.
    """
    gan, real_data, _ = trained_mt_wgan
    assert gan.feature_mean is not None, "feature_mean should be set after training"
    assert gan.feature_std is not None, "feature_std should be set after training"
    assert gan.feature_mean.shape == (_NUM_FEATURES,), (
        f"feature_mean shape {gan.feature_mean.shape} != ({_NUM_FEATURES},)"
    )
    assert gan.feature_std.shape == (_NUM_FEATURES,), (
        f"feature_std shape {gan.feature_std.shape} != ({_NUM_FEATURES},)"
    )
    # Stats must reflect the real data distribution (not z-scored data)
    real_flat = real_data.reshape(-1, _NUM_FEATURES)
    for f in range(_NUM_FEATURES):
        np.testing.assert_allclose(
            gan.feature_mean[f], real_flat[:, f].mean(), rtol=1e-3,
            err_msg=f"feature_mean[{f}] should be the raw data mean"
        )
        np.testing.assert_allclose(
            gan.feature_std[f], real_flat[:, f].std(), rtol=1e-3,
            err_msg=f"feature_std[{f}] should be the raw data std"
        )


def test_postprocess_output_on_correct_scale(trained_mt_wgan):
    """Postprocessed output must be on the correct real-data scale.

    The pre-fix pathology: large-scale features (e.g. σ_real ≈ 1e6) would
    appear in [-1, 1] range because postprocess was inverting a non-applied
    transform.  After the fix, postprocessed output must be in the same order
    of magnitude as real data for every feature.

    Test: |postprocess_mean - real_mean| < 2 * real_std for each feature.
    This is very generous (allows ±2σ offset from mode collapse) but catches
    the pre-fix bug where Feature 1 (real_mean=4.99e6, real_std=1e6) would
    appear with postprocess_mean near 0.09 (i.e. 4.99e6σ off).
    """
    gan, real_data, real_labels = trained_mt_wgan
    synth = _generate_synth(gan, real_data, real_labels)

    real_flat = real_data.reshape(-1, _NUM_FEATURES)
    synth_flat = synth.reshape(-1, _NUM_FEATURES)

    for f in range(_NUM_FEATURES):
        real_mean = real_flat[:, f].mean()
        real_std = real_flat[:, f].std()
        synth_mean = synth_flat[:, f].mean()
        # Tolerance is 2 * real_std — wide enough to absorb mode collapse
        # (generator stuck at Tanh=-1 gives synth_mean = real_mean - real_std)
        # but tight enough to catch the pre-fix pathology.
        tol = 2.0 * real_std
        assert abs(synth_mean - real_mean) <= tol, (
            f"Feature {f}: postprocess output mean {synth_mean:.4g} is "
            f"{abs(synth_mean - real_mean) / real_std:.1f}σ from real mean {real_mean:.4g}.  "
            f"Tolerance is ±{tol:.4g} (2σ).  "
            f"This suggests the z-score inversion is using wrong scale — "
            f"check that feature_std reflects raw data std."
        )


def test_postprocess_is_algebraic_inverse_of_zscore(trained_mt_wgan):
    """_postprocess must exactly invert a z-scored input.

    Directly verifies the algebraic correctness of the transform pair:
    forward z-score (x - mean) / std → _postprocess → x.
    This is unit-level: independent of generator training quality.
    """
    import mlx.core as mx

    gan, real_data, _ = trained_mt_wgan

    # Take a slice of real data and z-score it manually
    sample = real_data[:100].astype(np.float32)
    zscored = (sample - gan.feature_mean) / gan.feature_std  # (100, seq_len, F)

    # _postprocess should invert this back to the original
    recovered = gan._postprocess(mx.array(zscored))
    mx.eval(recovered)
    recovered_np = np.array(recovered)

    np.testing.assert_allclose(
        recovered_np, sample, rtol=1e-4, atol=1e-3,
        err_msg=(
            "_postprocess(zscore(x)) != x — the forward z-score and _postprocess "
            "are not algebraic inverses.  Check feature_mean/feature_std assignment."
        )
    )
