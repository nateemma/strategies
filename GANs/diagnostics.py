"""
GAN fidelity diagnostics for cross-task balanced augmentation.

When ``balance_multi_task`` finishes a run we may have generated tens or
hundreds of thousands of synthetic samples and concatenated them with
real data — at which point the question "are these samples actually
useful?" becomes urgent.  A trained downstream classifier that gets
worse with augmentation usually means one of four things:

  * Mode collapse — the generator produces near-identical samples for a
    given conditional, so std_synthetic ≪ std_real per feature.
  * Distributional shift — the generator's class-conditional mean
    differs from the real class-conditional mean by more than a few
    sigmas, so the model fits an off-manifold blob.
  * Conditional mis-specification — the GAN was asked for class=2 but
    silently produced something resembling class=1's distribution.
  * Joint-structure breakage — per-feature marginals look right but
    feature *correlations* don't match real data.  This is the failure
    mode we wouldn't catch with marginal stats alone, and the one that
    surfaced when ratio=0.8 augmentation regressed training even though
    every (task, class) bucket reported "ok" on marginals.

This module computes per ``(task, class)`` fidelity stats from the
concatenated real and synthetic data the balancer already has on hand.
No extra GAN calls or saved data are needed.

Output is plain text via a caller-provided ``log`` callable (no plots,
no files), so it integrates with the existing strategy logging without
introducing a matplotlib dependency.

The stats produced per (task, class):

Marginal (per-feature):
  * ``N_real``, ``N_synth`` — sample counts.
  * ``mean_shift_max`` — max ``|μ_syn − μ_real| / σ_real`` across
    features.  Flags off-distribution generation.
  * ``std_ratio_min`` — min ``σ_syn / σ_real`` across features.  Values
    well below 1.0 (say <0.3) indicate mode collapse on at least one
    feature.
  * ``std_ratio_mean`` — average diversity of synthetic samples
    relative to real, aggregated across features.

Joint (correlation matrix):
  * ``corr_max_abs_diff`` — max ``|ρ_real(i,j) − ρ_synth(i,j)|`` over
    all feature pairs.  Catches "marginals look right but joint
    structure is wrong".
  * ``corr_mean_abs_diff`` — average disagreement across pairs.

Temporal (lag-1 autocorrelation, 3D inputs only):
  * ``ac_max_abs_diff`` — max ``|ρ_lag1_real(f) − ρ_lag1_synth(f)|``
    across features.  Catches generators that produce noisy
    step-to-step output where real data evolves smoothly (think
    bb_width or EMA features that should be near-constant within a
    16-step window but the GAN generates jittery sequences).
  * ``ac_mean_abs_diff`` — average disagreement across features.
  Skipped silently for 2D / DataFrame inputs (no temporal axis).

When a class has no synthetic samples (e.g. it was already at target
and the balancer never picked it), the row prints "-" placeholders so
the table is still legible.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# How many "worst features" to print per (task, class).
_WORST_FEATURES_PER_BUCKET: int = 3

# How many "worst pairs" to print per (task, class) in the joint drilldown.
_WORST_PAIRS_PER_BUCKET: int = 3

# How many (task, class) buckets to drill into with worst-feature lines.
_DRILLDOWN_BUCKETS: int = 5

# Cap rows used for correlation estimation per bucket — corrcoef on
# (5M, 25) is fine but (5M, 25) × 18 buckets adds up.  100k rows is
# plenty for stable correlation estimates and keeps the diagnostic
# fast (sub-second per bucket).
_DEFAULT_MAX_SAMPLES_FOR_CORR: int = 100_000

# Below this synth count, correlation estimates are too noisy to be
# meaningful — print "-" instead of a misleading number.
_MIN_SAMPLES_FOR_CORR: int = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_real_vs_synthetic(
    real_data: Any,
    real_labels: Dict[str, np.ndarray],
    synth_data: Any,
    synth_labels: Dict[str, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    log: Any = print,
    max_samples_for_corr: int = _DEFAULT_MAX_SAMPLES_FOR_CORR,
    rng: Optional[np.random.Generator] = None,
    pair_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Print a per-(task, class) fidelity summary comparing real and
    synthetic samples that ``balance_multi_task`` produced.

    The function never raises on data quality — its job is to *report*
    problems, not enforce them.  A bad GAN should still be allowed to
    run; the operator decides what to do with the diagnostics.

    Args:
        real_data:      (N, *F) ndarray, (N, F) DataFrame, or list of
                        per-batch arrays / frames.  Mixed shapes are
                        flattened to (N_total, F_flat) per the rules in
                        ``_to_per_sample_matrix``.
        real_labels:    ``Dict[task, (N, num_classes) one-hot]``.  Must
                        align row-for-row with ``real_data``.
        synth_data:     Synthetic counterpart of ``real_data``.  Empty
                        list / 0-row arrays are tolerated and produce a
                        "no synthetic generated" message.
        synth_labels:   ``Dict[task, (M, num_classes) one-hot]``.
        feature_names:  Optional names for the F columns — used in
                        worst-feature and worst-pair lines.  Defaults
                        to ``f0..fF``.
        log:            ``print``-compatible callable.
        max_samples_for_corr: Cap on rows used for correlation-matrix
                        estimation per (task, class) bucket.  100k is
                        enough for stable estimates with F=25 features
                        and keeps the diagnostic sub-second per bucket
                        even when the full bucket has millions of rows.
        rng:            Optional ``np.random.Generator`` for the corr-
                        subsample step.  Pass a seeded one to make the
                        diagnostic reproducible (useful in tests).

    Returns:
        A dict of computed stats keyed by task → class → metric, useful
        for tests and for callers that want to assert on fidelity in CI.
        Each per-class entry includes both marginal (mean/std) stats
        and joint (correlation) stats with their own flags.
    """
    real_flat, real_label_flat = _flatten(real_data, real_labels)
    synth_flat, synth_label_flat = _flatten(synth_data, synth_labels)

    # Try to retain a 3D view for the temporal diagnostic.  Skipped
    # silently for 2D / DataFrame inputs (no temporal axis to compare).
    real_3d = _to_3d_or_none(real_data)
    synth_3d = _to_3d_or_none(synth_data)
    has_temporal = (
        real_3d is not None
        and synth_3d is not None
        and real_3d.shape[1] >= 2
        and real_3d.shape[1] == synth_3d.shape[1]
    )

    if real_flat.size == 0:
        log("    [diagnostics] no real data — skipping fidelity report")
        return {}

    if feature_names is None or len(feature_names) != real_flat.shape[1]:
        feature_names = [f"f{i}" for i in range(real_flat.shape[1])]

    if synth_flat.shape[0] == 0:
        log("    [diagnostics] no synthetic samples were generated — "
            "fidelity report skipped")
        return {}

    if rng is None:
        rng = np.random.default_rng()

    header_suffix = f" — pair={pair_name}" if pair_name else ""
    log(f"    === GAN fidelity diagnostic (real vs synthetic){header_suffix} ===")
    log(f"      real rows: {real_flat.shape[0]:>10d}    "
        f"synth rows: {synth_flat.shape[0]:>10d}    "
        f"features: {real_flat.shape[1]}")

    header = (
        f"      {'task':<10s}  {'cls':>3s}  {'N_real':>9s}  "
        f"{'N_syn':>9s}  {'|Δμ|max':>9s}  "
        f"{'σ_syn/σ_real':>14s}  flag"
    )
    log(header)
    log("      " + "-" * (len(header) - 6))

    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    drilldowns: List[Dict[str, Any]] = []
    joint_drilldowns: List[Dict[str, Any]] = []
    joint_rows: List[Dict[str, Any]] = []
    temporal_drilldowns: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []

    for task in sorted(real_label_flat.keys()):
        real_oh = real_label_flat[task]
        synth_oh = synth_label_flat.get(task)
        num_classes = real_oh.shape[1]

        real_idx = real_oh.argmax(axis=1)
        synth_idx = (
            synth_oh.argmax(axis=1) if synth_oh is not None and synth_oh.size
            else np.empty(0, dtype=np.int64)
        )

        per_class: Dict[int, Dict[str, Any]] = {}
        for c in range(num_classes):
            real_subset = real_flat[real_idx == c]
            synth_subset = synth_flat[synth_idx == c] if synth_idx.size else np.empty((0, real_flat.shape[1]))

            n_real = int(real_subset.shape[0])
            n_syn = int(synth_subset.shape[0])

            if n_syn == 0 or n_real == 0:
                log(
                    f"      {task:<10s}  {c:>3d}  {n_real:>9d}  "
                    f"{n_syn:>9d}  {'-':>9s}  {'-':>14s}  "
                    f"{'no real' if n_real == 0 else 'no synth'}"
                )
                per_class[c] = {
                    "n_real": n_real, "n_synth": n_syn,
                    "mean_shift_max": None,
                    "std_ratio_min": None, "std_ratio_mean": None,
                    "corr_max_abs_diff": None,
                    "corr_mean_abs_diff": None,
                    "ac_max_abs_diff": None,
                    "ac_mean_abs_diff": None,
                }
                continue

            stats = _per_feature_stats(real_subset, synth_subset)
            mean_shift_max = float(np.max(np.abs(stats["mean_shift"])))
            std_ratio_min = float(np.min(stats["std_ratio"]))
            std_ratio_mean = float(np.mean(stats["std_ratio"]))

            # Joint stats — subsample per bucket to keep corrcoef cost
            # bounded.  Also skip when synth N is too small for a
            # reliable estimate (the result would be noise).
            corr_stats: Optional[Dict[str, Any]] = None
            if n_syn >= _MIN_SAMPLES_FOR_CORR:
                real_for_corr = _subsample(real_subset, max_samples_for_corr, rng)
                synth_for_corr = _subsample(synth_subset, max_samples_for_corr, rng)
                corr_stats = _correlation_stats(real_for_corr, synth_for_corr)

            # Temporal stats — only when both real and synth are 3D
            # sequences with matching time dimension.  Subsample at
            # the *sequence* level (not timestep level) so each
            # sequence stays whole — autocorrelation is meaningful
            # only on contiguous timesteps.
            temporal_stats_bucket: Optional[Dict[str, Any]] = None
            if has_temporal:
                # Sample-level subsetting requires sample-level labels.
                # Real labels are (N_seq, num_classes); synth labels
                # similarly.  Index into 3D arrays directly.
                real_seq_oh = real_labels[task] if isinstance(real_labels, dict) else None
                synth_seq_oh = synth_labels[task] if isinstance(synth_labels, dict) else None
                # Synth labels for a list-of-batches input are already
                # concatenated by _flatten/_to_3d_or_none — but the user
                # may have passed real labels already at the sequence
                # level.  Detect by row count match.
                if (real_seq_oh is not None
                        and real_seq_oh.shape[0] == real_3d.shape[0]
                        and synth_seq_oh is not None
                        and synth_seq_oh.shape[0] == synth_3d.shape[0]):
                    real_seq_idx = real_seq_oh.argmax(axis=1)
                    synth_seq_idx = synth_seq_oh.argmax(axis=1)
                    real_3d_subset = real_3d[real_seq_idx == c]
                    synth_3d_subset = synth_3d[synth_seq_idx == c]
                    if (real_3d_subset.shape[0] >= _MIN_SAMPLES_FOR_CORR
                            and synth_3d_subset.shape[0] >= _MIN_SAMPLES_FOR_CORR):
                        # Subsample at sequence level for speed.
                        real_3d_subset = _subsample(
                            real_3d_subset, max_samples_for_corr, rng,
                        )
                        synth_3d_subset = _subsample(
                            synth_3d_subset, max_samples_for_corr, rng,
                        )
                        temporal_stats_bucket = _temporal_stats(
                            real_3d_subset, synth_3d_subset,
                        )

            marginal_flag = _flag_for_bucket(mean_shift_max, std_ratio_min)
            joint_flag = (
                _joint_flag(corr_stats["max_abs_diff"], corr_stats["mean_abs_diff"])
                if corr_stats else None
            )
            temporal_flag = (
                _temporal_flag(
                    temporal_stats_bucket["max_abs_diff"],
                    temporal_stats_bucket["mean_abs_diff"],
                )
                if temporal_stats_bucket else None
            )
            combined_flag = _combine_flags(marginal_flag, joint_flag)
            if temporal_flag and temporal_flag != "ok":
                combined_flag = (
                    f"{combined_flag},{temporal_flag}"
                    if combined_flag != "ok" else temporal_flag
                )

            log(
                f"      {task:<10s}  {c:>3d}  {n_real:>9d}  "
                f"{n_syn:>9d}  {mean_shift_max:>8.2f}σ  "
                f"{std_ratio_min:>5.2f}/{std_ratio_mean:>5.2f}  "
                f"{combined_flag}"
            )

            per_class[c] = {
                "n_real": n_real, "n_synth": n_syn,
                "mean_shift_max": mean_shift_max,
                "std_ratio_min": std_ratio_min,
                "std_ratio_mean": std_ratio_mean,
                "corr_max_abs_diff": corr_stats["max_abs_diff"] if corr_stats else None,
                "corr_mean_abs_diff": corr_stats["mean_abs_diff"] if corr_stats else None,
                "ac_max_abs_diff": (
                    temporal_stats_bucket["max_abs_diff"]
                    if temporal_stats_bucket else None
                ),
                "ac_mean_abs_diff": (
                    temporal_stats_bucket["mean_abs_diff"]
                    if temporal_stats_bucket else None
                ),
                "flag": combined_flag,
                "marginal_flag": marginal_flag,
                "joint_flag": joint_flag,
                "temporal_flag": temporal_flag,
            }
            drilldowns.append({
                "task": task,
                "class": c,
                "stats": stats,
                "severity": mean_shift_max + (1.0 - std_ratio_min),
            })
            if corr_stats is not None:
                joint_rows.append({
                    "task": task,
                    "class": c,
                    "n_real": n_real,
                    "n_syn": n_syn,
                    "max_abs_diff": corr_stats["max_abs_diff"],
                    "mean_abs_diff": corr_stats["mean_abs_diff"],
                    "joint_flag": joint_flag,
                })
                joint_drilldowns.append({
                    "task": task,
                    "class": c,
                    "stats": corr_stats,
                    "severity": corr_stats["max_abs_diff"]
                                + corr_stats["mean_abs_diff"],
                })
            if temporal_stats_bucket is not None:
                temporal_rows.append({
                    "task": task,
                    "class": c,
                    "max_abs_diff": temporal_stats_bucket["max_abs_diff"],
                    "mean_abs_diff": temporal_stats_bucket["mean_abs_diff"],
                    "flag": temporal_flag,
                })
                temporal_drilldowns.append({
                    "task": task,
                    "class": c,
                    "stats": temporal_stats_bucket,
                    "severity": (
                        temporal_stats_bucket["max_abs_diff"]
                        + temporal_stats_bucket["mean_abs_diff"]
                    ),
                })

        results[task] = per_class

    # Drill into the most problematic buckets (marginals).
    drilldowns.sort(key=lambda r: r["severity"], reverse=True)
    if drilldowns:
        log("")
        log(f"      worst features per bucket "
            f"(top {min(_DRILLDOWN_BUCKETS, len(drilldowns))} by severity):")
        for item in drilldowns[:_DRILLDOWN_BUCKETS]:
            stats = item["stats"]
            top = np.argsort(np.abs(stats["mean_shift"]))[::-1][:_WORST_FEATURES_PER_BUCKET]
            log(f"        {item['task']}=={item['class']}:")
            for feat_idx in top:
                name = feature_names[feat_idx]
                log(
                    f"          {name:<20s}  "
                    f"real μ={stats['mean_real'][feat_idx]:>+7.3f} "
                    f"σ={stats['std_real'][feat_idx]:>6.3f}    "
                    f"syn μ={stats['mean_synth'][feat_idx]:>+7.3f} "
                    f"σ={stats['std_synth'][feat_idx]:>6.3f}    "
                    f"Δμ={stats['mean_shift'][feat_idx]:>+5.2f}σ  "
                    f"σ_ratio={stats['std_ratio'][feat_idx]:>4.2f}"
                )

    # Joint-distribution diagnostic — separate section so the columns
    # don't fight for space with the marginal table.
    if joint_rows:
        log("")
        log("    === Joint-distribution diagnostic (correlation matrix) ===")
        log(
            f"      {'task':<10s}  {'cls':>3s}  {'used_real':>9s}  "
            f"{'used_syn':>9s}  {'|Δρ|max':>8s}  {'|Δρ|mean':>9s}  flag"
        )
        log("      " + "-" * 70)
        for row in joint_rows:
            log(
                f"      {row['task']:<10s}  {row['class']:>3d}  "
                f"{min(row['n_real'], max_samples_for_corr):>9d}  "
                f"{min(row['n_syn'], max_samples_for_corr):>9d}  "
                f"{row['max_abs_diff']:>8.3f}  {row['mean_abs_diff']:>9.3f}  "
                f"{row['joint_flag']}"
            )

    # Worst-pair drilldown — for each top-severity bucket, name the
    # feature pairs whose correlation disagrees the most.
    joint_drilldowns.sort(key=lambda r: r["severity"], reverse=True)
    if joint_drilldowns:
        log("")
        log(f"      worst feature PAIRS per bucket "
            f"(top {min(_DRILLDOWN_BUCKETS, len(joint_drilldowns))} by |Δρ|):")
        for item in joint_drilldowns[:_DRILLDOWN_BUCKETS]:
            stats = item["stats"]
            pairs = _top_pairs(stats["delta_corr"], _WORST_PAIRS_PER_BUCKET)
            log(f"        {item['task']}=={item['class']}:")
            for i, j, delta in pairs:
                name_i = feature_names[i]
                name_j = feature_names[j]
                log(
                    f"          {name_i:<18s} × {name_j:<18s}  "
                    f"real ρ={stats['corr_real'][i, j]:>+5.2f}  "
                    f"syn ρ={stats['corr_synth'][i, j]:>+5.2f}  "
                    f"Δρ={delta:>+5.2f}"
                )

    # Temporal-fidelity diagnostic — only when 3D inputs were
    # provided and at least one bucket had enough samples per side.
    if temporal_rows:
        log("")
        log("    === Temporal-distribution diagnostic (lag-1 autocorrelation) ===")
        log(
            f"      {'task':<10s}  {'cls':>3s}  {'|Δac|max':>9s}  "
            f"{'|Δac|mean':>10s}  flag"
        )
        log("      " + "-" * 60)
        for row in temporal_rows:
            log(
                f"      {row['task']:<10s}  {row['class']:>3d}  "
                f"{row['max_abs_diff']:>9.3f}  {row['mean_abs_diff']:>10.3f}  "
                f"{row['flag']}"
            )

    # Worst-feature autocorr drilldown.
    temporal_drilldowns.sort(key=lambda r: r["severity"], reverse=True)
    if temporal_drilldowns:
        log("")
        log(f"      worst features per bucket "
            f"(top {min(_DRILLDOWN_BUCKETS, len(temporal_drilldowns))} "
            f"by |Δ lag-1 autocorr|):")
        for item in temporal_drilldowns[:_DRILLDOWN_BUCKETS]:
            stats = item["stats"]
            top = _top_features_by_abs(
                stats["delta_autocorr"], _WORST_FEATURES_PER_BUCKET,
            )
            log(f"        {item['task']}=={item['class']}:")
            for feat_idx in top:
                name = feature_names[feat_idx]
                log(
                    f"          {name:<20s}  "
                    f"real ρ_lag1={stats['real_autocorr'][feat_idx]:>+5.2f}  "
                    f"syn ρ_lag1={stats['synth_autocorr'][feat_idx]:>+5.2f}  "
                    f"Δac={stats['delta_autocorr'][feat_idx]:>+5.2f}"
                )

    return results


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _flatten(
    data: Any,
    labels: Dict[str, np.ndarray],
) -> tuple:
    """Coerce data and labels to per-row matrices for marginal stats.

    Accepts:
      * np.ndarray (N, F)         → returned as-is, labels unchanged.
      * np.ndarray (N, T, F)      → reshaped to (N*T, F); labels are
                                    repeated along T so each timestep
                                    inherits its sample's class.
      * pd.DataFrame (N, F)       → ``.values`` taken, labels unchanged.
      * list of any of the above  → each is flattened and stacked; the
                                    label slices line up because the
                                    balancer guarantees per-batch
                                    alignment.

    Returns:
        ``(flat_data: ndarray (M, F), flat_labels: dict[task, ndarray])``
        with ``flat_labels[task]`` having ``M`` rows.
    """
    if isinstance(data, list):
        if not data:
            return np.empty((0, 0)), {t: np.empty((0, v.shape[1] if v.ndim == 2 else 0))
                                       for t, v in labels.items()}
        flat_parts: List[np.ndarray] = []
        label_parts: Dict[str, List[np.ndarray]] = {t: [] for t in labels}
        # Per-batch labels are sliced from the full labels dict by row.
        cursor = 0
        for batch in data:
            sub_labels = {
                t: labels[t][cursor:cursor + _leading_dim(batch)]
                for t in labels
            }
            sub_flat, sub_lbl_flat = _flatten(batch, sub_labels)
            if sub_flat.size:
                flat_parts.append(sub_flat)
                for t, v in sub_lbl_flat.items():
                    label_parts[t].append(v)
            cursor += _leading_dim(batch)
        if not flat_parts:
            return np.empty((0, 0)), {t: np.empty((0, 0)) for t in labels}
        flat = np.concatenate(flat_parts, axis=0)
        flat_labels = {t: np.concatenate(label_parts[t], axis=0) for t in labels}
        return flat, flat_labels

    if isinstance(data, pd.DataFrame):
        return data.to_numpy(), {t: np.asarray(v) for t, v in labels.items()}

    arr = np.asarray(data)
    if arr.size == 0:
        return arr.reshape(0, 0), {t: np.asarray(v) for t, v in labels.items()}

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim == 2:
        return arr, {t: np.asarray(v) for t, v in labels.items()}

    # 3D+ — fold non-trailing axes between leading (sample) and last
    # (feature) into one repeat factor.  Labels must repeat by the same
    # factor so each unrolled row inherits its sample's class.
    N = arr.shape[0]
    F = arr.shape[-1]
    repeat = int(np.prod(arr.shape[1:-1])) if arr.ndim > 2 else 1
    flat = arr.reshape(N * repeat, F)
    flat_labels = {
        t: np.repeat(np.asarray(v), repeat, axis=0) for t, v in labels.items()
    }
    return flat, flat_labels


def _leading_dim(batch: Any) -> int:
    if isinstance(batch, pd.DataFrame):
        return len(batch)
    return int(np.asarray(batch).shape[0])


def _per_feature_stats(real: np.ndarray, synth: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-feature comparison stats.

    Returns a dict of arrays each of shape (F,):
      * mean_real, mean_synth, std_real, std_synth
      * mean_shift = (mean_synth - mean_real) / max(std_real, eps)
      * std_ratio  = std_synth / max(std_real, eps)
    """
    eps = 1e-9
    mean_real = real.mean(axis=0)
    mean_synth = synth.mean(axis=0)
    std_real = real.std(axis=0)
    std_synth = synth.std(axis=0)
    safe_std_real = np.maximum(std_real, eps)
    return {
        "mean_real": mean_real,
        "mean_synth": mean_synth,
        "std_real": std_real,
        "std_synth": std_synth,
        "mean_shift": (mean_synth - mean_real) / safe_std_real,
        "std_ratio": std_synth / safe_std_real,
    }


def _flag_for_bucket(mean_shift_max: float, std_ratio_min: float) -> str:
    """Single-token marginal-fidelity flag for a (task, class) row."""
    flags = []
    if std_ratio_min < 0.3:
        flags.append("MODE_COLLAPSE")
    if mean_shift_max > 2.0:
        flags.append("OFF_DIST")
    elif mean_shift_max > 1.0:
        flags.append("shifted")
    if not flags:
        return "ok"
    return ",".join(flags)


def _joint_flag(max_abs_diff: float, mean_abs_diff: float) -> str:
    """Joint-fidelity flag.  Threshold rationale:

    * |Δρ|max > 0.5 — at least one feature pair has correlation off by
      half a unit (e.g. real ρ=+0.6, synth ρ=+0.1).  At that level the
      classifier definitely sees a different joint structure.
    * |Δρ|mean > 0.15 — average pairwise disagreement is substantial,
      meaning the GAN's covariance matrix is broadly off, not just a
      few outlier pairs.
    """
    flags = []
    if max_abs_diff > 0.5:
        flags.append("JOINT_BROKEN")
    elif max_abs_diff > 0.3:
        flags.append("joint_drift")
    if mean_abs_diff > 0.15:
        flags.append("corr_skew")
    if not flags:
        return "ok"
    return ",".join(flags)


def _combine_flags(marginal: str, joint: Optional[str]) -> str:
    """Merge marginal + joint flags into one display token."""
    if joint is None:
        return marginal
    if marginal == "ok" and joint == "ok":
        return "ok"
    parts = []
    if marginal != "ok":
        parts.append(marginal)
    if joint != "ok":
        parts.append(joint)
    return ",".join(parts)


def _correlation_stats(real: np.ndarray, synth: np.ndarray) -> Dict[str, Any]:
    """Compute per-bucket correlation matrices and pairwise disagreement.

    Returns:
        ``corr_real``  — (F, F) correlation matrix from real subset.
        ``corr_synth`` — (F, F) correlation matrix from synth subset.
        ``delta_corr`` — element-wise difference (real - synth).
        ``max_abs_diff`` — worst pairwise disagreement (off-diagonal,
            since the diagonal is always 1.0).
        ``mean_abs_diff`` — average over the upper triangle.

    The stats use Pearson correlation (numpy ``corrcoef``).  Constant
    features (zero variance) get ρ=0 to avoid NaNs.
    """
    F = real.shape[1]
    corr_real = _safe_corrcoef(real)
    corr_synth = _safe_corrcoef(synth)
    delta = corr_real - corr_synth

    # Off-diagonal mask (upper triangle, k=1) — diagonal always agrees,
    # and (i,j) == (j,i) so counting the lower triangle would double up.
    iu, ju = np.triu_indices(F, k=1)
    pair_diffs = np.abs(delta[iu, ju])
    if pair_diffs.size == 0:  # F == 1 edge case
        return {
            "corr_real": corr_real,
            "corr_synth": corr_synth,
            "delta_corr": delta,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }
    return {
        "corr_real": corr_real,
        "corr_synth": corr_synth,
        "delta_corr": delta,
        "max_abs_diff": float(pair_diffs.max()),
        "mean_abs_diff": float(pair_diffs.mean()),
    }


def _safe_corrcoef(arr: np.ndarray) -> np.ndarray:
    """``np.corrcoef`` that returns 0 instead of NaN for zero-variance
    columns.  Constant features make the standard formula divide by
    zero — catching that here is cheaper than special-casing every
    caller."""
    F = arr.shape[1]
    stds = arr.std(axis=0)
    nonconstant = stds > 0
    if not nonconstant.all():
        out = np.zeros((F, F), dtype=np.float64)
        if nonconstant.sum() >= 2:
            sub = arr[:, nonconstant]
            sub_corr = np.corrcoef(sub, rowvar=False)
            # Re-embed into full matrix.
            idx = np.where(nonconstant)[0]
            for a, ia in enumerate(idx):
                for b, ib in enumerate(idx):
                    out[ia, ib] = sub_corr[a, b]
        # Diagonal is 1 by convention even for constant features.
        np.fill_diagonal(out, 1.0)
        return out
    return np.corrcoef(arr, rowvar=False)


def _subsample(
    arr: np.ndarray, max_n: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``arr`` unchanged when small; otherwise a random row
    subset of size ``max_n`` (without replacement)."""
    if arr.shape[0] <= max_n:
        return arr
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx]


def _top_pairs(delta_corr: np.ndarray, k: int) -> List[tuple]:
    """Return top-k ``(i, j, delta)`` triples by ``|delta|`` from the
    upper triangle (i < j) of ``delta_corr``."""
    F = delta_corr.shape[0]
    iu, ju = np.triu_indices(F, k=1)
    diffs = delta_corr[iu, ju]
    if diffs.size == 0:
        return []
    abs_diffs = np.abs(diffs)
    order = np.argsort(abs_diffs)[::-1][:k]
    return [(int(iu[o]), int(ju[o]), float(diffs[o])) for o in order]


# ---------------------------------------------------------------------------
# Temporal-fidelity helpers
# ---------------------------------------------------------------------------


def _to_3d_or_none(data: Any) -> Optional[np.ndarray]:
    """Return a 3D ``(N, T, F)`` ndarray view of ``data`` if it is or
    can be coerced to one; ``None`` otherwise.

    Handles:
      * np.ndarray (N, T, F)         → as-is.
      * list of np.ndarray (n_i, T, F) → concatenated.
      * 2D ndarray, DataFrame, list of DataFrames → ``None`` (no
        temporal axis to analyze).
    """
    if isinstance(data, list):
        if not data:
            return None
        # Need every batch to be a 3D ndarray of compatible shape.
        try:
            arrays = [np.asarray(b) for b in data]
        except Exception:
            return None
        if any(a.ndim != 3 for a in arrays):
            return None
        if any(isinstance(b, pd.DataFrame) for b in data):
            return None
        try:
            return np.concatenate(arrays, axis=0)
        except Exception:
            return None
    if isinstance(data, pd.DataFrame):
        return None
    arr = np.asarray(data)
    if arr.ndim != 3:
        return None
    return arr


def _temporal_stats(real_3d: np.ndarray, synth_3d: np.ndarray) -> Dict[str, Any]:
    """Per-feature lag-1 autocorrelation comparison between real and
    synthetic 3D sequences.

    Returns dict with:
      * ``real_autocorr``  — (F,) Pearson ρ between consecutive
        timesteps per feature, computed over all pairs in all sequences.
      * ``synth_autocorr`` — same for synth.
      * ``delta_autocorr`` — element-wise real − synth.
      * ``max_abs_diff``   — worst per-feature autocorr disagreement.
      * ``mean_abs_diff``  — average disagreement across features.
    """
    F = real_3d.shape[-1]
    real_x = real_3d[:, :-1, :].reshape(-1, F)
    real_y = real_3d[:, 1:, :].reshape(-1, F)
    synth_x = synth_3d[:, :-1, :].reshape(-1, F)
    synth_y = synth_3d[:, 1:, :].reshape(-1, F)

    real_ac = _pairwise_autocorr(real_x, real_y)
    synth_ac = _pairwise_autocorr(synth_x, synth_y)
    delta = real_ac - synth_ac

    return {
        "real_autocorr": real_ac,
        "synth_autocorr": synth_ac,
        "delta_autocorr": delta,
        "max_abs_diff": float(np.max(np.abs(delta))),
        "mean_abs_diff": float(np.mean(np.abs(delta))),
    }


def _pairwise_autocorr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between two ``(M, F)`` arrays.

    Used to compute lag-1 autocorrelation: pass the consecutive-pairs
    flattening of a 3D sequence's first ``T-1`` timesteps as ``x`` and
    the last ``T-1`` as ``y``.
    """
    eps = 1e-9
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    sx = x.std(axis=0)
    sy = y.std(axis=0)
    cov = ((x - mu_x) * (y - mu_y)).mean(axis=0)
    return cov / (np.maximum(sx, eps) * np.maximum(sy, eps))


def _temporal_flag(max_abs_diff: float, mean_abs_diff: float) -> str:
    """Single-token temporal-fidelity flag.

    Threshold rationale (calibrated for trading-style indicators where
    real lag-1 autocorr is typically 0.85-0.99):

    * |Δac|max > 0.4  — at least one feature's temporal smoothness is
      severely off (e.g. real ρ=0.95, synth ρ=0.55 — synth is jumping
      around step-to-step where real evolves smoothly).
    * |Δac|max > 0.2  — moderate temporal drift.
    * |Δac|mean > 0.1 — average per-feature autocorrelation is broadly
      off, not just outliers.
    """
    flags = []
    if max_abs_diff > 0.4:
        flags.append("TEMPORAL_BROKEN")
    elif max_abs_diff > 0.2:
        flags.append("temporal_drift")
    if mean_abs_diff > 0.1:
        flags.append("autocorr_skew")
    if not flags:
        return "ok"
    return ",".join(flags)


def _top_features_by_abs(arr: np.ndarray, k: int) -> List[int]:
    """Return indices of the top-k entries by ``|arr|``."""
    if arr.size == 0:
        return []
    return [int(i) for i in np.argsort(np.abs(arr))[::-1][:k]]


def task_label_slices(
    running_labels_per_task: Dict[str, List[np.ndarray]],
    skip_first: int = 1,
) -> Dict[str, np.ndarray]:
    """Helper: concatenate ``running_labels[task][skip_first:]`` per task.

    The balancer keeps a running list where the 0th slot is always the
    real input labels.  Diagnostics only want the synthetic batches
    that were appended after, hence ``skip_first=1`` by default.
    """
    out: Dict[str, np.ndarray] = {}
    for task, batches in running_labels_per_task.items():
        tail = batches[skip_first:]
        if not tail:
            out[task] = np.empty((0, batches[0].shape[1] if batches else 0))
        else:
            out[task] = np.concatenate(tail, axis=0)
    return out
