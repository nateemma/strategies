"""
Class-balanced augmentation utilities for GANInterface.

This module owns the *orchestration* of class balancing — the per-class
needs computation, the generation loop, passthrough swapping, optional
diagnostics — for both single-task and multi-task GAN consumers.  It's
backend-agnostic: every concrete GAN type plugs in via ``GANInterface``
and contributes only its own ``generate()`` calling convention.

Two public entry points:

* ``balance_single_task(interface, data, labels, target_ratio, …)``
  — for WGAN, CTAB_GAN, and any future single-task backend.  Loops over
  classes, generates per-class deficits, dispatches the generate call
  based on ``interface.gan_type`` (``one_hot=`` for WGAN, ``class_label=``
  for CTAB_GAN).  Replaces the duplicated per-strategy loops in
  ``BaseNNStrategy.wgan_enhance_training_data`` and
  ``ctab_gan_enhance_training_data``.

* ``balance_multi_task(interface, data, labels, target_ratios, …)``
  — for MT_WGAN, MT_CTAB_GAN, and any future multi-task backend.  Runs a
  deficit-driven greedy loop that picks the largest (task, class)
  deficit each round and assigns labels for non-target tasks
  probabilistically from each task's own current deficit distribution.
  Solves the cross-task interference problem that a naive per-task loop
  can't (where balancing one task re-skews the others).

Strategies stay agnostic of which concrete GAN backend they're using —
they just declare ``gan_type`` and pass the interface in here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from GANs.diagnostics import summarize_real_vs_synthetic
from GANs.GANType import GANType
from GANs.passthrough import swap_passthrough_columns


# ---------------------------------------------------------------------------
# Defaults — match the values previously hardcoded in
# NNMT_WGAN._BALANCE_BATCH_SIZE / _BALANCE_MAX_ROUNDS.
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 50_000
DEFAULT_MAX_ROUNDS: int = 200


# Backends whose generate() returns a 3-D ndarray with seq_len=1 and
# expects a 2-D real-data input (i.e. WGAN-GP at single-task tabular).
# We squeeze the seq dimension before concatenating so the augmented
# output matches the shape the caller passed in.
_SQUEEZE_SEQ_DIM_TYPES: set = {GANType.WGAN}


# ---------------------------------------------------------------------------
# Public API — single-task
# ---------------------------------------------------------------------------


def balance_single_task(
    interface: Any,
    data: Any,
    labels: np.ndarray,
    target_ratio: Union[float, Dict[int, float]],
    *,
    log: Any = print,
    debug_log: Any = None,
    diagnostics: bool = False,
    feature_names: Optional[Sequence[str]] = None,
    passthrough_columns: Optional[Sequence[Union[int, str]]] = None,
) -> Tuple[Any, np.ndarray]:
    """
    Augment ``(data, labels)`` so each class reaches
    ``ratio * majority_class_count`` samples.

    Backend-agnostic — the GAN-specific generate() calling convention is
    derived from ``interface.gan_type``:

    * ``GANType.WGAN``       → ``interface.generate(n=..., one_hot=...)``,
                               returns ``(n, 1, F)`` ndarray (seq dim
                               squeezed before concatenation).
    * ``GANType.CTAB_GAN``   → ``interface.generate(n=..., class_label=int)``,
                               returns ``pd.DataFrame``.

    Args:
        interface:    A fitted/loaded ``GANInterface`` (single-task type).
        data:         Real training data — np.ndarray ``(N, F)`` or
                      pd.DataFrame ``(N, F)``.  Used as the running
                      augmented set's first batch.
        labels:       Real labels.  One of:
                        * 1-D ndarray of class indices ``(N,)``
                        * 2-D one-hot ndarray ``(N, num_classes)``
                      Output labels match the input shape.
        target_ratio: Either a ``float`` (broadcast across every class:
                      ``target = ratio * majority_count``) or a
                      ``Dict[class_idx, float]`` for per-class targets.
                      ``0`` / ``None`` disables augmentation.
        log:          ``print``-compatible callable for top-level info.
        debug_log:    Verbose per-class progress.  Defaults to ``log``.
        diagnostics:  When True, run a per-class fidelity comparison
                      between real and synthetic samples after balancing.
        feature_names: Column names used in the diagnostic drilldown.
        passthrough_columns: Columns whose values should be copied from
                      the real ``data`` pool rather than taken from the
                      GAN's output.  Names for DataFrame backends,
                      integer indices for ndarray backends.

    Returns:
        ``(aug_data, aug_labels)`` — same types/shapes as the inputs,
        concatenated along axis 0.  Pass-through (returns inputs
        unchanged) when no class needs augmentation.
    """
    if debug_log is None:
        debug_log = log

    # Input validation / empty short-circuit.
    if _is_empty(data) or len(labels) == 0:
        debug_log("    balance_single_task: empty input — skipping augmentation")
        return data, labels

    # --- Normalise label shape ------------------------------------------ #
    # We carry the real labels in their original shape so the output
    # matches; internally we use a class-indices view for counting.
    label_shape = "one_hot" if (labels.ndim == 2 and labels.shape[1] > 1) else "indices"
    if label_shape == "one_hot":
        num_classes = labels.shape[1]
        class_indices = np.argmax(labels, axis=1).astype(int)
    else:
        flat = np.asarray(labels).astype(int).flatten()
        num_classes = int(flat.max()) + 1 if flat.size else 0
        class_indices = flat

    if num_classes == 0:
        log("    balance_single_task: no classes found — skipping augmentation")
        return data, labels

    # --- Compute per-class needs ---------------------------------------- #
    classes_seen, counts = np.unique(class_indices, return_counts=True)
    classes_sorted = np.sort(classes_seen)
    original_counts = {int(c): int(n) for c, n in zip(classes_seen, counts)}
    debug_log(
        f"    balance_single_task: class counts {original_counts} "
        f"(gan_type={interface.gan_type.name})"
    )

    needs_map = _resolve_single_task_needs(
        target_ratio=target_ratio,
        original_counts=original_counts,
        classes_sorted=classes_sorted,
    )
    debug_log(
        f"    balance_single_task: per-class additions {needs_map}"
    )
    if not needs_map or all(v <= 0 for v in needs_map.values()):
        debug_log("    balance_single_task: nothing to add — at or above target")
        return data, labels

    # --- Generate per class --------------------------------------------- #
    rng = np.random.default_rng()
    aug_data_batches: List[Any] = []
    aug_label_batches: List[np.ndarray] = []

    for class_idx in classes_sorted:
        need = int(needs_map.get(int(class_idx), 0))
        if need <= 0:
            continue

        gen = _generate_for_class(
            interface=interface,
            n=need,
            class_idx=int(class_idx),
            num_classes=num_classes,
        )
        # Squeeze seq dimension for WGAN's (n, 1, F) → (n, F).
        if interface.gan_type in _SQUEEZE_SEQ_DIM_TYPES and getattr(gen, "ndim", 0) == 3:
            gen = gen[:, 0, :]

        if passthrough_columns:
            gen = swap_passthrough_columns(
                synth=gen,
                real_pool=data,
                columns=passthrough_columns,
                rng=rng,
                feature_names=feature_names,
            )

        aug_data_batches.append(gen)
        aug_label_batches.append(
            _build_single_task_labels(
                class_idx=int(class_idx),
                n=need,
                num_classes=num_classes,
                shape=label_shape,
                dtype=labels.dtype,
            )
        )
        debug_log(
            f"      generated {need} samples for class {int(class_idx)}"
        )

    if not aug_data_batches:
        debug_log("    balance_single_task: no batches generated — returning original")
        return data, labels

    # --- Concatenate ----------------------------------------------------- #
    aug_data = _concatenate_data([data, *aug_data_batches])
    aug_labels = np.concatenate([labels, *aug_label_batches], axis=0)

    # --- Summary --------------------------------------------------------- #
    new_indices = (
        np.argmax(aug_labels, axis=1) if label_shape == "one_hot"
        else np.asarray(aug_labels).astype(int).flatten()
    )
    new_classes, new_counts = np.unique(new_indices, return_counts=True)
    new_counts_map = {int(c): int(n) for c, n in zip(new_classes, new_counts)}
    log(
        f"    balance_single_task: rows {len(class_indices)} -> {len(new_indices)}; "
        f"counts {original_counts} -> {new_counts_map}"
    )

    # --- Optional fidelity diagnostic ------------------------------------ #
    if diagnostics:
        try:
            # Wrap into the multi-task-shaped dict the helper expects:
            # one synthetic "pseudo-task" containing the class one-hot.
            real_oh = (
                labels.astype(np.float32) if label_shape == "one_hot"
                else np.eye(num_classes, dtype=np.float32)[class_indices]
            )
            synth_oh = np.concatenate(aug_label_batches, axis=0)
            if label_shape != "one_hot":
                synth_oh = np.eye(num_classes, dtype=np.float32)[
                    np.asarray(synth_oh).astype(int).flatten()
                ]
            summarize_real_vs_synthetic(
                real_data=data,
                real_labels={"class": real_oh},
                synth_data=aug_data_batches,
                synth_labels={"class": synth_oh},
                feature_names=feature_names,
                log=debug_log,
            )
        except Exception as exc:  # diagnostics must never tank a training run
            log(f"    [diagnostics] skipped (error: {exc})")

    return aug_data, aug_labels


# ---------------------------------------------------------------------------
# Public API — multi-task
# ---------------------------------------------------------------------------


def balance_multi_task(
    interface: Any,
    data: Any,
    labels: Dict[str, np.ndarray],
    target_ratios: Union[
        float,
        Dict[str, Union[float, Dict[int, float]]],
    ],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    log: Any = print,
    debug_log: Any = None,
    diagnostics: bool = False,
    feature_names: Optional[Sequence[str]] = None,
    passthrough_columns: Optional[Sequence[Union[int, str]]] = None,
) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Iteratively augment ``(data, labels)`` so every (task, class) pair
    reaches its target count.  Backend-agnostic — works for MT_WGAN,
    MT_CTAB_GAN, and any future multi-task GAN exposed via GANInterface.

    Args:
        interface:      A fitted/loaded ``GANInterface`` whose ``generate()``
                        accepts ``n=`` and ``task_labels=`` and returns either
                        ``(samples_ndarray, labels_dict)`` or
                        ``(samples_dataframe, labels_dict)``.
        data:           Real training data — np.ndarray ``(N, *F)`` for
                        sequential backends or pd.DataFrame ``(N, F)`` for
                        tabular.  Used as the starting point of the running
                        augmented set.
        labels:         Real labels as ``Dict[task_name, one_hot_ndarray]``
                        with shape ``(N, num_classes_per_task)``.
        target_ratios:  Per-task target as a fraction of that task's majority
                        class count.  Either:
                          * a single ``float`` — broadcast to every task in
                            ``labels`` (convenience for ratio sweeps);
                          * a ``Dict[task, float]`` — per-task ratio applied
                            to every class within that task;
                          * a ``Dict[task, Dict[class_idx, float]]`` — full
                            per-(task, class) override.
                        ``None`` / ``0.0`` for a task disables augmentation
                        for that task.
        batch_size:     Maximum samples generated per round (smaller of this
                        and the largest remaining deficit).  Default 50,000.
        max_rounds:     Hard cap on iterations to prevent runaway.  Default 200.
                        If hit, ``log`` emits a warning.
        log:            ``print``-compatible callable for top-level info /
                        warnings.  Default: built-in ``print``.
        debug_log:      ``print``-compatible callable for verbose per-round
                        progress lines.  Defaults to ``log`` if not given.
        diagnostics:    When True, after balancing finishes, run a
                        per-(task, class) fidelity comparison between
                        real and synthetic samples (mean shift and
                        std ratio per feature, mode-collapse flagging).
                        Output is text via ``debug_log``.  Off by
                        default — turn on when investigating "balance
                        looks right but model regresses" failure modes.
        feature_names:  Optional column names used in the worst-feature
                        drilldown of the diagnostic.  Falls back to
                        ``f0..fF`` when omitted.
        passthrough_columns: Optional list of columns whose values
                        should be copied from the real ``data`` pool
                        rather than taken from the GAN's output.  Use
                        for deterministic / rigid-structure features
                        (calendar sin/cos pairs, one-hot categoricals)
                        that GANs reliably mis-reproduce.  Names for
                        DataFrame backends, integer indices for ndarray
                        backends.  ``None`` disables the swap.

    Returns:
        ``(aug_data, aug_labels)`` — same types as the inputs, augmented in
        place by concatenation along the leading axis.  Pass-through when
        every target_ratio is ``None``/``0`` (returns original
        ``data, labels`` unchanged).
    """
    if debug_log is None:
        debug_log = log

    task_names = list(labels.keys())

    # Scalar broadcast: ``target_ratios=0.5`` means "0.5 across every task".
    # Handy for ratio sweeps where the operator doesn't want to spell out
    # the same number per task.
    if isinstance(target_ratios, (int, float)):
        target_ratios = {task: float(target_ratios) for task in task_names}

    # --- Compute per-task target counts ----------------------------------- #
    targets_by_task: Dict[str, int] = _resolve_targets(labels, target_ratios)

    if not targets_by_task:
        log("    balance_multi_task: no augmentation targets — returning original data")
        return data, labels

    # --- Running state: list of batches, concatenated at the end ---------- #
    running_data: List[Any] = [data]
    running_labels: Dict[str, List[np.ndarray]] = {t: [v] for t, v in labels.items()}
    rng = np.random.default_rng()

    before_summary = {t: _running_counts(running_labels, t).copy() for t in task_names}

    # --- Iterative greedy fill ------------------------------------------- #
    rounds_run = 0
    for rounds_run in range(1, max_rounds + 1):
        deficits = _compute_deficits(running_labels, targets_by_task, task_names)
        target_task, target_class, target_deficit = _pick_largest_deficit(deficits)
        if target_task is None or target_deficit <= 0:
            break

        n = min(target_deficit, batch_size)
        batch_labels = _build_batch_labels(
            target_task, target_class, n, labels, deficits, rng
        )

        gen_result = interface.generate(n=n, task_labels=batch_labels)
        if isinstance(gen_result, tuple) and len(gen_result) == 2:
            gen_data, _ = gen_result
        else:  # defensive — older / non-conforming backends
            gen_data = gen_result

        # Replace passthrough columns with values copied from the real
        # ``data`` pool.  GANs reliably mis-reproduce deterministic
        # features (calendar sin/cos, one-hot categoricals); copying
        # from real samples sidesteps that failure mode entirely.
        if passthrough_columns:
            gen_data = swap_passthrough_columns(
                synth=gen_data,
                real_pool=data,
                columns=passthrough_columns,
                rng=rng,
                feature_names=feature_names,
            )

        running_data.append(gen_data)
        for t in task_names:
            running_labels[t].append(batch_labels[t])

        if rounds_run <= 5 or rounds_run % 10 == 0:
            debug_log(
                f"      round {rounds_run:3d}: "
                f"+{n} samples for {target_task}={target_class} "
                f"(deficit was {target_deficit})"
            )

    if rounds_run >= max_rounds:
        log(
            f"    WARNING: balance_multi_task hit round cap ({max_rounds}) "
            f"without satisfying every deficit; some tasks may still be "
            f"under-represented.  Bump max_rounds or investigate whether "
            f"the GAN is producing usable samples for the affected pairs."
        )

    # --- Concatenate everything ------------------------------------------ #
    aug_data = _concatenate_data(running_data)
    aug_labels = {
        t: np.concatenate(running_labels[t], axis=0) for t in task_names
    }

    # --- Per-task before/after summary ----------------------------------- #
    log(f"    Iterative balance complete after {rounds_run} round(s)")
    for task in task_names:
        if task not in targets_by_task:
            continue
        before = before_summary[task]
        after_idx = np.argmax(aug_labels[task], axis=1)
        after = np.bincount(after_idx, minlength=before.size)
        target = targets_by_task[task]
        before_pct = (before / max(int(before.sum()), 1)) * 100.0
        after_pct = (after / max(int(after.sum()), 1)) * 100.0
        debug_log(
            f"      {task:<10s} target={target:>9d}  "
            f"before={[f'{v:>7d}' for v in before]} "
            f"({[f'{p:5.1f}%' for p in before_pct]})  "
            f"after={[f'{v:>7d}' for v in after]} "
            f"({[f'{p:5.1f}%' for p in after_pct]})"
        )

    # --- Optional fidelity diagnostic ------------------------------------ #
    # Compare per-(task, class) feature distributions of the real input
    # against the concatenated synthetic batches we produced.  Only the
    # synthetic part is analysed (running_data[1:] / running_labels[t][1:])
    # — the real part stays as the baseline.
    if diagnostics and len(running_data) > 1:
        synth_data_batches = running_data[1:]
        synth_labels_per_task = {
            t: (np.concatenate(running_labels[t][1:], axis=0)
                if len(running_labels[t]) > 1
                else np.empty((0, running_labels[t][0].shape[1])))
            for t in task_names
        }
        try:
            summarize_real_vs_synthetic(
                real_data=data,
                real_labels=labels,
                synth_data=synth_data_batches,
                synth_labels=synth_labels_per_task,
                feature_names=feature_names,
                log=debug_log,
            )
        except Exception as exc:  # diagnostics must never tank a training run
            log(f"    [diagnostics] skipped (error: {exc})")

    return aug_data, aug_labels


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_targets(
    labels: Dict[str, np.ndarray],
    target_ratios: Dict[str, Union[float, Dict[int, float], None]],
) -> Dict[str, int]:
    """Resolve ``target_ratios`` into a single int target per task.

    The ratio may be a scalar (apply to all classes) or a dict (per-class
    overrides).  When a per-class dict is given we take the max — the
    deficit loop will still satisfy any smaller per-class targets along
    the way, so a single max-target value is sufficient for termination.
    """
    targets: Dict[str, int] = {}
    for task in labels:
        ratio_spec = target_ratios.get(task)
        if ratio_spec is None:
            continue
        if isinstance(ratio_spec, dict):
            ratio = max(ratio_spec.values()) if ratio_spec else 0.0
        else:
            ratio = float(ratio_spec)
        if ratio <= 0.0:
            continue
        idx = np.argmax(labels[task], axis=1)
        _, counts = np.unique(idx, return_counts=True)
        current_max = int(counts.max()) if counts.size else 0
        if current_max <= 0:
            continue
        targets[task] = int(current_max * ratio)
    return targets


def _running_counts(
    running_labels: Dict[str, List[np.ndarray]],
    task: str,
) -> np.ndarray:
    """Sum class counts across all batches accumulated so far for one task."""
    if not running_labels[task]:
        return np.zeros(0, dtype=np.int64)
    num_classes = running_labels[task][0].shape[1]
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in running_labels[task]:
        idx = np.argmax(batch, axis=1)
        counts += np.bincount(idx, minlength=num_classes).astype(np.int64)
    return counts


def _compute_deficits(
    running_labels: Dict[str, List[np.ndarray]],
    targets_by_task: Dict[str, int],
    task_names: List[str],
) -> Dict[str, np.ndarray]:
    """Per-task ``max(0, target − running_count)`` array (shape: num_classes)."""
    deficits: Dict[str, np.ndarray] = {}
    for task in task_names:
        if task not in targets_by_task:
            num_classes = running_labels[task][0].shape[1] if running_labels[task] else 0
            deficits[task] = np.zeros(num_classes, dtype=np.int64)
            continue
        counts = _running_counts(running_labels, task)
        target = targets_by_task[task]
        deficits[task] = np.maximum(target - counts, 0)
    return deficits


def _pick_largest_deficit(
    deficits: Dict[str, np.ndarray],
) -> Tuple[Any, int, int]:
    """Find the (task, class, deficit) with the largest single value."""
    best_task = None
    best_class = -1
    best_deficit = 0
    for task, def_arr in deficits.items():
        if def_arr.size == 0:
            continue
        idx = int(np.argmax(def_arr))
        value = int(def_arr[idx])
        if value > best_deficit:
            best_task = task
            best_class = idx
            best_deficit = value
    return best_task, best_class, best_deficit


def _sample_other_task_labels(
    other_task: str,
    n: int,
    labels: Dict[str, np.ndarray],
    deficits: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """One-hot labels for ``other_task`` weighted by its current deficits.

    Tasks with no remaining deficit (all zeros) sample uniformly so the
    augmentation doesn't disturb their existing distribution further.
    """
    num_classes = labels[other_task].shape[1]
    def_arr = deficits.get(other_task, np.zeros(num_classes, dtype=np.int64))
    total = int(def_arr.sum())
    if total > 0:
        probs = def_arr.astype(np.float64) / total
    else:
        probs = np.full(num_classes, 1.0 / num_classes)
    picks = rng.choice(num_classes, size=n, p=probs)
    return np.eye(num_classes, dtype=np.float32)[picks]


def _build_batch_labels(
    target_task: str,
    target_class: int,
    n: int,
    labels: Dict[str, np.ndarray],
    deficits: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Build the ``task_labels`` dict for one ``interface.generate(...)`` call."""
    batch: Dict[str, np.ndarray] = {}

    num_target_classes = labels[target_task].shape[1]
    target_oh = np.zeros((n, num_target_classes), dtype=np.float32)
    target_oh[:, target_class] = 1.0
    batch[target_task] = target_oh

    for other_task in labels:
        if other_task == target_task:
            continue
        batch[other_task] = _sample_other_task_labels(
            other_task, n, labels, deficits, rng
        )
    return batch


def _concatenate_data(batches: List[Any]) -> Any:
    """Concatenate batches along axis 0, choosing np vs pd polymorphically.

    DataFrame batches are aligned on the column order of the first batch
    so callers can mix DataFrames produced by GAN backends (which may
    return columns in their own training order) with the strategy's
    canonical layout — ``pd.concat`` would otherwise drift columns.

    Mixed-type batches (DataFrame + ndarray) are tolerated: when the
    first batch is a DataFrame we promote ndarray batches to DataFrames
    using the canonical columns; when the first batch is an ndarray we
    demote any DataFrame batches via ``.values``.  This handles GAN
    backends that return ndarrays even though the strategy's real-data
    pool is a DataFrame (e.g. CTAB_GAN reduced-output variants).
    """
    if not batches:
        raise ValueError("_concatenate_data: no batches to concatenate")
    first = batches[0]
    if isinstance(first, pd.DataFrame):
        canonical_cols = list(first.columns)
        aligned: List[pd.DataFrame] = []
        for b in batches:
            if isinstance(b, pd.DataFrame):
                aligned.append(b[canonical_cols])
                continue
            # Promote ndarray to DataFrame so pd.concat accepts it.
            arr = np.asarray(b)
            if arr.ndim != 2:
                raise ValueError(
                    f"_concatenate_data: cannot mix {arr.ndim}-D ndarray with "
                    f"DataFrame (expected 2-D, shape was {arr.shape})"
                )
            if arr.shape[1] != len(canonical_cols):
                raise ValueError(
                    f"_concatenate_data: ndarray batch has {arr.shape[1]} cols "
                    f"but reference DataFrame has {len(canonical_cols)}"
                )
            aligned.append(pd.DataFrame(arr, columns=canonical_cols))
        return pd.concat(aligned, axis=0, ignore_index=True)
    arrays = [
        b.values if isinstance(b, pd.DataFrame) else np.asarray(b)
        for b in batches
    ]
    return np.concatenate(arrays, axis=0)


def _is_empty(data: Any) -> bool:
    """True if ``data`` is None or has zero rows."""
    if data is None:
        return True
    if isinstance(data, pd.DataFrame):
        return data.empty
    arr = np.asarray(data)
    return arr.size == 0


# ---------------------------------------------------------------------------
# Single-task internals
# ---------------------------------------------------------------------------


def _resolve_single_task_needs(
    *,
    target_ratio: Union[float, Dict[int, float]],
    original_counts: Dict[int, int],
    classes_sorted: np.ndarray,
) -> Dict[int, int]:
    """Compute ``max(0, target − have)`` per class.

    ``target_ratio`` may be a scalar (broadcast across classes:
    ``target = ratio * majority_count``) or a per-class dict.  The
    majority count is the largest entry in ``original_counts`` for
    scalar targets; per-class targets multiply against the same
    majority count so the dict semantics match ``balance_multi_task``.
    """
    if not original_counts:
        return {}

    majority = max(original_counts.values())
    if majority <= 0:
        return {}

    if isinstance(target_ratio, dict):
        per_class_ratios = {int(k): float(v) for k, v in target_ratio.items()}
    else:
        ratio = float(target_ratio) if target_ratio is not None else 0.0
        if ratio <= 0:
            return {}
        per_class_ratios = {
            int(c): ratio for c in classes_sorted
        }

    needs: Dict[int, int] = {}
    for c in classes_sorted:
        c_int = int(c)
        ratio = per_class_ratios.get(c_int, 0.0)
        if ratio <= 0:
            continue
        target = int(majority * ratio)
        have = original_counts.get(c_int, 0)
        if target > have:
            needs[c_int] = target - have
    return needs


def _generate_for_class(
    *,
    interface: Any,
    n: int,
    class_idx: int,
    num_classes: int,
) -> Any:
    """Dispatch to ``interface.generate(n=…, …)`` using the right kwarg.

    The single-task GAN backends each have their own conditioning
    convention.  Centralising the dispatch here keeps the strategy code
    completely agnostic of GAN-type-specific calling shapes.
    """
    gan_type = interface.gan_type

    if gan_type == GANType.WGAN:
        one_hot = np.zeros((n, num_classes), dtype=np.float32)
        one_hot[:, class_idx] = 1.0
        return interface.generate(n=n, one_hot=one_hot)

    if gan_type == GANType.CTAB_GAN:
        return interface.generate(n=n, class_label=int(class_idx))

    if gan_type == GANType.CGAN:
        one_hot = np.zeros((n, num_classes), dtype=np.float32)
        one_hot[:, class_idx] = 1.0
        return interface.generate(n=n, one_hot=one_hot)

    raise ValueError(
        f"balance_single_task: GANType.{gan_type.name} is not a single-task "
        f"backend (use balance_multi_task instead)."
    )


def _build_single_task_labels(
    *,
    class_idx: int,
    n: int,
    num_classes: int,
    shape: str,
    dtype: Any,
) -> np.ndarray:
    """Build labels for ``n`` synthetic samples of one class, in caller's shape."""
    if shape == "one_hot":
        out = np.zeros((n, num_classes), dtype=dtype)
        out[:, class_idx] = 1
        return out
    return np.full(n, class_idx, dtype=dtype)
