"""The ONE fixed learnability probe (T029, FR-015, FR-015a).

The leaderboard's second column. It answers a different question from the first: not "were
these moments collectable" but "can a model TELL which of them were". A trigger whose
economic value is near zero while its label is highly learnable is the spec's explanation
(b) — the trigger hands the model an easy target and the apparent edge is the entry rule.

ONE probe, not the best probe (FR-015). The same model, the same features, the same splits
and the same metric for every trigger. Varying any of them would make the column
incomparable, and an incomparable column cannot support a verdict. It is therefore
deliberately NOT tuned per trigger, and a trigger that would score better under a different
architecture still gets this one.

DIAGNOSTIC ONLY, never an acceptance input (FR-015a, FR-018). This whole feature exists
because learnability tracked return across four arms while one of them lost money
(val_mcc 0.204/0.521/0.580/0.622 against -0.8/+21.8/+35.4/+40.0pp). A high MCC here is a
fact about predictability, not a licence to promote anything. Constitution III.

WHAT IS REUSED, AND WHY NOT `run.run_one`
-----------------------------------------
Everything that defines the protocol comes from `research/clfstudy`, by import:

  * the splits — `splits.build("v1")`, the study's own three windows (A/B/C);
  * the scaler discipline — `features.fit_scaler` on TRAIN-role rows of included pairs
    only, then `features.apply_scaler`, exactly as `run._build_tensors` does it;
  * the tensor construction — `windows.make_tensors`;
  * the model — `ClassifierTypeMLX.RIDGE`, i.e. `NNNClassifierMLX_Ridge`;
  * the metric — `sklearn.metrics.matthews_corrcoef`, the same call `metrics.score` makes.

`run.run_one` is NOT called, for one hard reason: it appends to
`research/clfstudy/results/results.jsonl`, an append-only record of ~1,575 runs with a test
pinning historical `config_id`s. Probe runs are not study runs and must not land there.
This module writes to `research/outcomes/` and nowhere else.

THE EFFICIENCY THAT MAKES THIS AFFORDABLE. Every trigger shares the same FEATURES; only the
label differs. So tensors are built ONCE per (window, input_set) and the label swapped per
trigger, instead of rebuilding per trigger. Measured from the study's own 279 existing RIDGE
rows, a fit is a median 29.4s, so 6 triggers x 3 windows x 3 seeds is about 27 minutes of
fitting rather than hours of re-loading.

THE PROBE'S KNOWN BIAS, stated here and required in the report (contracts/trigger-cli.md)
-----------------------------------------------------------------------------------------
RIDGE is a linear model over the flattened window. **It cannot express a logical AND.** A
trigger-defined label is literally `trigger AND positive outcome`, so the probe is
structurally handicapped on exactly the labels this feature studies, and every absolute MCC
here UNDERSTATES what a non-linear model could reach.

The verdict rule is unaffected, because both sides of its comparison use the same probe and
the handicap applies equally. The absolute numbers are NOT learnability ceilings and must
not be read as "this label is unlearnable" — only as "this label is less linearly separable
than that one". Feature 018 already measured the gap: across ~2,000 runs RIDGE's median MCC
is 0.138 while the top architectures tie around 0.56.

A SECOND LIMIT, found before running (T024): `geometry`'s label rate is 1.76%, below the
~5.5% line at which all-Hold collapse has been observed in this repository. Its row may come
back collapse-limited rather than information-limited, which is a property of the label's
sparsity and not of the trigger's quality. `collapsed` is therefore reported per fit rather
than inferred from a low MCC afterwards.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.clfstudy import data as _cls_data  # noqa: E402
from research.clfstudy import features as _cls_features  # noqa: E402
from research.clfstudy import splits as _cls_splits  # noqa: E402
from research.clfstudy import windows as _cls_windows  # noqa: E402
from research.outcomes import labels as _labels  # noqa: E402
from research.outcomes import triggers as _triggers  # noqa: E402

#: The study's split set. Fixed, not a parameter: FR-015's "one fixed probe" is meaningless
#: if the splits can move between triggers.
SPLIT_SET = "v1"

#: The study's own defaults for the probe's shape: `ind24` features, `seq_len=16`, the same
#: as the 279 existing RIDGE rows in `results.jsonl`.
#:
#: NOT comparable to those rows' MCC, though, and the first smoke fit showed why: a trigger
#: label is BINARY (HOLD/BUY, no SELL) while the study's is TERNARY, and a two-class problem
#: is simply easier. Measured, gbb/window A/seed 0 scores 0.5906 here against the study's
#: RIDGE median of 0.138 -- a difference in the number of classes, not in learnability. This
#: column is comparable ACROSS TRIGGERS, which is all FR-015 requires and all the verdict
#: rule uses; it is not a continuation of the study's RIDGE series.
INPUT_SET = "ind24"
SEQ_LEN = 16
STRIDE = 1

#: Seeds and windows give the spread R9 asks for as the verdict rule's standard error. Three
#: of each matches what the study's own RIDGE rows carry (seeds 0/1/2, windows A/B/C), so the
#: spread is measured on the same grid rather than a new one.
SEEDS = (0, 1, 2)

#: Ternary class convention, from `metrics.score`: `proba[:, 2]` is BUY. A trigger label is
#: BINARY, so it maps onto HOLD/BUY and never emits SELL. Recorded as constants because
#: silently using 0/1 would put the positive class where the study's SELL class lives and
#: make every MCC incomparable with the study's own rows.
CLASS_HOLD = 1
CLASS_BUY = 2

#: Above this share of a single predicted class, a fit is flagged collapsed. Matches
#: `metrics.score`'s `collapse_frac` default so the flag means the same thing here.
COLLAPSE_FRAC = 0.995

#: This package's OWN checkpoint directory. Never clfstudy's, so a probe fit and a study
#: run cannot collide in either direction.
CKPT_DIR = Path(__file__).resolve().parent / ".cache" / "probe_ckpt"


@dataclass(frozen=True)
class ProbeFit:
    """One (window, seed) fit."""

    window_id: str
    seed: int
    mcc: float
    n_train: int
    n_test: int
    label_rate_train: float
    label_rate_test: float
    collapsed: bool
    train_seconds: float = 0.0


@dataclass(frozen=True)
class ProbeResult:
    """A trigger's learnability column entry (data-model.md's `learnability_*`)."""

    trigger: str
    fits: tuple[ProbeFit, ...] = field(default_factory=tuple)

    @property
    def mcc(self) -> float:
        """Mean MCC across windows and seeds."""
        vals = [f.mcc for f in self.fits if np.isfinite(f.mcc)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def se(self) -> float:
        """Standard error of that mean across (window, seed) — R9's uncertainty.

        The spread is across BOTH windows and seeds deliberately. A seed-only spread would
        understate it badly: the study's own RIDGE rows move far more between windows A/B/C
        (different market eras) than between seeds, and the verdict rule has to survive that
        variation, not just optimiser noise.
        """
        vals = [f.mcc for f in self.fits if np.isfinite(f.mcc)]
        if len(vals) < 2:
            return float("nan")
        return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    @property
    def n_collapsed(self) -> int:
        return sum(1 for f in self.fits if f.collapsed)

    @property
    def collapse_limited(self) -> bool:
        """True when a majority of fits collapsed, so a low MCC says more about the label's
        sparsity than about the trigger. Reported rather than silently folded into the mean."""
        return bool(self.fits) and self.n_collapsed > len(self.fits) / 2


# =================================================================================================
# Window features — built ONCE per window and shared across triggers
# =================================================================================================

@dataclass
class WindowFeatures:
    """Scaled, role-assigned frames for one window, with the label column left empty.

    Reused across every trigger: the features are identical and only the label changes, so
    rebuilding per trigger would multiply the dominant cost for nothing.
    """

    window_id: str
    scaled: dict[str, pd.DataFrame]
    roles: dict[str, np.ndarray]
    cols: list[str]
    included: list[str]


def build_window_features(
    window_id: str,
    raw_frames: dict[str, pd.DataFrame],
    *,
    input_set: str = INPUT_SET,
    split_set: str = SPLIT_SET,
) -> WindowFeatures:
    """Mirror `run._build_tensors`' feature path, using clfstudy's own helpers.

    The scaler is fit on TRAIN-role rows of INCLUDED pairs only and then applied — the
    leakage boundary `run.py`'s module docstring describes. Calling
    `features.fit_scaler`/`apply_scaler` rather than re-deriving them is the point: a second
    scaler implementation is how the probe's protocol would drift from the study's.
    """
    ss = _cls_splits.build(split_set)
    w = next((x for x in ss.windows if x.window_id == window_id), None)
    if w is None:
        raise ValueError(f"window {window_id!r} not in split set {split_set!r}")

    first_dates = {p: pd.Timestamp(f["date"].min()) for p, f in raw_frames.items()}
    included = _cls_splits.included_pairs(w, first_dates)
    if not included:
        raise ValueError(f"window {window_id!r} includes no pairs")

    roles = {p: _cls_splits.assign_roles(raw_frames[p]["date"], w) for p in included}
    cols = _cls_features.columns_for(input_set)

    train_df = pd.concat(
        [raw_frames[p][np.asarray(roles[p]) == "train"] for p in included],
        ignore_index=True,
    )
    if train_df.empty:
        raise ValueError(f"window {window_id!r} has no training rows")
    scaler = _cls_features.fit_scaler(train_df, cols)

    # `apply_scaler` returns ONLY the scaled feature columns, so `date` has to be
    # re-attached -- `windows.make_tensors` needs it to assign rows to splits. Exactly what
    # `run._build_tensors` does at the same point; `label` is attached later, per trigger.
    scaled = {}
    for p in included:
        g = _cls_features.apply_scaler(raw_frames[p], scaler, cols)
        g["date"] = raw_frames[p]["date"].to_numpy()
        scaled[p] = g
    return WindowFeatures(window_id, scaled, roles, list(cols), list(included))


# =================================================================================================
# The probe
# =================================================================================================

def _ternary(mask: np.ndarray) -> np.ndarray:
    """Binary trigger label -> the study's ternary convention. Never emits SELL."""
    return np.where(np.asarray(mask, bool), CLASS_BUY, CLASS_HOLD).astype(int)


def probe_label(
    wf: WindowFeatures,
    label_by_pair: dict[str, np.ndarray],
    *,
    trigger_name: str,
    seed: int,
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE,
) -> ProbeFit:
    """Fit RIDGE on one window with the given per-pair binary label; score test MCC.

    Construction goes through `registry.build`, not a direct constructor call, so the probe
    inherits the study's exact configuration discipline: hyperparameters, the training seed,
    the optimizer knobs from `ExperimentConfig.train`, and the checkpoint path. Building the
    classifier by hand would reproduce none of that and would silently diverge.

    TWO HAZARDS THIS HANDLES, both of which this repository has been bitten by:

      * CHECKPOINT COLLISION. `registry.build` puts `window_id` in the checkpoint filename
        precisely because a bare `<config_id>_<seed>` collides across windows, making window
        B resume from A's weights. The same collision applies ACROSS TRIGGERS here -- every
        trigger shares one `config_id`, since only the label differs and the label is not
        part of the config -- so the trigger name is folded into the `window_id` passed in.
        Without that, six triggers would train one model and five would report its MCC.
      * STALE WEIGHTS. `_clear_stale_model_files` then `force_train=True`, exactly as
        `run.run_one` does, so a previous fit's artefact can never be loaded instead of
        training. "A renamed strategy self-trains" is a recorded hazard here.

    The checkpoint directory is this package's own, never clfstudy's, so probe fits cannot
    collide with study checkpoints in either direction.

    MLX is imported lazily so this module stays importable (and unit-testable) without it.
    """
    import time

    from research.clfstudy import registry as _cls_registry
    from research.clfstudy.registry import ExperimentConfig
    from research.clfstudy.run import _clear_stale_model_files

    frames = {}
    for p, f in wf.scaled.items():
        g = f.copy()
        g["label"] = _ternary(label_by_pair[p])
        frames[p] = g

    tensors = _cls_windows.make_tensors(
        frames, wf.roles, wf.cols, seq_len, stride=stride, label_col="label",
    )
    X_train, y_train_1h, y_train_int = tensors["train"]
    X_val, y_val_1h, _ = tensors["val"]
    X_test, y_test_int = tensors["test"][0], tensors["test"][1]
    del tensors, frames

    cw = _cls_features.class_weights(y_train_int, n_classes=_cls_registry.NUM_CLASSES)
    cfg = ExperimentConfig(arch="RIDGE", input_set=INPUT_SET, seq_len=seq_len,
                           split_set=SPLIT_SET)
    clf = _cls_registry.build(
        cfg, nf=len(wf.cols), seq_len=seq_len, seed=seed,
        checkpoint_dir=CKPT_DIR,
        window_id=f"{wf.window_id}_{trigger_name}",
    )
    _clear_stale_model_files(clf)

    t0 = time.perf_counter()
    clf.train(X_train, X_val, y_train_1h, y_val_1h, force_train=True, class_weights=cw)
    secs = time.perf_counter() - t0
    if getattr(clf, "epochs_run", 1) < 1:
        raise RuntimeError(f"{trigger_name}/{wf.window_id}/seed{seed}: no epochs run")

    proba = np.asarray(clf.predict(X_test))
    y_pred = proba.argmax(axis=1) if proba.ndim == 2 else np.asarray(proba, int)
    frac = float(np.bincount(y_pred, minlength=3).max() / max(len(y_pred), 1))
    y_true = np.asarray(y_test_int)
    mcc = (
        float(matthews_corrcoef(y_true, y_pred))
        if len(set(y_true.tolist())) > 1 else float("nan")
    )

    return ProbeFit(
        window_id=wf.window_id, seed=seed, mcc=mcc,
        n_train=int(len(y_train_int)), n_test=int(len(y_true)),
        label_rate_train=float((np.asarray(y_train_int) == CLASS_BUY).mean()),
        label_rate_test=float((y_true == CLASS_BUY).mean()),
        collapsed=frac >= COLLAPSE_FRAC, train_seconds=secs,
    )


def probe_trigger(
    trigger: "_triggers.Trigger | str",
    window_features: Sequence[WindowFeatures],
    outcomes_by_pair: dict[str, pd.DataFrame],
    raw_frames: dict[str, pd.DataFrame],
    *,
    seeds: Iterable[int] = SEEDS,
) -> ProbeResult:
    """`probe_label` across every window and seed, giving the mean MCC and R9's spread.

    The label is formed by `labels.trigger_label_mask`, so the probe sees exactly the label
    the leaderboard's other column is about -- `trigger AND __outcome_positive`, with the
    alignment check that function enforces -- rather than a separately-derived one.
    """
    name = trigger if isinstance(trigger, str) else trigger.name
    fits: list[ProbeFit] = []
    for wf in window_features:
        label_by_pair = {
            p: _labels.trigger_label_mask(
                raw_frames[p], outcomes_by_pair[p], trigger
            ).to_numpy()
            for p in wf.scaled
        }
        for seed in seeds:
            fits.append(probe_label(wf, label_by_pair, trigger_name=name, seed=seed))
    return ProbeResult(trigger=name, fits=tuple(fits))


def format_row(r: ProbeResult) -> str:
    flag = "COLLAPSE-LIMITED" if r.collapse_limited else ""
    return (
        f"{r.trigger:<14}{r.mcc:>+10.4f}{r.se:>+10.4f}{len(r.fits):>7}"
        f"{r.n_collapsed:>11}  {flag}"
    )


def header() -> str:
    return f"{'trigger':<14}{'mean MCC':>10}{'SE':>10}{'n fits':>7}{'collapsed':>11}"
