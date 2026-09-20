# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Architecture registry and `ExperimentConfig` for the NNC classifier study (T014).

Everything a phase file needs to go from YAML to a trained `MLXClassifierNary` instance:

* `ExperimentConfig` — the typed, validated form of one YAML list entry (data-model.md
  §ExperimentConfig); `config_id()` is the identity used to key `results.jsonl` and skip
  already-completed runs (research R9).
* `resolve()` / `is_baseline()` — arch-name -> `ClassifierTypeMLX` member, with the two
  training-free pseudo-archs (`MAJORITY`, `PERSISTENCE`) handled separately by `run.py`.
* `build()` — constructs and fully configures a classifier instance (hyperparameters, seed,
  optimizer settings, checkpoint routing) but does not train it.
* `param_count()` — trainable-parameter count, valid only after `build(...).train(...)` has
  run at least once (model creation is lazy).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from research.clfstudy import NUM_CLASSES
from research.clfstudy.labels import GENERIC_LABEL_NAMES
from NNNC.NNNClassifierMLX import ClassifierTypeMLX, create_classifier_mlx

# -----------------------------------------------------------------------
# Baseline pseudo-archs (research R6) — not registry members, bypass training.
# -----------------------------------------------------------------------

BASELINE_ARCHS = ("MAJORITY", "PERSISTENCE")

#: input_set values that need no further parsing (research R11)
_PLAIN_INPUT_SETS = ("ind24", "raw8", "both32")

#: ablation groups for `ind24-minus:<group>` (research R11); hardcoded here (rather than
#: imported from features.py) so this module has no dependency on the feature pipeline.
INPUT_SET_GROUPS = (
    "trend",
    "volatility",
    "momentum",
    "oscillators",
    "guards",
    "derived",
)

#: `train` dict defaults (data-model.md §ExperimentConfig); user-supplied keys override these.
TRAIN_DEFAULTS: Dict[str, Any] = {
    "max_epochs": 40,
    "patience": 10,
    "batch_size": 256,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "stride": 1,
}

#: `label` dict default -- means "exactly the production gbb label, unchanged" (research T018+).
#: Kept OUT of `config_id` when a config's `label` equals this exactly (byte-for-byte, including
#: no extra keys), so every config written before `label` existed keeps its id.
DEFAULT_LABEL: Dict[str, Any] = {"name": "gbb"}

#: `label` dict defaults for `name == "triple_barrier"`; user-supplied keys override these
#: (mirrors `TRAIN_DEFAULTS`'s merge-over-defaults pattern).
_LABEL_TRIPLE_BARRIER_DEFAULTS: Dict[str, Any] = {
    "conflict_resolution": True,
    "peak_filter": False,
}

_VALID_LABEL_NAMES = ("gbb", "triple_barrier") + GENERIC_LABEL_NAMES


def resolve(arch: str) -> ClassifierTypeMLX:
    """`arch` name -> `ClassifierTypeMLX` member. Raises `KeyError` with the valid names."""
    try:
        return ClassifierTypeMLX[arch]
    except KeyError:
        valid = ", ".join(m.name for m in ClassifierTypeMLX)
        raise KeyError(
            f"unknown arch {arch!r}; valid archs: {valid} (or baselines: "
            f"{', '.join(BASELINE_ARCHS)})"
        ) from None


def is_baseline(arch: str) -> bool:
    """True for the training-free pseudo-archs; these are never registry members."""
    return arch in BASELINE_ARCHS


def _parse_input_set(input_set: str) -> None:
    """Raise `ValueError` unless `input_set` is a recognised value (research R11)."""
    if input_set in _PLAIN_INPUT_SETS:
        return
    if input_set.startswith("ind24-minus:"):
        group = input_set.split(":", 1)[1]
        if group in INPUT_SET_GROUPS:
            return
        raise ValueError(
            f"unknown ablation group {group!r} in input_set {input_set!r}; "
            f"valid groups: {', '.join(INPUT_SET_GROUPS)}"
        )
    raise ValueError(
        f"invalid input_set {input_set!r}; expected one of {_PLAIN_INPUT_SETS} or "
        f"'ind24-minus:<group>'"
    )


@dataclass
class ExperimentConfig:
    """One YAML `configs:` list entry (data-model.md §ExperimentConfig).

    `phase`, `seeds` and `tags` are echoed to `results.jsonl` but excluded from `config_id`
    (research R9) — they describe *when/how often* a config is run, not *what* it is.
    """

    arch: str
    hp: Dict[str, Any] = field(default_factory=dict)
    input_set: str = "ind24"
    seq_len: int = 16
    split_set: str = "v1"
    train: Dict[str, Any] = field(default_factory=dict)
    phase: int = 0
    seeds: List[int] = field(default_factory=lambda: [0])
    tags: List[str] = field(default_factory=list)
    label: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LABEL))

    def __post_init__(self) -> None:
        # Merge user-supplied `train` keys over the defaults so every config carries the
        # full dict (both for `config_id` and for `build()`); user values win.
        merged = dict(TRAIN_DEFAULTS)
        merged.update(self.train or {})
        self.train = merged

        # Same merge-over-defaults pattern for `label`, but only for `triple_barrier` --
        # `gbb` must stay exactly `DEFAULT_LABEL` (no extra keys) so `config_id` can omit
        # it by equality (research: config_id byte-identical for every pre-existing config).
        self.label = dict(self.label) if self.label else dict(DEFAULT_LABEL)
        if self.label.get("name") == "triple_barrier":
            merged_label = dict(_LABEL_TRIPLE_BARRIER_DEFAULTS)
            merged_label.update(self.label)
            self.label = merged_label

    def validate(self) -> None:
        """Raise `ValueError` on the first violated rule (data-model.md §ExperimentConfig)."""
        if not (is_baseline(self.arch) or self.arch in ClassifierTypeMLX.__members__):
            valid = ", ".join(list(BASELINE_ARCHS) + [m.name for m in ClassifierTypeMLX])
            raise ValueError(f"unknown arch {self.arch!r}; valid: {valid}")
        _parse_input_set(self.input_set)
        if self.seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {self.seq_len}")
        if self.train["max_epochs"] < 1:
            raise ValueError(f"train.max_epochs must be >= 1, got {self.train['max_epochs']}")
        if self.train["stride"] < 1:
            raise ValueError(f"train.stride must be >= 1, got {self.train['stride']}")
        self._validate_label()

    def _validate_label(self) -> None:
        name = self.label.get("name")
        if name not in _VALID_LABEL_NAMES:
            raise ValueError(
                f"unknown label {name!r}; valid label names: {', '.join(_VALID_LABEL_NAMES)}"
            )
        if name == "gbb":
            extra = sorted(set(self.label) - {"name"})
            if extra:
                raise ValueError(f"label 'gbb' accepts no extra keys, got {extra}")
            return

        if name in GENERIC_LABEL_NAMES:
            self._validate_generic_label(name)
            return

        # triple_barrier
        for key in ("horizon", "up", "down"):
            if key not in self.label:
                raise ValueError(f"label 'triple_barrier' missing required key {key!r}")

        horizon = self.label["horizon"]
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError(
                f"label 'triple_barrier' key 'horizon' must be an int >= 1, got {horizon!r}"
            )
        for key in ("up", "down"):
            value = self.label[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"label 'triple_barrier' key {key!r} must be a float > 0, got {value!r}"
                )
        conflict_resolution = self.label.get("conflict_resolution")
        if not isinstance(conflict_resolution, bool):
            raise ValueError(
                "label 'triple_barrier' key 'conflict_resolution' must be bool, got "
                f"{conflict_resolution!r}"
            )
        peak_filter = self.label.get("peak_filter")
        if not isinstance(peak_filter, bool):
            raise ValueError(
                f"label 'triple_barrier' key 'peak_filter' must be bool, got {peak_filter!r}"
            )
        if "atr_scale" in self.label:
            atr_scale = self.label["atr_scale"]
            if isinstance(atr_scale, bool) or not isinstance(atr_scale, (int, float)) or atr_scale <= 0:
                raise ValueError(
                    f"label 'triple_barrier' key 'atr_scale' must be a float > 0, got {atr_scale!r}"
                )
        extra = sorted(
            set(self.label)
            - {"name", "horizon", "up", "down", "conflict_resolution", "peak_filter", "atr_scale"}
        )
        if extra:
            raise ValueError(f"label 'triple_barrier' has unknown keys {extra}")

    def _validate_generic_label(self, name: str) -> None:
        """Validate any `GENERIC_LABEL_NAMES` entry: `horizon` (int >= 1) is required;
        every other key (e.g. `min_gain`, `min_loss`, `max_drawdown`, `top_quantile`,
        `min_ratio`, `min_sharpe`, `atr_scale`) must be a float/int > 0 -- which key names a
        given labeler actually accepts is enforced later, at label-generation time, by
        `labels.ternary_labels_generic`'s signature filtering (research: config-time
        validation only needs to reject non-numeric junk, not know every labeler's
        signature)."""
        if "horizon" not in self.label:
            raise ValueError(f"label {name!r} missing required key 'horizon'")
        horizon = self.label["horizon"]
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError(
                f"label {name!r} key 'horizon' must be an int >= 1, got {horizon!r}"
            )
        for key, value in self.label.items():
            if key in ("name", "horizon"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"label {name!r} key {key!r} must be a float > 0, got {value!r}"
                )


def config_id(cfg: ExperimentConfig) -> str:
    """First 12 hex chars of the SHA-1 over the canonical id-field JSON (research R9).

    `label` is included only when it differs from `DEFAULT_LABEL` (`{"name": "gbb"}`), so
    every config written before `label` existed -- and every config that still means "the
    production gbb label" -- hashes to exactly the id it always had.
    """
    payload = {
        "arch": cfg.arch,
        "hp": cfg.hp,
        "input_set": cfg.input_set,
        "seq_len": cfg.seq_len,
        "split_set": cfg.split_set,
        "train": cfg.train,
    }
    if cfg.label != DEFAULT_LABEL:
        payload["label"] = cfg.label
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def load_phase(path: str | Path) -> Tuple[Dict[str, Any], List[ExperimentConfig]]:
    """Load a phase YAML file: `{phase, description, gate_from, configs: [...]}`.

    Every entry is validated before it is returned (data-model.md §Phase file).
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    entries = doc.get("configs", []) or []
    header = {k: v for k, v in doc.items() if k != "configs"}

    configs: List[ExperimentConfig] = []
    for entry in entries:
        entry = dict(entry)
        cfg = ExperimentConfig(
            arch=entry["arch"],
            hp=entry.get("hp", {}),
            input_set=entry.get("input_set", "ind24"),
            seq_len=entry.get("seq_len", 16),
            split_set=entry.get("split_set", "v1"),
            train=entry.get("train", {}),
            phase=entry.get("phase", header.get("phase", 0)),
            seeds=entry.get("seeds", [0]),
            tags=entry.get("tags", []),
            label=entry.get("label", dict(DEFAULT_LABEL)),
        )
        cfg.validate()
        configs.append(cfg)

    return header, configs


def build(
    cfg: ExperimentConfig,
    nf: int,
    seq_len: int,
    seed: int,
    checkpoint_dir: str | Path,
    *,
    window_id: str,
):
    """Construct a fully-configured (untrained) classifier for `cfg` at seed `seed`.

    Sets hyperparameters (`clf.hp`), the training seed, the optimizer knobs from
    `cfg.train`, and routes the checkpoint to `checkpoint_dir` (see the note below on why
    that needs more than setting `clf.checkpoint_path`).

    `window_id` is required (keyword-only) and is part of the checkpoint filename: a bare
    `<config_id>_<seed>.safetensors` collides across windows A/B/C of the same
    (config, seed) within a phase run, so window B would `train()`-resume from window A's
    saved weights instead of starting from scratch (and C from B's).
    """
    clf, _ = create_classifier_mlx(resolve(cfg.arch), "POOLED", nf, seq_len, NUM_CLASSES)

    clf.hp = dict(cfg.hp)
    clf.train_seed = seed
    clf.max_epochs = cfg.train["max_epochs"]
    clf.early_patience = cfg.train["patience"]
    clf.weight_decay = cfg.train["weight_decay"]
    clf.set_batch_size(cfg.train["batch_size"])
    clf.set_learning_rate(cfg.train["lr"])

    checkpoint_dir = str(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir, f"{config_id(cfg)}_{window_id}_{seed}.safetensors"
    )

    # `MLXBasePredictor.set_model_path()` also sets `self.name`/`self.root_dir`, which
    # `save()` (called unconditionally at the end of `train()`) needs to be non-empty.
    clf.set_model_path(ckpt_path)

    # `MLXBasePredictor.get_checkpoint_path()` *ignores* `self.checkpoint_path` and always
    # recomputes `/tmp/<self.name>/checkpoint.<ext>` — and `train()` calls the *method*,
    # not the attribute, both to save the best-epoch weights and to restore them at the
    # end. Setting `clf.checkpoint_path` alone is therefore a no-op for where the
    # checkpoint actually lands. MLXBasePredictor.py is out of scope for this change, so
    # the bound method is overridden on this instance instead — the only way to redirect
    # it without touching that file.
    clf.checkpoint_path = ckpt_path
    clf.get_checkpoint_path = lambda: ckpt_path

    return clf


def param_count(clf) -> int:
    """Trainable-parameter count of `clf.model` (valid only after `train()` has run once,
    since model creation is lazy — `clf.model` is `None` beforehand)."""
    from mlx.utils import tree_flatten

    return sum(p.size for _, p in tree_flatten(clf.model.trainable_parameters()))
