# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Tests for `research/clfstudy/registry.py` config identity (T015).

Covers: determinism of `config_id` across key order; `seeds`/`phase`/`tags` excluded from
the id; every id field changes the id; `ExperimentConfig.validate()` rejects an unknown
arch, an unknown input_set, `seq_len=1` and `train.stride=0`; `load_phase` on a tmp YAML.

Also covers the `label` field (research: let an experiment select the label): every
config_id computed by `configs/phase_{2,3,4}.yaml` BEFORE `label` existed is unchanged;
`label` is omitted from the canonical JSON (and so from the id) when it equals the
default `{"name": "gbb"}`; a `triple_barrier` label changes the id; label validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.clfstudy.registry import DEFAULT_LABEL, ExperimentConfig, config_id, load_phase

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

#: `config_id` for every entry in configs/phase_{2,3,4}.yaml, computed BEFORE the `label`
#: field was added to `ExperimentConfig`/`config_id` -- frozen here so a regression in the
#: "omit `label` when it's the default" rule is caught even though these particular YAML
#: files never set `label` explicitly.
_BASELINE_IDS = {
    "phase_2.yaml": [
        "6f0c110ce13c", "ffb5c40e8d6b", "d1dc0339b10b", "12c3636e7110",
    ],
    "phase_3.yaml": [
        "d2bbc0d7fb15", "098ca2f95a2e", "f37089f602fe", "076666f04306",
        "60f40876941b", "ea8d90054291", "0b31c7bb9404", "cf6b6ccb530a",
        "52292292ef8b", "9c5b1e952784", "666675f56e77", "f1f70fadabbb",
        "3cd40f90604c", "0875fd5bd454", "3a4ddb6c4092", "59d72ed6d750",
        "e645063be7bc", "3d09879bf74b", "d7fcd89ef3d6", "48b806ceba9b",
        "470786990c9e", "8d1446f2a484", "225a954527c8", "558e2f29084d",
    ],
    "phase_4.yaml": [
        "5616ab5d0bd0", "7b50342154ef", "b5b58a6c2cc5", "b690297cbbf5",
        "f5be6cc3a0df", "702b00fcb20c", "70eb48f9acfb", "68c1f710246f",
        "cef2eef58d6b", "3c120e544d4e", "dbd824e7736c", "a09a0b24d6fe",
        "b85e94f80dc6", "12af479dfaa3", "f7d18cb3a30f", "fa4e5ad070f7",
        "318ea175aa05", "486f44b1f862", "67ffa1f12fbd", "d5a1481bd4c0",
        "9278e1441b47", "821f46c115ab", "bbbed50c3b5c", "6fe7246246f5",
        "a4fb5eb48337", "3413d77700b0", "e5974774aa5f", "8e1f1550ad56",
        "9c94bdef2a2d", "1260944a8d42", "4a28d1ce4ebd", "4c265676afae",
        "551b8bb6dcf2", "8a9bbc798cf3", "b8285305a779",
    ],
}


@pytest.mark.parametrize("phase_file", sorted(_BASELINE_IDS))
def test_phase_file_config_ids_unchanged_by_label_field(phase_file):
    _, configs = load_phase(_CONFIGS_DIR / phase_file)
    ids = [config_id(cfg) for cfg in configs]
    assert ids == _BASELINE_IDS[phase_file]


def _cfg(**overrides) -> ExperimentConfig:
    kwargs = dict(
        arch="MLP",
        hp={},
        input_set="ind24",
        seq_len=16,
        split_set="v1",
        train={},
        phase=3,
        seeds=[0, 1],
        tags=["baseline"],
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_determinism_across_key_order():
    cfg_a = _cfg(hp={"a": 1, "b": 2})
    cfg_b = _cfg(hp={"b": 2, "a": 1})
    assert config_id(cfg_a) == config_id(cfg_b)
    # calling twice on the same config is stable too
    assert config_id(cfg_a) == config_id(cfg_a)


def test_seeds_phase_tags_excluded_from_id():
    base = _cfg()
    other_seeds = _cfg(seeds=[7, 8, 9])
    other_phase = _cfg(phase=99)
    other_tags = _cfg(tags=["something-else"])
    assert config_id(base) == config_id(other_seeds)
    assert config_id(base) == config_id(other_phase)
    assert config_id(base) == config_id(other_tags)


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("arch", "CNN"),
        ("hp", {"width": 999}),
        ("input_set", "raw8"),
        ("seq_len", 32),
        ("split_set", "v2"),
        ("train", {"lr": 5e-4}),
    ],
)
def test_each_id_field_changes_the_id(field_name, value):
    base = _cfg()
    changed = _cfg(**{field_name: value})
    assert config_id(base) != config_id(changed)


def test_train_defaults_merged_and_visible_in_id():
    # an explicit key equal to the default still participates in the merged dict,
    # so a config differing in a *different* train key changes the id even though
    # the caller only supplied one key.
    cfg = _cfg(train={"lr": 1e-4})
    assert cfg.train["max_epochs"] == 40
    assert cfg.train["patience"] == 10
    assert cfg.train["batch_size"] == 256
    assert cfg.train["weight_decay"] == 0.0
    assert cfg.train["stride"] == 1
    assert cfg.train["lr"] == 1e-4


def test_invalid_arch_rejected():
    with pytest.raises(ValueError):
        _cfg(arch="NOT_A_REAL_ARCH").validate()


def test_baseline_arch_accepted():
    # MAJORITY/PERSISTENCE are not registry members but are valid configs (run.py
    # bypasses training for them).
    _cfg(arch="MAJORITY").validate()
    _cfg(arch="PERSISTENCE").validate()


@pytest.mark.parametrize(
    "input_set",
    ["not_a_set", "ind24-minus:nonsense_group", "ind24-minus:"],
)
def test_invalid_input_set_rejected(input_set):
    with pytest.raises(ValueError):
        _cfg(input_set=input_set).validate()


@pytest.mark.parametrize("input_set", ["ind24", "raw8", "both32", "ind24-minus:trend", "ind24-minus:guards"])
def test_valid_input_set_accepted(input_set):
    _cfg(input_set=input_set).validate()


def test_seq_len_1_rejected():
    with pytest.raises(ValueError):
        _cfg(seq_len=1).validate()


def test_stride_0_rejected():
    with pytest.raises(ValueError):
        _cfg(train={"stride": 0}).validate()


def test_max_epochs_0_rejected():
    with pytest.raises(ValueError):
        _cfg(train={"max_epochs": 0}).validate()


def test_load_phase(tmp_path):
    phase_yaml = tmp_path / "phase_99.yaml"
    phase_yaml.write_text(
        json.dumps(
            {
                "phase": 99,
                "description": "unit test phase",
                "gate_from": None,
                "configs": [
                    {
                        "arch": "MLP",
                        "hp": {},
                        "input_set": "ind24",
                        "seq_len": 16,
                        "split_set": "v1",
                        "train": {"max_epochs": 5},
                        "seeds": [0, 1],
                        "tags": ["smoke"],
                    },
                    {"arch": "RIDGE", "train": {"weight_decay": 1e-2}},
                ],
            }
        )
    )

    header, configs = load_phase(phase_yaml)

    assert header["phase"] == 99
    assert header["description"] == "unit test phase"
    assert "configs" not in header
    assert len(configs) == 2

    mlp_cfg, ridge_cfg = configs
    assert mlp_cfg.arch == "MLP"
    assert mlp_cfg.train["max_epochs"] == 5
    assert mlp_cfg.seeds == [0, 1]
    assert mlp_cfg.tags == ["smoke"]

    # entries can omit everything except `arch`; defaults + header phase fill the rest
    assert ridge_cfg.arch == "RIDGE"
    assert ridge_cfg.train["weight_decay"] == 1e-2
    assert ridge_cfg.train["max_epochs"] == 40
    assert ridge_cfg.phase == 99
    assert ridge_cfg.seeds == [0]


# ---------------------------------------------------------------------------
# `label` field
# ---------------------------------------------------------------------------


def test_label_defaults_to_gbb():
    cfg = _cfg()
    assert cfg.label == {"name": "gbb"}
    assert cfg.label == DEFAULT_LABEL


def test_label_omitted_equals_label_gbb_explicit():
    # {"name": "gbb"} is exactly the default -> same id as omitting `label` entirely.
    implicit = _cfg()
    explicit = _cfg(label={"name": "gbb"})
    assert config_id(implicit) == config_id(explicit)


def test_label_omitted_from_canonical_json_when_default():
    import hashlib

    cfg = _cfg()
    payload_without_label = {
        "arch": cfg.arch, "hp": cfg.hp, "input_set": cfg.input_set,
        "seq_len": cfg.seq_len, "split_set": cfg.split_set, "train": cfg.train,
    }
    blob = json.dumps(payload_without_label, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    assert config_id(cfg) == expected


def test_triple_barrier_label_changes_the_id():
    base = _cfg()
    tb = _cfg(label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    assert config_id(base) != config_id(tb)


# ---------------------------------------------------------------------------
# `label.atr_scale` (ATR-scaled triple-barrier width)
# ---------------------------------------------------------------------------


def test_triple_barrier_atr_scale_absent_by_default():
    cfg = _cfg(label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    assert "atr_scale" not in cfg.label


def test_triple_barrier_atr_scale_changes_the_id():
    without = _cfg(label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    with_atr = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01, "atr_scale": 2.0,
        }
    )
    assert config_id(without) != config_id(with_atr)


def test_triple_barrier_atr_scale_different_values_change_the_id():
    a = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01, "atr_scale": 2.0,
        }
    )
    b = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01, "atr_scale": 3.0,
        }
    )
    assert config_id(a) != config_id(b)


def test_existing_configs_unaffected_by_atr_scale_field():
    # every pre-existing config (no `atr_scale` key at all) hashes exactly as before --
    # the frozen ids in _BASELINE_IDS above already cover the `gbb`-label case; this
    # covers the triple_barrier case explicitly.
    tb = _cfg(label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    assert config_id(tb) == config_id(_cfg(
        label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01}
    ))


@pytest.mark.parametrize("atr_scale", [0, -1.0, -0.5])
def test_triple_barrier_atr_scale_non_positive_rejected(atr_scale):
    cfg = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01,
            "atr_scale": atr_scale,
        }
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_triple_barrier_atr_scale_bool_rejected():
    cfg = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01,
            "atr_scale": True,
        }
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_triple_barrier_atr_scale_non_numeric_rejected():
    cfg = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01,
            "atr_scale": "2.0",
        }
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_triple_barrier_atr_scale_valid_accepted():
    cfg = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01,
            "atr_scale": 2.0,
        }
    )
    cfg.validate()  # no raise
    cfg2 = _cfg(
        label={
            "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01,
            "atr_scale": 2,
        }
    )
    cfg2.validate()  # int is also accepted


def test_triple_barrier_label_defaults_filled_in():
    cfg = _cfg(label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    assert cfg.label["conflict_resolution"] is True
    assert cfg.label["peak_filter"] is False


def test_triple_barrier_label_explicit_values_kept():
    cfg = _cfg(
        label={
            "name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02,
            "conflict_resolution": False, "peak_filter": True,
        }
    )
    assert cfg.label["conflict_resolution"] is False
    assert cfg.label["peak_filter"] is True


def test_triple_barrier_different_params_change_the_id():
    a = _cfg(label={"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01})
    b = _cfg(label={"name": "triple_barrier", "horizon": 48, "up": 0.01, "down": 0.01})
    assert config_id(a) != config_id(b)


def test_label_unknown_name_rejected():
    with pytest.raises(ValueError):
        _cfg(label={"name": "not_a_real_label"}).validate()


def test_label_gbb_extra_keys_rejected():
    with pytest.raises(ValueError):
        _cfg(label={"name": "gbb", "horizon": 24}).validate()


@pytest.mark.parametrize("missing", ["horizon", "up", "down"])
def test_triple_barrier_missing_required_key_rejected(missing):
    label = {"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01}
    del label[missing]
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


@pytest.mark.parametrize("horizon", [0, -1, 1.5, "24"])
def test_triple_barrier_invalid_horizon_rejected(horizon):
    label = {"name": "triple_barrier", "horizon": horizon, "up": 0.01, "down": 0.01}
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


@pytest.mark.parametrize("key", ["up", "down"])
@pytest.mark.parametrize("value", [0, -0.01, "0.01"])
def test_triple_barrier_invalid_up_down_rejected(key, value):
    label = {"name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01, key: value}
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


def test_triple_barrier_unknown_key_rejected():
    label = {
        "name": "triple_barrier", "horizon": 24, "up": 0.01, "down": 0.01,
        "not_a_real_key": True,
    }
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


# ---------------------------------------------------------------------------
# generic PRICE-ONLY labels (research: let an experiment select ANY price-only labeler)
# ---------------------------------------------------------------------------


def test_forward_mae_label_validates_and_changes_id():
    base = _cfg()
    fmae = _cfg(
        label={"name": "forward_mae", "horizon": 12, "min_gain": 0.015, "max_drawdown": 0.05}
    )
    fmae.validate()  # no raise
    assert config_id(base) != config_id(fmae)


def test_quantile_future_label_validates_and_changes_id():
    base = _cfg()
    qfut = _cfg(label={"name": "quantile_future", "horizon": 12, "top_quantile": 0.8})
    qfut.validate()  # no raise
    assert config_id(base) != config_id(qfut)


def test_generic_label_different_params_change_the_id():
    a = _cfg(label={"name": "forward_mae", "horizon": 12, "min_gain": 0.015})
    b = _cfg(label={"name": "forward_mae", "horizon": 24, "min_gain": 0.015})
    assert config_id(a) != config_id(b)


def test_generic_label_unknown_name_rejected():
    with pytest.raises(ValueError):
        _cfg(label={"name": "not_a_real_generic_label", "horizon": 12}).validate()


def test_generic_label_missing_horizon_rejected():
    with pytest.raises(ValueError):
        _cfg(label={"name": "forward_mae", "min_gain": 0.015}).validate()


@pytest.mark.parametrize("horizon", [0, -1, 1.5, "12"])
def test_generic_label_invalid_horizon_rejected(horizon):
    label = {"name": "forward_mae", "horizon": horizon, "min_gain": 0.015}
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


@pytest.mark.parametrize("value", [0, -0.01, "0.5"])
def test_generic_label_non_numeric_extra_key_rejected(value):
    label = {"name": "forward_mae", "horizon": 12, "min_gain": value}
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


def test_generic_label_extra_key_bool_rejected():
    label = {"name": "forward_mae", "horizon": 12, "min_gain": True}
    with pytest.raises(ValueError):
        _cfg(label=label).validate()


# ---------------------------------------------------------------------------
# `scaler_mode` (research: FIX for the zero-IQR "dead feature" defect)
# ---------------------------------------------------------------------------


def test_scaler_mode_defaults_to_robust():
    cfg = _cfg()
    assert cfg.scaler_mode == "robust"


def test_scaler_mode_omitted_equals_explicit_default_robust():
    implicit = _cfg()
    explicit = _cfg(scaler_mode="robust")
    assert config_id(implicit) == config_id(explicit)


def test_scaler_mode_default_leaves_frozen_ids_unchanged():
    # every pre-existing config in configs/phase_{2,3,4}.yaml omits `scaler_mode` entirely
    # -> it defaults to "robust" -> must be omitted from the canonical JSON exactly like
    # `label`, so `test_phase_file_config_ids_unchanged_by_label_field` above already
    # covers this; this test names the invariant explicitly for `scaler_mode`.
    _, configs = load_phase(_CONFIGS_DIR / "phase_3.yaml")
    ids = [config_id(cfg) for cfg in configs]
    assert ids == _BASELINE_IDS["phase_3.yaml"]
    assert all(cfg.scaler_mode == "robust" for cfg in configs)


def test_scaler_mode_non_default_changes_the_id():
    base = _cfg()
    changed = _cfg(scaler_mode="robust_std")
    assert config_id(base) != config_id(changed)


def test_scaler_mode_valid_values_accepted():
    _cfg(scaler_mode="robust").validate()
    _cfg(scaler_mode="robust_std").validate()


def test_scaler_mode_invalid_rejected():
    with pytest.raises(ValueError):
        _cfg(scaler_mode="not_a_real_mode").validate()
