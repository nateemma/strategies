# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Tests for `research/clfstudy/registry.py` config identity (T015).

Covers: determinism of `config_id` across key order; `seeds`/`phase`/`tags` excluded from
the id; every id field changes the id; `ExperimentConfig.validate()` rejects an unknown
arch, an unknown input_set, `seq_len=1` and `train.stride=0`; `load_phase` on a tmp YAML.
"""
from __future__ import annotations

import json

import pytest

from research.clfstudy.registry import ExperimentConfig, config_id, load_phase


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
