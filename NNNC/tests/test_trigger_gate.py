"""T034 — NNNC_TriggerGate borrows arm D's fold models and can never train (023 research R12)."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation import fingerprint as fp


STRAT = Path(__file__).resolve().parents[2]


def _gate():
    return fp.load_strategy_class("NNNC_TriggerGate")


def _fold_class(fid="f03"):
    return type(f"NNNC_TGT_{fid}", (_gate(),), {})


def test_model_path_points_at_arm_d_fold_model():
    inst = object.__new__(_fold_class("f03"))
    p = Path(inst.get_model_path())
    assert p == STRAT / "saved_data/NNNC_ARO_f03/NNNC_ARO_f03.keras"
    assert any(p.parent.glob("*.safetensors"))
    assert not (STRAT / "saved_data/NNNC_TGT_f03").exists()


def test_base_class_without_fold_suffix_refuses():
    with pytest.raises(ValueError):
        object.__new__(_gate()).get_model_path()


def test_training_entry_point_raises():
    with pytest.raises(RuntimeError, match="must never train"):
        object.__new__(_fold_class()).train_model([], [], None)


def test_framework_equals_arm_d():
    cfg = fp.load_config(fp.STRAT_DIR / "config" / "config_wf.json")
    g = fp.resolve(_gate(), fp.load_params_json("NNNC_TriggerGate"), cfg)[0]
    d = fp.resolve(
        fp.load_strategy_class("NNNC_A_Raw_Open"), fp.load_params_json("NNNC_A_Raw_Open"), cfg
    )[0]
    assert g == d
