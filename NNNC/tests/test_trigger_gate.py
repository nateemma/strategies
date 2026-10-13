"""T034 — NNNC_TriggerGate runs on arm D's fold models and can never train (023 research R12).

Since 030 the fold models are found by MODEL KEY (the twin differs from arm D only after the
model), not by BorrowedModelMixin's class-name redirect; strict walk-forward evaluation refuses
to train a missing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation import fingerprint as fp


STRAT = Path(__file__).resolve().parents[2]


def _gate():
    return fp.load_strategy_class("NNNC_TriggerGate")


CFG = fp.load_config(STRAT / "config" / "config_wf.json")
TRAIN = {"f03": "20210101-20230928"}  # arm D's f03 training window (plan_NNNC_A_Raw_Open)


def _fold_class(fid="f03"):
    # what run_folds generates: the twin pinned to the fold's training window and seed
    return type(f"NNNC_TGT_{fid}", (_gate(),), {"MODEL_TRAIN_TIMERANGE": TRAIN[fid],
                                                 "MODEL_SEED": 1})


def _inst(cls):
    inst = object.__new__(cls)
    inst.config = CFG
    return inst


def test_model_path_points_at_arm_d_fold_model():
    p = Path(_inst(_fold_class("f03")).get_model_path())
    assert p == STRAT / "saved_data/NNNC_ARO_f03/NNNC_ARO_f03.keras"
    assert any(p.parent.glob("*.safetensors"))
    assert not (STRAT / "saved_data/NNNC_TGT_f03").exists()


def test_unpinned_base_class_has_no_model_and_strict_mode_refuses(monkeypatch):
    from Framework import ModelKey as MK

    monkeypatch.setenv("FT_MODEL_EVAL_STRICT", "1")
    with pytest.raises(MK.ModelMissing, match="refuses to train"):
        _inst(_gate()).get_model_path()


def test_the_retired_mixin_cannot_be_used():
    import sys as _sys

    _sys.path.insert(0, str(STRAT / "NNNC"))
    from BorrowedModelMixin import BorrowedModelMixin

    with pytest.raises(TypeError, match="retired"):
        type("X", (BorrowedModelMixin,), {})


def test_framework_equals_arm_d():
    cfg = fp.load_config(fp.STRAT_DIR / "config" / "config_wf.json")
    g = fp.resolve(_gate(), fp.load_params_json("NNNC_TriggerGate"), cfg)[0]
    d = fp.resolve(
        fp.load_strategy_class("NNNC_A_Raw_Open"), fp.load_params_json("NNNC_A_Raw_Open"), cfg
    )[0]
    assert g == d
