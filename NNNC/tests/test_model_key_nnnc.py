"""030 T021/T025: NNNC variants share their model's weights by key -- no BorrowedModelMixin.

Classes are loaded, never instantiated; keys are resolved against config_wf.json with a fold's
training window and seed pinned, exactly as evaluation/run_folds.py's generated fold classes pin
them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from evaluation.fingerprint import load_config, load_strategy_class  # noqa: E402
from Framework import ModelKey as MK  # noqa: E402


CFG = load_config(STRAT / "config" / "config_wf.json")
PLAN = json.loads((STRAT / "evaluation/folds/plan_NNNC_A_Raw_Open.json").read_text())
F03 = next(f for f in PLAN["folds"] if f["fold_id"] == "f03")


def _fold(cls, fold=F03, seed=1):
    return type(f"{cls.__name__}_{fold['fold_id']}", (cls,),
                {"MODEL_TRAIN_TIMERANGE": fold["train_timerange"], "MODEL_SEED": seed,
                 "__module__": cls.__module__})


def _key(cls):
    return MK.key_hash(MK.resolve(_fold(cls), CFG))


ARM_D = load_strategy_class("NNNC_A_Raw_Open")


def test_an_exit_only_variant_of_arm_d_resolves_to_arm_ds_key():
    variant = type("ExitVariant", (ARM_D,), {
        "FAMILY_EXIT_OVERRIDES": {"reversion": {"take_profit_atr_mult": 2.8970711921574543}},
        "__module__": ARM_D.__module__})
    assert _key(variant) == _key(ARM_D)


def test_a_renamed_copy_of_arm_d_resolves_to_the_same_key():
    renamed = type("SomethingElseEntirely", (ARM_D,), {"__module__": ARM_D.__module__})
    assert _key(renamed) == _key(ARM_D)


def test_a_gain_target_change_is_a_different_model():
    other = type("GainChanged", (ARM_D,), {"MIN_BUY_GAIN_THRESHOLD": 0.03,
                                           "__module__": ARM_D.__module__})
    assert _key(other) != _key(ARM_D)


def test_a_different_fold_or_seed_is_a_different_model():
    f04 = next(f for f in PLAN["folds"] if f["fold_id"] == "f04")
    k = MK.key_hash(MK.resolve(_fold(ARM_D, F03), CFG))
    assert MK.key_hash(MK.resolve(_fold(ARM_D, f04), CFG)) != k
    assert MK.key_hash(MK.resolve(_fold(ARM_D, F03, seed=7), CFG)) != k


def test_gate_arm_with_its_startup_pin_resolves_to_arm_ds_key():
    gate = load_strategy_class("NNNC_SW_ARO_Gate")
    if not MK.merged_dict(gate, "MODEL_TRAINED_WITH"):
        pytest.skip("gate arm not yet pinned (T024)")
    assert _key(gate) == _key(ARM_D)


@pytest.mark.parametrize("name,source", [
    ("NNNC_TriggerGate", "NNNC_A_Raw_Open"), ("NNNC_SIProbe", "NNNC_A_Raw_Open"),
    ("NNNC_Live_C2", "NNNC_A_Raw_Open"), ("NNNC_Live_TrigGate", "NNNC_A_Raw_Open"),
    ("NNNC_SW_ARO_C2", "NNNC_A_Raw_Open"), ("NNNC_SW_GBR_C2", "NNNC_GbbRaw_Open"),
    ("NNNC_SW_GFM_C2", "NNNC_GbbFwdMae_Open"), ("NNNC_SW_MLX_C2", "NNNC_MLX"),
    ("NNNC_SW_ARO_Gate", "NNNC_A_Raw_Open"), ("NNNC_SW_ARO_GateOff", "NNNC_A_Raw_Open"),
    ("NNNC_Live_Gate", "NNNC_A_Raw_Open"), ("NNNC_Live_GateOff", "NNNC_A_Raw_Open"),
    ("NNNC_SW_GBR_Gate", "NNNC_GbbRaw_Open"), ("NNNC_SW_GBR_GateOff", "NNNC_GbbRaw_Open"),
    ("NNNC_SW_GFM_Gate", "NNNC_GbbFwdMae_Open"), ("NNNC_SW_GFM_GateOff", "NNNC_GbbFwdMae_Open"),
    ("NNNC_SW_MLX_Gate", "NNNC_MLX"), ("NNNC_SW_MLX_GateOff", "NNNC_MLX"),
])
def test_former_borrowed_users_resolve_to_their_source_models_key(name, source):
    cls = load_strategy_class(name)
    if any(k.__name__ == "BorrowedModelMixin" for k in cls.__mro__):
        pytest.skip("still borrowing (before T024)")
    assert _key(cls) == _key(load_strategy_class(source)), name
