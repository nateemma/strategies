"""T009 — `foldplan derive` and the no-model validate rule (023 contract C2).

A SimpleTrigger arm has no model, so every fold legitimately uses the SAME class and NO model
dir. `validate()` was written for NNNC, where both would be collisions (two folds overwriting
one model). The exemption must apply only to a plan that declares itself model-free.
"""
from __future__ import annotations

import json

import pytest

from evaluation import foldplan

SRC = foldplan.ROOT_ABS / "user_data/strategies/evaluation/folds/plan_NNNC_A_Raw_Open.json"
COPIED = ("fold_id", "train_start", "train_end", "test_start", "test_end",
          "train_timerange", "test_timerange", "test_days")


@pytest.fixture(scope="module")
def src():
    return json.loads(SRC.read_text())


def test_derive_copies_windows_verbatim(src):
    plan = foldplan.derive(src, family="SimpleTrigger", base="X", tag="st_x")
    assert plan["config"] == src["config"]
    assert len(plan["folds"]) == len(src["folds"])
    for new, old in zip(plan["folds"], src["folds"]):
        for k in COPIED:
            assert new[k] == old[k], (new["fold_id"], k)


def test_derive_no_model_shape(src):
    plan = foldplan.derive(src, family="SimpleTrigger", base="X", tag="st_x")
    assert plan["no_model"] is True
    assert plan["params_snapshot"] is None and plan["params_sha"] is None
    assert plan["validated"] is True and plan["validation_errors"] == []
    assert "thresholds" not in plan
    for f in plan["folds"]:
        assert f["model_dir"] is None
        assert f["strategy_class"] == "X"
        assert f["arm_tag"] == f"st_x_{f['fold_id']}"
    assert plan["run_id"] != src["run_id"]          # never overwrite the source's results file


def test_validate_exempts_only_no_model_plans(src):
    plan = foldplan.derive(src, family="SimpleTrigger", base="X", tag="st_x")
    assert foldplan.validate(plan["folds"], no_model=True) == []
    errs = foldplan.validate(plan["folds"], no_model=False)
    assert any("strategy_class" in e for e in errs)
    assert any("model_dir" in e for e in errs)
    # no_model with a model_dir set is still an error
    bad = [dict(f, model_dir="x") for f in plan["folds"]]
    assert foldplan.validate(bad, no_model=True)


def test_no_model_plan_with_thresholds_is_rejected(src):
    plan = foldplan.derive(src, family="SimpleTrigger", base="X", tag="st_x")
    assert foldplan.validate_plan(plan) == []
    plan["thresholds"] = {"f00": 0.5}
    assert foldplan.validate_plan(plan)


def test_derive_nnnc_with_model_source(src, tmp_path):
    params = tmp_path / "NNNC_TriggerGate.json"
    params.write_text(json.dumps({"strategy_name": "NNNC_TriggerGate", "params": {}}))
    plan = foldplan.derive(src, family="NNNC", base="NNNC_TriggerGate", tag="wf_trgate",
                           model_source="NNNC_ARO", class_prefix="NNNC_TGT",
                           params_file=params, snapshot_dir=tmp_path)
    assert plan.get("no_model") is False
    for f in plan["folds"]:
        fid = f["fold_id"]
        assert f["strategy_class"] == f"NNNC_TGT_{fid}"
        assert f["model_dir"].endswith(f"saved_data/NNNC_ARO_{fid}")
        assert f["arm_tag"] == f"wf_trgate_{fid}"
    assert plan["params_snapshot"] and plan["params_sha"]
    assert foldplan.validate_plan(plan) == []


def test_derive_nnnc_requires_params_file(src, tmp_path):
    with pytest.raises(FileNotFoundError):
        foldplan.derive(src, family="NNNC", base="NNNC_Nope", tag="t", model_source="NNNC_ARO",
                        class_prefix="NNNC_NOPE", params_file=tmp_path / "missing.json",
                        snapshot_dir=tmp_path)
