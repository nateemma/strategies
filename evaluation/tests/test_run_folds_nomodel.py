"""T011 — run_folds' no-model path, model_dir-based model check, causality preflight (023 C2).

The real loop runs with every side-effecting step monkeypatched, so these tests observe exactly
which steps a plan triggers without generating, training or backtesting anything.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from evaluation import foldplan, run_folds

SRC = foldplan.ROOT_ABS / "user_data/strategies/evaluation/folds/plan_NNNC_A_Raw_Open.json"


class _Calls:
    def __init__(self):
        self.generate, self.train, self.model_exists, self.evaluate = [], [], [], []


@pytest.fixture
def harness(tmp_path, monkeypatch):
    calls = _Calls()
    (tmp_path / "evaluation/folds").mkdir(parents=True)
    (tmp_path / "regime/arms").mkdir(parents=True)
    monkeypatch.setattr(run_folds, "STRAT_DIR", tmp_path)

    def fake_generate(f, *a, **k):
        calls.generate.append(f["fold_id"])
        return None, None

    def fake_train(f, *a, **k):
        calls.train.append(f["fold_id"])
        return True

    def fake_model_exists(model_dir_or_cls):
        calls.model_exists.append(model_dir_or_cls)
        return True

    def fake_evaluate(f, config):
        calls.evaluate.append((f["fold_id"], f["strategy_class"]))
        rec = {"returncode": 0, "n_trades": 5}
        (tmp_path / "regime/arms" / f"{f['arm_tag']}_{f['fold_id'].upper()}.json").write_text(
            json.dumps(rec))
        return 0

    monkeypatch.setattr(run_folds, "generate_fold", fake_generate)
    monkeypatch.setattr(run_folds, "train_fold", fake_train)
    monkeypatch.setattr(run_folds, "model_exists", fake_model_exists)
    monkeypatch.setattr(run_folds, "evaluate_fold", fake_evaluate)
    return calls, tmp_path


def _write_plan(tmp_path, plan):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    return p


def _fake_class(monkeypatch, needs_training=False, trigger="gbb"):
    cls = types.SimpleNamespace(
        strategy_config=types.SimpleNamespace(needs_training=needs_training),
        TRIGGER_NAME=trigger)
    monkeypatch.setattr(run_folds, "_load_strategy_class", lambda name: cls)
    return cls


def _run(monkeypatch, plan_path):
    monkeypatch.setattr(sys, "argv", ["run_folds", "--plan", str(plan_path)])
    return run_folds.main()


def test_no_model_plan_skips_generate_train_and_model_check(harness, monkeypatch):
    calls, tmp = harness
    _fake_class(monkeypatch)
    monkeypatch.setattr(run_folds, "_trigger_runnable", lambda name: (True, "PASS"))
    plan = foldplan.derive(json.loads(SRC.read_text()), "SimpleTrigger", "X", "st_x")
    assert _run(monkeypatch, _write_plan(tmp, plan)) == 0
    assert calls.generate == [] and calls.train == [] and calls.model_exists == []
    assert len(calls.evaluate) == len(plan["folds"]) == 15
    assert {c for _, c in calls.evaluate} == {"X"}


def test_no_model_plan_aborts_if_class_would_train(harness, monkeypatch):
    calls, tmp = harness
    _fake_class(monkeypatch, needs_training=True)
    monkeypatch.setattr(run_folds, "_trigger_runnable", lambda name: (True, "PASS"))
    plan = foldplan.derive(json.loads(SRC.read_text()), "SimpleTrigger", "X", "st_x")
    assert _run(monkeypatch, _write_plan(tmp, plan)) != 0
    assert calls.evaluate == []


def test_no_model_plan_aborts_on_non_pass_trigger_before_any_evaluation(harness, monkeypatch):
    calls, tmp = harness
    _fake_class(monkeypatch, trigger="local_min_followthrough")
    monkeypatch.setattr(run_folds, "_trigger_runnable",
                        lambda name: (False, "FAIL in trigger_causality.json"))
    plan = foldplan.derive(json.loads(SRC.read_text()), "SimpleTrigger", "X", "st_x")
    assert _run(monkeypatch, _write_plan(tmp, plan)) != 0
    assert calls.evaluate == []


def test_nnnc_plan_checks_model_dir_not_class(harness, monkeypatch, tmp_path):
    calls, tmp = harness
    params = tmp_path / "NNNC_TriggerGate.json"
    params.write_text(json.dumps({"strategy_name": "NNNC_TriggerGate", "params": {}}))
    plan = foldplan.derive(json.loads(SRC.read_text()), "NNNC", "NNNC_TriggerGate", "wf_tg",
                           model_source="NNNC_ARO", class_prefix="NNNC_TGT",
                           params_file=params, snapshot_dir=tmp_path)
    plan["params_snapshot"] = str(tmp_path / next(p.name for p in tmp_path.glob("params_*")))
    assert _run(monkeypatch, _write_plan(tmp, plan)) == 0
    checked = set(calls.model_exists)
    assert all(c.startswith("NNNC_ARO_f") for c in checked), checked
    assert not any(c.startswith("NNNC_TGT") for c in checked)
    assert calls.train == []


def test_model_exists_uses_model_dir_basename(tmp_path, monkeypatch):
    monkeypatch.setattr(run_folds, "SAVED", tmp_path)
    (tmp_path / "NNNC_ARO_f03").mkdir()
    (tmp_path / "NNNC_ARO_f03" / "NNNC_ARO_f03.safetensors").write_text("x")
    fold = {"strategy_class": "NNNC_TGT_f03",
            "model_dir": "user_data/strategies/saved_data/NNNC_ARO_f03"}
    assert run_folds.model_exists(run_folds.model_key(fold)) is True
    assert run_folds.model_key({"strategy_class": "C", "model_dir": None}) == "C"
