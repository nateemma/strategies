# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
"""Tests for the per-fold `threshold` override added to `run_folds.generate_fold`.

All tests monkeypatch `run_folds.STRAT_DIR` to a tmp_path -- `generate_fold` writes directly
into `<STRAT_DIR>/<family>/`, and the real value points at the live NNNC/ directory this study
must never touch outside of the harness's own, deliberate runs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_folds


FOLD = {
    "fold_id": "f00",
    "train_start": "2021-01-01",
    "train_end": "2022-12-31",
    "test_start": "2023-01-01",
    "test_end": "2023-04-01",
    "train_timerange": "20210101-20230101",
    "test_timerange": "20230101-20230401",
    "test_days": 90,
    "strategy_class": "TESTCLS_f00",
    "model_dir": "saved_data/TESTCLS_f00",
    "arm_tag": "wf_test_f00",
    "status": "planned",
}

PARAMS = {
    "strategy_name": "placeholder",
    "params": {"buy": {"prediction_threshold": 0.8, "entry_rvol_threshold": -1.0}},
}


def _isolated_strat_dir(tmp_path, monkeypatch, family="FAM"):
    monkeypatch.setattr(run_folds, "STRAT_DIR", tmp_path)
    (tmp_path / family).mkdir()
    return family


def test_no_thresholds_supplied_is_byte_identical_to_current_behaviour(tmp_path, monkeypatch):
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    _, js_a = run_folds.generate_fold(FOLD, "Base", family, PARAMS)
    blob_a = js_a.read_bytes()
    _, js_b = run_folds.generate_fold(FOLD, "Base", family, PARAMS)  # no threshold= at all
    blob_b = js_b.read_bytes()
    assert blob_a == blob_b
    assert json.loads(blob_a)["params"]["buy"]["prediction_threshold"] == 0.8


def test_explicit_threshold_none_matches_the_no_arg_call(tmp_path, monkeypatch):
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    _, js_default = run_folds.generate_fold(FOLD, "Base", family, PARAMS)
    _, js_explicit_none = run_folds.generate_fold(FOLD, "Base", family, PARAMS, threshold=None)
    assert js_default.read_bytes() == js_explicit_none.read_bytes()


def test_threshold_override_reaches_the_generated_json(tmp_path, monkeypatch):
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    _, js = run_folds.generate_fold(FOLD, "Base", family, PARAMS, threshold=0.42)
    written = json.loads(js.read_text())
    assert written["params"]["buy"]["prediction_threshold"] == 0.42
    assert written["strategy_name"] == FOLD["strategy_class"]       # unchanged invariant
    # every other buy-space key from the snapshot survives untouched
    assert written["params"]["buy"]["entry_rvol_threshold"] == -1.0


def test_threshold_override_does_not_mutate_the_params_snapshot_in_memory(tmp_path, monkeypatch):
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    run_folds.generate_fold(FOLD, "Base", family, PARAMS, threshold=0.42)
    assert PARAMS["params"]["buy"]["prediction_threshold"] == 0.8


def test_threshold_override_with_no_params_file_is_a_noop(tmp_path, monkeypatch):
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    py, js = run_folds.generate_fold(FOLD, "Base", family, None, threshold=0.5)
    assert js is None
    assert py.exists()


# =============================================================================================
# Feature 022 T004/T006 — the `guards` plan key must be INERT when absent
#
# The fixture in evaluation/fixtures/run_folds_no_guards_key.json was generated from the
# PRE-change run_folds.py and committed before the key was added. That ordering is the whole
# point: a fixture generated afterwards records whatever the new code does and then asserts the
# new code does it, which proves nothing. `run_folds.py` is shared harness code that other
# features' plans read, so a regression here is not contained to feature 022.
# =============================================================================================

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_folds_no_guards_key.json"


def test_generate_fold_without_guards_matches_the_PRE_CHANGE_fixture(tmp_path, monkeypatch):
    """Byte-identical to the output recorded before the `guards` key existed."""
    import hashlib
    fix = json.loads(FIXTURE.read_text())
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    py, js = run_folds.generate_fold(FOLD, "Base", family, PARAMS)
    assert hashlib.sha256(js.read_bytes()).hexdigest() == fix["json_sha256"], (
        "the generated params file changed for a plan with NO guards key -- fix run_folds.py, "
        "do NOT regenerate the fixture"
    )
    assert hashlib.sha256(py.read_bytes()).hexdigest() == fix["py_sha256"]
    assert js.read_text() == fix["json_text"]


def test_explicit_guards_none_matches_the_no_arg_call(tmp_path, monkeypatch):
    """Mirrors the `threshold=None` precedent above: passing the new arg as None is a no-op."""
    family = _isolated_strat_dir(tmp_path, monkeypatch)
    _, js_default = run_folds.generate_fold(FOLD, "Base", family, PARAMS)
    _, js_none = run_folds.generate_fold(FOLD, "Base", family, PARAMS, guards=None)
    assert js_default.read_bytes() == js_none.read_bytes()
