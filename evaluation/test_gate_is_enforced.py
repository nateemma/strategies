"""Feature 022 T009/T010 — FR-006's trackability gate must be ENFORCED, not merely written down.

WHY THIS FILE EXISTS. `/speckit-analyze` found that the feature's headline gate was a prose
note in tasks.md: Constitution Principle VII is non-negotiable and FR-006 says arm C must not
run on a failed trackability verdict, but nothing prevented an operator or an agent from
building the plan and running fifteen folds anyway. The same feature already applies the
"a guard that cannot fail is worthless" standard to its own inertness negative control, so it
applies here.

The gate binds ONLY on a plan that carries a `guards` key. Every pre-existing plan has no such
key and is therefore untouched -- pinned separately by
`test_generate_fold_without_guards_matches_the_PRE_CHANGE_fixture`.
"""
import json
from pathlib import Path

import pytest

from evaluation import run_folds

PLAN_NO_GUARDS = {"run_id": "nog", "folds": []}
PLAN_WITH_GUARDS = {
    "run_id": "gbbgrid_test",
    "guards": {"f00": {"entry_atr_pct": 0.013, "entry_adx_threshold": 10.0,
                       "entry_rvol_threshold": 1.9}},
    "folds": [],
}


def _verdict_file(tmp_path, monkeypatch, payload: dict | None):
    """Point run_folds at an isolated folds dir, optionally writing a verdict there."""
    monkeypatch.setattr(run_folds, "FOLDS_DIR", tmp_path)
    if payload is not None:
        (tmp_path / "trackability_gbbgrid_test.json").write_text(json.dumps(payload))
    return tmp_path


def test_a_plan_without_a_guards_key_is_never_gated(tmp_path, monkeypatch):
    """Every existing plan must be unaffected, verdict file or not."""
    _verdict_file(tmp_path, monkeypatch, None)
    run_folds._require_trackable_verdict(PLAN_NO_GUARDS)   # must not raise


def test_a_guards_plan_with_NO_verdict_file_is_REFUSED(tmp_path, monkeypatch):
    _verdict_file(tmp_path, monkeypatch, None)
    with pytest.raises(SystemExit) as e:
        run_folds._require_trackable_verdict(PLAN_WITH_GUARDS)
    assert e.value.code != 0


def test_a_guards_plan_with_a_NOT_TRACKABLE_verdict_is_REFUSED(tmp_path, monkeypatch):
    """The expected branch of feature 022, and the one the gate exists to stop."""
    _verdict_file(tmp_path, monkeypatch, {"verdict": "not_trackable",
                                          "mean_regret_delta": +0.004})
    with pytest.raises(SystemExit) as e:
        run_folds._require_trackable_verdict(PLAN_WITH_GUARDS)
    assert e.value.code != 0


def test_the_refusal_QUOTES_the_verdict_it_found(tmp_path, monkeypatch, capsys):
    """A refusal that does not say why sends the reader back to the code."""
    _verdict_file(tmp_path, monkeypatch, {"verdict": "not_trackable"})
    with pytest.raises(SystemExit):
        run_folds._require_trackable_verdict(PLAN_WITH_GUARDS)
    assert "not_trackable" in capsys.readouterr().out


def test_a_guards_plan_with_a_TRACKABLE_verdict_is_ALLOWED(tmp_path, monkeypatch):
    _verdict_file(tmp_path, monkeypatch, {"verdict": "trackable",
                                          "mean_regret_delta": -0.004})
    run_folds._require_trackable_verdict(PLAN_WITH_GUARDS)   # must not raise


def test_an_unparseable_verdict_file_is_REFUSED_not_ignored(tmp_path, monkeypatch):
    """Failing open would make the gate worthless on a truncated write."""
    monkeypatch.setattr(run_folds, "FOLDS_DIR", tmp_path)
    (tmp_path / "trackability_gbbgrid_test.json").write_text("{not json")
    with pytest.raises(SystemExit) as e:
        run_folds._require_trackable_verdict(PLAN_WITH_GUARDS)
    assert e.value.code != 0
