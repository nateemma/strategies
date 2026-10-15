"""034 T009: pick dispatch, 4-dp storage, the control's separate output, one fold per process."""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.label_screen import pick as P


def _plans(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "FOLDS", tmp_path)
    monkeypatch.setattr(P, "CONTROL_SHARE", tmp_path / "thresholds_control_share.json")
    for name in ("ls_fmae_sz", "ls_fmae_es", "ls_control_sz"):
        (tmp_path / f"plan_{name}.json").write_text(json.dumps({"folds": []}))


def test_store_rounds_to_four_decimals_into_the_role_plan(tmp_path, monkeypatch):
    _plans(tmp_path, monkeypatch)
    P.store("FMAE", "sz", "f00", 0.123456789, {"achieved_share": 0.5})
    P.store("FMAE", "es", "f00", 0.98765432, {"target_rate": P.ENTRY_RATE})
    sz = json.loads((tmp_path / "plan_ls_fmae_sz.json").read_text())
    es = json.loads((tmp_path / "plan_ls_fmae_es.json").read_text())
    assert sz["thresholds"] == {"f00": 0.1235} and es["thresholds"] == {"f00": 0.9877}
    assert sz["threshold_picks"]["f00"]["achieved_share"] == 0.5


def test_control_goes_to_its_own_file_never_a_plan(tmp_path, monkeypatch):
    _plans(tmp_path, monkeypatch)
    before = (tmp_path / "plan_ls_control_sz.json").read_text()
    P.store("CONTROL", "sz", "f03", 0.71234, {})
    assert json.loads(P.CONTROL_SHARE.read_text())["f03"]["threshold"] == 0.7123
    assert (tmp_path / "plan_ls_control_sz.json").read_text() == before


def test_roles():
    P.check_role("FMAE", "es")
    P.check_role("BOG", "sz")
    with pytest.raises(SystemExit):
        P.check_role("BOG", "es")
    with pytest.raises(SystemExit):
        P.check_role("CONTROL", "es")


def test_one_fold_per_process():
    with pytest.raises(SystemExit):
        P.main(["--tag", "FMAE", "--fold", "f00,f01", "--role", "sz"])


def test_sz_score_is_p_buy_without_the_argmax():
    probs = np.array([[0.5, 0.2, 0.3], [0.1, 0.2, 0.7], [np.nan, np.nan, np.nan]])
    assert P.p_buy(probs).tolist() == [0.3, 0.7, 0.0]
