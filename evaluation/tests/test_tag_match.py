"""T005 — arm records resolve by EXACT name (023 research R6).

Both lookups used to glob `{tag}_*.json`, so tag `wf_x_f00` also matched the seed-7 record
`wf_x_f00_s7_F00.json`, and which one came back depended on directory order.
"""
from __future__ import annotations

import json

from evaluation import pool, run_folds


def _write(d, name, marker):
    (d / name).write_text(json.dumps({"marker": marker}))


def test_pool_arm_record_is_exact(tmp_path, monkeypatch):
    _write(tmp_path, "wf_x_f00_s7_F00.json", "seed7")
    _write(tmp_path, "wf_x_f00_F00.json", "seed1")
    monkeypatch.setattr(pool, "ARMS", tmp_path)
    assert pool.arm_record("wf_x_f00", "F00")["marker"] == "seed1"
    assert pool.arm_record("wf_x_f00_s7", "F00")["marker"] == "seed7"
    assert pool.arm_record("wf_x_f01", "F01") is None


def test_run_folds_arm_record_is_exact(tmp_path, monkeypatch):
    arms = tmp_path / "regime" / "arms"
    arms.mkdir(parents=True)
    _write(arms, "wf_x_f00_s7_F00.json", "seed7")
    _write(arms, "wf_x_f00_F00.json", "seed1")
    monkeypatch.setattr(run_folds, "STRAT_DIR", tmp_path)
    fold = {"arm_tag": "wf_x_f00", "fold_id": "f00"}
    assert run_folds.arm_record(fold)["marker"] == "seed1"
