"""036 T005: the live-fold tool's plan, refusals and key identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[3]
for p in (STRAT, STRAT / "NNNC", STRAT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.live_pnl import live_fold as LF  # noqa: E402


CONTROL = json.loads((STRAT / "evaluation" / "folds" / "plan_wf_aro_g24.json").read_text())


def test_plan_is_one_fold_on_the_controls_snapshot():
    p = LF.build_plan("20261016", "live1", CONTROL)
    assert len(p["folds"]) == 1
    f = p["folds"][0]
    assert f["train_timerange"] == "20210101-20261016"
    assert f["test_timerange"] == "20261016-20270114"
    assert f["strategy_class"] == "NNNC_LIVE_live1" and f["fold_id"] == "live1"
    assert p["params_snapshot"] == CONTROL["params_snapshot"]
    assert p["params_sha"] == CONTROL["params_sha"]
    assert p["base_strategy"] == "NNNC_A_Raw_Open" and "thresholds" not in p


def test_refusals():
    last = {"A/USDT": pd.Timestamp("2026-10-15 23:45", tz="UTC"),
            "B/USDT": pd.Timestamp("2026-10-12 20:00", tz="UTC")}
    assert any("B/USDT" in r for r in LF.refusals("20261016", "live1", last, [], []))
    ok = {"A/USDT": last["A/USDT"], "B/USDT": last["A/USDT"]}
    assert LF.refusals("20261016", "live1", ok, [], []) == []
    assert LF.refusals("20261016", "live1", ok, [{"run_id": "live1"}], [])
    assert LF.refusals("20261016", "live1", ok, [], ["freqtrade backtesting -s X"])


def test_rec_class_has_the_fold_key():
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK
    from utils.ExecutionRecorder import ExecutionRecorderMixin

    cfg = load_config(STRAT / "config" / "config_wf.json")
    base = load_strategy_class("NNNC_A_Raw_Open")
    fold = type("NNNC_LIVE_t", (base,), {"MODEL_TRAIN_TIMERANGE": "20210101-20261016",
                                         "MODEL_SEED": 1, "__module__": base.__module__})
    rec = type("NNNC_LIVE_t_REC", (ExecutionRecorderMixin, fold),
               {"RECORD_EXECUTION": True, "RECORD_DIR": "/tmp/x", "RECORD_RUN_ID": "t",
                "__module__": base.__module__})
    assert MK.key_hash(MK.resolve(rec, cfg)) == MK.key_hash(MK.resolve(fold, cfg))


def test_rec_source_imports_and_switches_on():
    src = LF._rec_source("live9")
    assert "class NNNC_LIVE_live9_REC(ExecutionRecorderMixin, NNNC_LIVE_live9):" in src
    assert "RECORD_EXECUTION = True" in src and "RECORD_RUN_ID = 'live9'" in src
    assert '"data" / "live_nnnc"' in src
