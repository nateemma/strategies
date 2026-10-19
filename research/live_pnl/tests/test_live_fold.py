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
    p = LF.build_plan("20261016", "live1", CONTROL, book="aro")
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


def test_default_book_is_cs_aro_on_its_generated_book_class():
    p = LF.build_plan("20261016", "live2", CONTROL)
    assert LF.DEFAULT_BOOK == "cs_aro" and p["book"] == "cs_aro" and p["run_id"] == "live_cs_aro"
    assert p["base_strategy"] == "NNNC_LIVE_live2_BOOK"
    assert p["params_snapshot"] == CONTROL["params_snapshot"] and "thresholds" not in p
    assert LF.plan_name("cs_aro") == "plan_live_cs_aro.json"


def test_live_width_cut_is_the_latest_folds_matched_cut():
    cuts = json.loads(LF.WIDTH_TWIN.read_text())["models"]["ARO"]["cuts"]
    fid, cut = LF.live_width_cut()
    assert fid == "f14" and cut == cuts["f14"]


def test_book_class_keeps_the_aro_model_key(tmp_path):
    """The cs_aro live fold reuses the aro fold's weights for the same training window."""
    import importlib.util

    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    train = "20210101-20261016"
    src = tmp_path / "NNNC_LIVE_t_BOOK.py"
    src.write_text(LF._book_source("t", train, 0.0944, "f14"))
    spec = importlib.util.spec_from_file_location("NNNC_LIVE_t_BOOK", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    book = mod.NNNC_LIVE_t_BOOK
    assert book.WIDTH_CUTS == {train: 0.0944} and book.HELD_PAIR_ADD is None
    cfg = load_config(STRAT / "config" / "config_wf.json")
    aro = load_strategy_class("NNNC_A_Raw_Open")
    pin = {"MODEL_TRAIN_TIMERANGE": train, "MODEL_SEED": 1}
    a = type("A", (aro,), {**pin, "__module__": aro.__module__})
    b = type("B", (book,), {**pin, "__module__": book.__module__})
    assert MK.key_hash(MK.resolve(b, cfg)) == MK.key_hash(MK.resolve(a, cfg))
