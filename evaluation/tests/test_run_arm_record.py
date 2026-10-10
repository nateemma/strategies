"""T013 — run_arm's new record fields (023 contract C3)."""
from __future__ import annotations

import json

import pandas as pd

from regime import run_arm


def test_exit_reason_counts():
    trades = pd.DataFrame({"exit_reason": ["take_profit", "trailing_stop_loss", "take_profit",
                                           "unclog_12h"]})
    assert run_arm.exit_reason_counts(trades) == {
        "take_profit": 2, "trailing_stop_loss": 1, "unclog_12h": 1}
    assert run_arm.exit_reason_counts(pd.DataFrame({"pair": ["A"]})) == {}


def test_aggregate_sink_sums_pairs(tmp_path):
    for pair, (s, v) in {"A_USDT": (3, 7), "B_USDT": (5, 9)}.items():
        (tmp_path / f"X_{pair}.json").write_text(json.dumps({"counts": {
            "signal": {"pass": s, "total": 10}, "volume": {"pass": v, "total": 10}}}))
    assert run_arm.aggregate_sink(tmp_path) == {
        "signal": {"pass": 8, "total": 20}, "volume": {"pass": 16, "total": 20}}
    assert run_arm.aggregate_sink(None) is None
    assert run_arm.aggregate_sink(tmp_path / "missing") is None


def test_build_env_sink_set_and_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", "/inherited")
    assert run_arm.build_env(tmp_path, tmp_path / "s")["FT_ENTRY_CONDITION_SINK"] == str(tmp_path / "s")
    assert "FT_ENTRY_CONDITION_SINK" not in run_arm.build_env(tmp_path, None)


def test_framework_fields_for_arm_d(tmp_path):
    # no scratch JSON here -> class defaults; the point is that it resolves and never raises
    out = run_arm.framework_fields("NNNC_A_Raw_Open", tmp_path,
                                   run_arm.STRAT_DIR / "config" / "config_wf.json")
    assert out.get("framework_sha"), out
    assert out["no_model"] is False
    assert out["trigger"] is None
