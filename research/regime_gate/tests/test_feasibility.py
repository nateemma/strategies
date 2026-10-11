"""026 T004: feasibility gate logic (contract C2, research R2/R3).

The rule (threshold percentile x direction) is chosen from EARLY reversion trades only; the verdict
needs the "off" state to lose in BOTH halves, "on" to earn early, >= 50% of early trades kept "on",
and a late "off" that is not one long calendar block (M1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from research.regime_gate import feasibility as F


def _trades(vals, pnl, half):
    return pd.DataFrame({"value": vals, "pnl": pnl, "half": half})


def test_rule_choice_ignores_late_trades():
    rng = np.random.default_rng(0)
    early = _trades(rng.uniform(0, 1, 300), rng.normal(0.1, 1, 300), "early")
    late_a = _trades(rng.uniform(0, 1, 300), rng.normal(0, 1, 300), "late")
    late_b = late_a.assign(pnl=np.where(late_a.value > 0.8, -5.0, 1.0))
    r1 = F.choose_rule(pd.concat([early, late_a]))
    r2 = F.choose_rule(pd.concat([early, late_b]))
    assert r1 == r2


def test_late_only_separation_does_not_pass():
    rng = np.random.default_rng(1)
    early = _trades(rng.uniform(0, 1, 400), np.ones(400), "early")  # no early information
    late = _trades(rng.uniform(0, 1, 400), np.zeros(400), "late")
    late["pnl"] = np.where(late.value > 0.7, -5.0, 1.0)  # perfect late separation
    rec = F.evaluate(pd.concat([early, late]), off_bar_stats={"late_longest_run_share": 0.1})
    assert rec["verdict"] == "STOP"


def test_on_share_below_half_is_excluded():
    vals = np.linspace(0, 1, 100)
    pnl = np.where(vals < 0.6, -1.0, 1.0)  # best "off" would be the bottom 67% -> on-share 33%
    rule = F.choose_rule(_trades(vals, pnl, "early"))
    assert rule is not None and rule["on_share_early"] >= 0.5


def test_choice_minimises_early_off_pnl_and_ties_go_to_higher_on_share():
    vals = np.linspace(0, 1, 300)
    pnl = np.where(vals >= 0.67, -2.0, 0.5)  # losers live at the top
    rule = F.choose_rule(_trades(vals, pnl, "early"))
    assert rule["off_when"] == "ge" and rule["percentile"] == 67


def _passing_frame():
    vals = np.r_[np.linspace(0, 1, 200), np.linspace(0, 1, 200)]
    half = ["early"] * 200 + ["late"] * 200
    pnl = np.where(vals >= 0.67, -1.0, 0.8)
    return _trades(vals, pnl, half)


@pytest.mark.parametrize("breaker", ["off_early", "off_late", "on_early", "calendar"])
def test_each_pass_condition_is_required(breaker):
    df = _passing_frame()
    stats = {"late_longest_run_share": 0.2}
    assert F.evaluate(df, off_bar_stats=stats)["verdict"] == "PASS"
    if breaker == "off_early":
        df.loc[(df.half == "early") & (df.value >= 0.67), "pnl"] = 0.5
    elif breaker == "off_late":
        df.loc[(df.half == "late") & (df.value >= 0.67), "pnl"] = 0.5
    elif breaker == "on_early":
        df.loc[(df.half == "early") & (df.value < 0.67), "pnl"] = -0.1
    else:
        stats = {"late_longest_run_share": 0.85}
    assert F.evaluate(df, off_bar_stats=stats)["verdict"] == "STOP"


def test_nan_values_count_as_on():
    rule = {"threshold": 0.5, "off_when": "ge", "percentile": 50, "on_share_early": 1.0}
    assert F.off_mask(pd.Series([np.nan, 0.6, 0.4]), rule).tolist() == [False, True, False]


def test_off_bar_share_and_runs():
    idx = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0, np.nan, 1], index=idx, dtype=float)
    rule = {"threshold": 0.5, "off_when": "ge", "percentile": 50, "on_share_early": 1.0}
    st = F.off_stats(s, rule, idx[0], idx[-1] + pd.Timedelta(minutes=15))
    assert st["off_share"] == pytest.approx(0.6)
    assert st["runs"] == 3 and st["longest_run_share"] == pytest.approx(0.3)


def test_exit_codes(tmp_path, monkeypatch):
    assert (
        F.main(["--primary", "nope_arm", "--cross", "st_gbb", "--out", str(tmp_path / "f.md")]) == 2
    )
    monkeypatch.setattr(F, "run", lambda *a, **k: {"inst": {"verdict": "STOP"}})
    monkeypatch.setattr(F, "_check_inputs", lambda *a: None)
    assert F.main(["--primary", "x", "--cross", "y", "--out", str(tmp_path / "f.md")]) == 3
    monkeypatch.setattr(F, "run", lambda *a, **k: {"inst": {"verdict": "PASS"}})
    assert F.main(["--primary", "x", "--cross", "y", "--out", str(tmp_path / "f.md")]) == 0


def test_parse_gate_round_trips_and_rejects_bad_direction():
    inst, rule = F.parse_gate("volreg:lt:1.0332")
    assert inst == "volreg" and rule == {"off_when": "lt", "threshold": 1.0332}
    assert F.off_mask(pd.Series([1.0, 1.1, np.nan]), rule).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        F.parse_gate("volreg:gt:1.0")
