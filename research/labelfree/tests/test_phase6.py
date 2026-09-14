"""Regression tests for the Phase 6 modules (T027, T032a, T032b, T036, T037, T033b).

Each test pins a defect that was actually found while building these, not a hypothetical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import stats as ST                       # noqa: E402
from research.labelfree.arms import screen_degenerate            # noqa: E402
from research.labelfree.baselines import (_rolling_std,          # noqa: E402
                                          ema_trend_exposure)


# --- T036 activity ---------------------------------------------------------

def test_always_on_is_one_trade():
    a = ST.activity(np.ones((1, ST.BARS_PER_YEAR)))
    assert a["trades_per_year"] == 1.0
    assert a["mean_hold_hours"] == pytest.approx(8760.0)


def test_flat_path_has_no_trades_and_undefined_hold():
    a = ST.activity(np.zeros((1, 1000)))
    assert a["trades_per_year"] == 0.0
    assert np.isnan(a["mean_hold_hours"])


def test_trade_is_a_contiguous_run_not_a_weight_change():
    """Exposure ramping 0.2->0.9 inside one position is ONE trade, not seven."""
    w = np.concatenate([np.zeros(50), np.linspace(0.2, 0.9, 100), np.zeros(50)])[None, :]
    assert ST.activity(w)["trades_per_year"] * (w.shape[1] / ST.BARS_PER_YEAR) == 1.0


# --- T032a baselines -------------------------------------------------------

def test_ema_rule_actually_trades():
    """Found 2026-09-12: band +/-0.1 on the RAW ratio never fires on 15m bars, so the
    baseline sat 100% flat on both folds. A baseline that never trades is not a baseline."""
    t = np.linspace(0, 1, 5000)
    price = (100 * np.exp(0.5 * t + 0.02 * np.sin(40 * t)))[None, :]
    w = ema_trend_exposure(price)
    assert 0.0 < w.mean() < 1.0


def test_ema_dead_zone_limits_churn():
    t = np.linspace(0, 1, 5000)
    price = (100 * np.exp(0.5 * t + 0.02 * np.sin(40 * t)))[None, :]
    w = ema_trend_exposure(price)
    assert np.abs(np.diff(w, axis=1)).sum() < 0.2 * w.shape[1]


def test_rolling_std_is_causal():
    """A spike at the end must not change any earlier value, or the baseline peeks."""
    x = np.zeros((1, 500))
    base = _rolling_std(x, 60).copy()
    x[0, -1] = 99.0
    assert np.allclose(_rolling_std(x, 60)[0, :-1], base[0, :-1])


# --- T033b annualisation ---------------------------------------------------

def test_annualised_pp_identity_over_one_year():
    assert ST.annualised_pp(np.log(1.10), 365) == pytest.approx(10.0, abs=1e-6)


def test_annualisation_exponentiates_once():
    """Scaling a percentage instead of the log is the error that once turned a working
    instrument into a reported 0.62x regression."""
    naive = (np.exp(np.log(1.10)) - 1) * 100 * (365 / 90)
    correct = ST.annualised_pp(np.log(1.10), 90)
    # 47.19 vs 40.56 -- the gap is larger than the floor the study resolves at, so the
    # two methods disagree about whether an effect is real.
    assert abs(correct - naive) > ST.FLOOR_PP_PER_YEAR


# --- T027 degeneracy screen ------------------------------------------------

def _run(lr, w):
    return {"status": "ok", "test_log_return": lr, "test_mean_exposure": w}


def test_degenerate_arm_is_excluded_and_its_return_withheld():
    """An arm that does nothing posts a superb return by avoiding the control's losses."""
    s = screen_degenerate({"lazy": [_run(9.9, 0.001)] * 5,
                           "real": [_run(-0.1, 0.40)] * 5})
    assert "lazy" in s["inconclusive"] and "lazy" not in s["eligible"]
    assert s["inconclusive"]["lazy"]["return"] == "WITHHELD (SC-015)"
    assert "real" in s["eligible"]


def test_one_flat_seed_does_not_void_an_arm_that_trades():
    runs = [_run(-0.1, 0.40)] * 4 + [_run(-0.1, 0.0)]
    assert "arm" in screen_degenerate({"arm": runs})["eligible"]


# --- T037 deflation interpretability --------------------------------------

def test_deflation_refuses_to_pass_on_seed_axis_alone():
    """DSR/CSCV need a time axis; 5 seeds cannot satisfy SC-004 however good they look."""
    out = ST.trial_log_deflation()
    assert out["interpretable"] is False
    assert out["sc004_verdict"].startswith("CANNOT BE EVALUATED")


# --- baselines span mode (written and cancelled twice before landing) -------

def test_span_parse():
    from research.labelfree.baselines import parse_span
    assert parse_span("f05..f08") == ["f05", "f06", "f07", "f08"]
    assert parse_span("f05,f08") == ["f05", "f08"]
    assert parse_span("f07") == ["f07"]


def test_one_fold_span_reproduces_single_fold_exactly():
    """The drift guard. A second code path for a number this module already owns is exactly
    the failure that made brief 02 quote a hand-computed buy-and-hold, flagged in its own
    report as 'not the audited baselines.py output'."""
    import json
    p = ROOT / "research/labelfree/baselines_f05_f14.json"
    if not p.exists():
        pytest.skip("span baselines not run")
    from research.labelfree.baselines import run, run_span
    single = run("f07")
    span = run_span(["f07"])
    for name in ("do_nothing", "buy_and_hold", "ema_trend_20_60"):
        assert span["aggregate"][name]["log_return_total"] == pytest.approx(
            single["baselines"][name]["log_return"], abs=1e-12)


def test_span_annualisation_matches_the_floor_convention():
    """A span figure must be comparable with a recomputed floor: sum the log returns across
    the span, annualise once by the span's length. Scaling a percentage instead is the error
    that once turned a working instrument into a reported 0.62x regression."""
    import json
    p = ROOT / "research/labelfree/baselines_f05_f14.json"
    if not p.exists():
        pytest.skip("span baselines not run")
    r = json.loads(p.read_text())
    bh = r["aggregate"]["buy_and_hold"]
    assert bh["log_return_per_year"] == pytest.approx(
        bh["log_return_total"] / r["years"], rel=1e-9)
    assert bh["pp_per_year"] == pytest.approx(
        (np.exp(bh["log_return_per_year"]) - 1) * 100, rel=1e-9)


def test_only_the_production_control_is_profitable_over_the_eval_span():
    """Measured 2026-09-13 over f05..f14 (2.46 yr): the production control is the ONLY
    baseline that makes money. Buy-and-hold loses 14.6 pp/yr and the TSMOM rule loses 77.4.
    'Beat all four baselines' therefore means beating +7.86 pp/yr, not beating zero."""
    import json
    p = ROOT / "research/labelfree/baselines_f05_f14.json"
    if not p.exists():
        pytest.skip("span baselines not run")
    agg = json.loads(p.read_text())["aggregate"]
    assert agg["production_control"]["pp_per_year"] > 0
    assert agg["buy_and_hold"]["pp_per_year"] < 0
    assert agg["ema_trend_20_60"]["pp_per_year"] < agg["buy_and_hold"]["pp_per_year"]


# --- SC-004: the trial-log schema that four features reported unevaluable ---

def test_block_returns_compress_a_series_preserving_total():
    """Blocks must sum to the same total: they are a compression of the series, not a
    resampling of it."""
    from research.labelfree.trials import block_returns, N_BLOCKS
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.01, 5000)
    b = block_returns(s)
    assert len(b) == N_BLOCKS
    assert sum(b) == pytest.approx(s.sum(), rel=1e-9)


def test_block_returns_handles_short_and_empty_series():
    from research.labelfree.trials import block_returns
    assert block_returns([]) == []
    assert len(block_returns([0.1, 0.2, 0.3])) == 3          # fewer bars than blocks
    assert block_returns([np.nan, np.inf]) == []             # nothing finite


def test_time_axis_path_makes_deflation_interpretable(tmp_path):
    """The fix for SC-004. With block returns present the statistics are computed over a
    TIME axis -- what DSR and CSCV are actually defined over -- instead of over 5 seeds,
    where a reliably mediocre configuration scores a high 'Sharpe' purely because its
    across-seed dispersion is tiny."""
    import json
    from research.labelfree import stats as ST
    from research.labelfree.trials import block_returns

    rng = np.random.default_rng(1)
    recs = []
    for cfg in ("armA", "armB", "armC"):
        for seed in (1, 2, 3):
            tid = f"{cfg}-s{seed}@run1"
            recs.append({"trial_id": tid, "written_at": "t", "result": None})
            recs.append({"trial_id": tid, "completed_at": "t",
                         "result": {"test_log_return": float(rng.normal(-0.1, 0.05))},
                         "block_returns": block_returns(rng.normal(0, 0.01, 2000))})
    out = ST.trial_log_deflation(recs)
    assert out["n_with_block_returns"] == 9
    assert out["time_axis_available"] is True
    assert out["time_axis_blocks"] == 16
    assert "dsr_time_axis" in out and "pbo_time_axis" in out
    assert np.isfinite(out["pbo_time_axis"])


def test_trials_without_block_returns_still_count_toward_deflation():
    """A trial that cannot contribute a time axis must still deflate the statistic: the
    trial COUNT is what the deflation corrects for."""
    from research.labelfree import stats as ST
    from research.labelfree.trials import block_returns
    rng = np.random.default_rng(2)
    recs = []
    for i in range(40):                                   # 40 pre-registered, no series
        recs.append({"trial_id": f"noisy{i}-s1@r", "written_at": "t", "result": None})
    for cfg in ("a", "b"):
        for seed in (1, 2):
            tid = f"{cfg}-s{seed}@r"
            recs.append({"trial_id": tid, "written_at": "t", "result": None})
            recs.append({"trial_id": tid, "completed_at": "t",
                         "result": {"test_log_return": -0.1},
                         "block_returns": block_returns(rng.normal(0, 0.01, 800))})
    out = ST.trial_log_deflation(recs)
    assert out["n_pre_registered"] == 44
    assert out["n_with_block_returns"] == 4


def test_pooled_comparator_refuses_a_losing_arm():
    """The retro-fit to evaluation/compare_pooled.py.

    Measured 2026-09-13: an arm losing 3.78pp against a control losing 18.90pp produced
    '+4.106 pp/yr IMPROVED' under the old code -- ABOVE the 3.35 pp/yr floor, i.e. it would
    have cleared SC-001. That is brief 02's failure mode reaching an acceptance gate."""
    import copy
    import json
    import os
    import subprocess
    import tempfile

    src = ROOT / "evaluation/pooled_nnnc.json"
    if not src.exists():
        pytest.skip("pooled control record unavailable")
    base = json.loads(src.read_text())
    arm, ctl = copy.deepcopy(base), copy.deepcopy(base)
    for i, t in enumerate(arm["trades"]):
        t["profit_abs"] = -1.0
        t["open_date"] = f"2024-01-01 00:{i % 60:02d}:00+00:00"
    for t in ctl["trades"]:
        t["profit_abs"] = -5.0

    d = Path(tempfile.mkdtemp())
    (d / "arm.json").write_text(json.dumps(arm))
    (d / "ctl.json").write_text(json.dumps(ctl))
    r = subprocess.run([os.sys.executable, str(ROOT / "evaluation/compare_pooled.py"),
                        "--arm", str(d / "arm.json"), "--control", str(d / "ctl.json")],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert "INCONCLUSIVE_ARM_LOSES" in r.stdout
    assert "IMPROVED" not in r.stdout


def test_pooled_comparator_self_comparison_is_unchanged():
    """The retro-fit must not disturb the existing guards or the pinned statistic."""
    import os
    import subprocess
    src = ROOT / "evaluation/pooled_nnnc.json"
    if not src.exists():
        pytest.skip("pooled control record unavailable")
    r = subprocess.run([os.sys.executable, str(ROOT / "evaluation/compare_pooled.py"),
                        "--arm", str(src), "--control", str(src)],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert "INCONCLUSIVE_ABSORBED" in r.stdout
