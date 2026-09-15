"""Calibration statistics and the admissibility routing (T022-T026, T031)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.stress import calibrate as CAL          # noqa: E402


def test_tolerances_are_the_pre_registered_ones():
    """These are inputs from research.md R3, fixed before the first calibration run.

    The temptation with a single load-bearing gate is to widen whichever band fails. When the
    gate DID fail during implementation (es 1.84, dd 1.76), the cause was a bug in the
    generator and the bands were left alone. This test exists so a later edit has to be
    deliberate.
    """
    assert CAL.KS_MAX == 0.10
    assert CAL.ES_RATIO == (0.7, 1.4)
    assert CAL.DD_RATIO == (0.7, 1.6)


def test_ks_threshold_admits_sampling_noise_but_not_a_different_shape():
    """0.10 is set from what the comparison can RESOLVE: at n=378 the 95% critical value
    against an identical distribution is 1.36/sqrt(378) = 0.0700."""
    import math
    crit = 1.36 / math.sqrt(378)
    assert crit < CAL.KS_MAX, "the tolerance must admit same-distribution sampling noise"
    assert CAL.KS_MAX < 2 * crit, "the tolerance must not admit a visibly different shape"


def test_tail_bands_are_asymmetric_in_the_safe_direction():
    """A synthetic tail FATTER than real is conservative; THINNER lets a fragile policy
    through. Both bands must therefore allow more fat than thin."""
    for lo, hi in (CAL.ES_RATIO, CAL.DD_RATIO):
        assert (hi - 1.0) > (1.0 - lo), f"band ({lo}, {hi}) is not asymmetric toward fat tails"


def test_expected_shortfall_is_the_mean_of_the_worst_tail():
    x = np.arange(100, dtype=float)          # 0..99
    assert CAL.expected_shortfall(x, 0.05) == pytest.approx(np.arange(5).mean())


def test_expected_shortfall_is_nan_and_empty_safe():
    assert np.isnan(CAL.expected_shortfall(np.array([])))
    assert CAL.expected_shortfall(np.array([1.0, np.nan, -3.0]), 0.5) == pytest.approx(-3.0)


def test_max_drawdown_on_a_known_path():
    # equity 1, 3, 0, 2  -> peak 3, trough 0 -> drawdown 3
    assert CAL.max_drawdown(np.array([1.0, 2.0, -3.0, 2.0])) == pytest.approx(3.0)


def test_max_drawdown_is_zero_for_a_monotone_gain():
    assert CAL.max_drawdown(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)


def test_require_calibration_blocks_without_a_passing_file(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(SystemExit, match="BLOCKED"):
        CAL.require_calibration(missing)

    failing = tmp_path / "cal.json"
    failing.write_text('{"passed": false, "bank": "b", "ks_distance": 0.4,'
                       ' "es_ratio": 2.0, "dd_ratio": 2.0}')
    with pytest.raises(SystemExit, match="KILL"):
        CAL.require_calibration(failing)

    ok = tmp_path / "ok.json"
    ok.write_text('{"passed": true, "bank": "b"}')
    assert CAL.require_calibration(ok)["passed"] is True


def test_stress_routes_through_the_shared_admissibility_gate():
    """T031/M1: the participation check must not be reimplemented locally.

    The shared module exists because this check kept being rewritten per site with each
    rewrite guarding only one side, and feature 013 had to fix its calibration for
    discrete-trade strategies after it marked PRODUCTION degenerate.
    """
    import inspect

    from research.labelfree.stress import run as RUN
    src = inspect.getsource(RUN)
    assert "AD.Arm" in src and "exposure_floor" in src, \
        "stress.run does not route through admissibility.Arm with a relative floor"
    assert RUN.EXPOSURE_FRACTION == 0.5
