"""Signal-context timing (feature 017) — regression tests for the validated invariants.

Two real defects were found and fixed during implementation, and both are pinned here because
each looked like a finding until it was diagnosed:

1. The fidelity matcher compared recorded trades against signal RUN STARTS, so a trade opening
   MID-RUN read as a miss. Four folds fell below the 0.95 floor for a reason that had nothing to
   do with the reconstruction.
2. `reconcile()` compared a GROSS close-rate return against a NET `profit_ratio`, so all 378
   trades missed by 0.00202 +/- 0.0003 — the round-trip fee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.signalctx import FIDELITY_MIN, NOISE_FLOOR_PP, RECONCILE_TOL
from research.signalctx import confound as C
from research.signalctx import paths as P
from research.signalctx import signals as S


def test_runs_collapse_to_events():
    """A signal is an EVENT, not a bar (f08: 139 bars are 120 events)."""
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    assert len(S._events(mask)) == 3
    assert list(S.run_lengths(mask)) == [3, 1, 2]


def test_single_bar_run_is_one_event():
    assert len(S._events(np.array([1], dtype=bool))) == 1
    assert list(S.run_lengths(np.array([1], dtype=bool))) == [1]


def test_reconcile_is_net_of_fees():
    """Regression: a gross-vs-net comparison missed every trade by the round-trip fee."""
    t = {"open_rate": 100.0, "close_rate": 110.0, "fee_open": 0.001, "fee_close": 0.001,
         "profit_ratio": (110.0 * 0.999) / (100.0 * 1.001) - 1.0}
    assert P.reconcile(t) < 1e-12
    gross = 110.0 / 100.0 - 1.0
    assert abs(gross - t["profit_ratio"]) > RECONCILE_TOL, (
        "the fixture no longer distinguishes gross from net, so it cannot catch a regression")


def test_verdict_refuses_the_wrong_currency():
    """FR-007 — the 7.633 pp floor is an annualised RETURN quantity.

    The measured win-rate effect is -15.82 points, which clears 7.633 numerically while carrying
    no return information. Gating on it would have passed a dead feature.
    """
    res = {"cmh_p": 0.001, "effect_return_pp_yr": 20.0, "effect_winrate_pp": -15.82}
    assert C.verdict(res) == "survives"
    with pytest.raises(ValueError, match="different currency|must not be applied"):
        C.verdict(res, key="effect_winrate_pp")


def test_verdict_refuses_an_unresolvable_effect():
    """Significance is not size: inside the noise floor is 'not resolved', whatever the p."""
    res = {"cmh_p": 1e-9, "effect_return_pp_yr": NOISE_FLOOR_PP - 0.1}
    assert C.verdict(res) == "proxy_for_drawdown"


def test_verdict_requires_significance():
    res = {"cmh_p": 0.20, "effect_return_pp_yr": 99.0}
    assert C.verdict(res) == "proxy_for_drawdown"


def test_effect_sizes_report_both_currencies():
    m = pd.DataFrame({
        "group": ["repeat"] * 4 + ["control"] * 4,
        "win": [True, False, False, False, True, True, True, False],
        "profit_ratio": [0.01, -0.02, -0.03, -0.01, 0.02, 0.03, 0.01, -0.01],
    })
    es = C.effect_sizes(m, span_years=1.0)
    assert es["effect_winrate_pp"] < 0
    assert "effect_return_pp_yr" in es
    assert es["effect_winrate_pp"] != es["effect_return_pp_yr"], "the two must not collapse"


def test_fidelity_floor_is_pinned():
    assert FIDELITY_MIN == 0.95
