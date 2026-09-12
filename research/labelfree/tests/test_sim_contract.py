"""Simulator contract invariants (T011) + FR-014a assertion (T021a).

contracts/simulator.md: each invariant maps to a requirement; a violation is a defect.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from research.labelfree import sim                                      # noqa: E402
from research.labelfree.costs import CostModel                          # noqa: E402


def test_hard_cap_never_exceeds_fraction():
    """FR-005: executed delta <= 10% of the candle's quote volume."""
    assert sim.liquidity_cap(1000.0, 500.0) == 100.0
    assert sim.liquidity_cap(1000.0, 50.0) == 50.0


def test_soft_cap_is_differentiable_and_bounded():
    """FR-005: the training cap must not zero the gradient where it binds."""
    v = sim.soft_liquidity_cap(1000.0, np.array([50.0, 100.0, 500.0]))
    assert np.all(v <= 100.0 + 1e-6)
    # strictly increasing in the requested amount -> non-zero gradient past the cap
    assert v[2] > v[1] > v[0]


def test_max_concurrent_respects_cap():
    """FR-006: concurrent positions must not exceed max_open_trades."""
    t = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True)
    tr = pd.DataFrame({"open_date": [t[0], t[0], t[1]],
                       "close_date": [t[2], t[1], t[2]]})
    assert sim.max_concurrent(tr) == 2


def test_training_path_has_no_deployment_exit_stack():
    """FR-014a / T021a: training exits are band crossings only.

    The stack reasons exist solely so the *calibration replay* can take stack-priced
    exits as given; no stack exit may be produced by the simulator itself.
    """
    src = (pathlib.Path(sim.__file__)).read_text()
    for forbidden in ("def apply_roi", "def apply_stoploss", "def stoploss_grace",
                      "def trailing_stop"):
        assert forbidden not in src, f"simulator implements {forbidden} (FR-008a/FR-014a)"


def test_bar_returns_slices_time_not_batch():
    """Regression: a bare o[2:] on a pooled (n_pairs, T) panel slices the BATCH axis.

    The result is silently near-zero rather than an error, so pooled training would
    optimise against no returns at all. Caught 2026-09-10 by checking values, not shapes.
    """
    import mlx.core as mx
    from research.labelfree import sim as S
    o = mx.array(np.array([[1, 2, 4, 8, 16]] * 3, dtype=np.float32))
    r = S.bar_returns(o)
    mx.eval(r)
    assert r.shape == (3, 3), f"expected (3,3), got {r.shape} — batch axis was sliced"
    assert np.allclose(np.asarray(r), np.log(2)), "batched bar_returns are not log(2)"
    o1 = mx.array(np.array([1, 2, 4, 8, 16], dtype=np.float32))
    r1 = S.bar_returns(o1)
    mx.eval(r1)
    assert np.allclose(np.asarray(r1), np.log(2)), "1-D path regressed"


def test_turnover_penalty_is_finite_on_a_pooled_panel():
    """Regression: the penalty term must index TIME, not the pair axis.

    On a 10-pair panel a bare w[BURN_IN:] is an empty slice, its mean is NaN, and the loss
    goes NaN while training silently continues — 26 runs were wasted this way.
    """
    import mlx.core as mx
    from research.labelfree import losses as L
    from research.labelfree.train import BURN_IN
    w = mx.array(np.random.default_rng(0).random((10, BURN_IN + 200)).astype(np.float32))
    val = float(L.turnover_penalty(w[..., BURN_IN:], 0.001))
    assert np.isfinite(val), "turnover penalty is not finite on a pooled panel"
    assert float(L.turnover_penalty(w[BURN_IN:], 0.001)) != val or w.shape[0] > BURN_IN


def test_participation_guard_normalises_by_mask():
    """Regression: the floor must measure exposure over TRADEABLE bars only.

    f08's earliest training chunks are ~30% valid (only 3 of 10 pairs have 2021 history).
    An unnormalised mean(w) understates exposure by exactly the invalid fraction, so a
    policy holding a compliant 0.10 reads as 0.03 and gets penalised for data gaps it does
    not control. That distorted even the utility arm, which never had the degeneracy the
    guard exists to prevent.
    """
    import mlx.core as mx
    from research.labelfree import losses as L
    w = mx.concatenate([mx.full((300,), 0.10), mx.zeros((700,))])   # 30% valid, w=0.10
    m = mx.concatenate([mx.ones((300,)), mx.zeros((700,))])
    assert float(L.participation_penalty(w, 0.05, 100.0, mask=m)) == 0.0, \
        "compliant policy penalised because masked bars were counted"
    assert float(L.participation_penalty(w, 0.05, 100.0)) > 0.0, \
        "unmasked path should still show the defect this guards against"
    flat = mx.zeros((1000,))
    assert float(L.participation_penalty(flat, 0.05, 100.0, mask=m)) > 0.0, \
        "a genuinely flat policy must still be caught"


def test_standardisation_gradient_is_finite_at_zero_variance():
    """Regression: EPS must go INSIDE sqrt, not after it.

    `sqrt(var) + EPS` is finite in the forward pass even at var == 0, but d/dx sqrt(x) is
    infinite at x = 0, so the backward pass returns NaN. A fully-masked pair has w
    identically 0 and var exactly 0, so this fires whenever a chunk holds a pair with no
    data — it killed all five Arm C seeds. A forward-only check cannot catch it.
    """
    import mlx.core as mx
    from research.labelfree import losses as L
    zero = mx.zeros((100,))
    g = mx.grad(lambda w: L.aux_vol_loss(w, zero))(zero)
    mx.eval(g)
    assert np.isfinite(np.asarray(g)).all(), "aux loss gradient is NaN at zero variance"


def test_cost_model_spread_is_estimated_not_constant():
    """FR-004a: half-spread varies per bar and never collapses to zero."""
    n = 400
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    df = pd.DataFrame({"open": px, "close": px,
                       "high": px * (1 + abs(rng.normal(0, .004, n))),
                       "low": px * (1 - abs(rng.normal(0, .004, n))),
                       "volume": rng.uniform(1e3, 1e5, n)})
    hs = CostModel().half_spread(df)
    assert hs.min() > 0, "half-spread collapsed to zero"
    assert hs.nunique() > 1, "half-spread is a flat constant (forbidden by FR-004a)"
