"""Training objectives (T023).

FR-014: both are implemented and run as separate logged trial families.

  utility        -(mean(r_net) - lambda/2 * var(r_net))   (Kubo & Nakagawa 2025)
                 unbiased under minibatching, which naive Sharpe is not
  robust_sharpe  -(pooled Sharpe + SoftMin worst-window Sharpe)   (DeePM)

The literature's central finding is that the NAIVE direct-P&L objective overfits
(val 1.61 -> test 0.48 Sharpe) and only the structured version holds, which is why both
run rather than one (research.md R5).
"""
from __future__ import annotations

import mlx.core as mx

EPS = 1e-8


def utility(r_net: mx.array, lam: float = 2.0) -> mx.array:
    """Mean-variance utility. Minibatch-unbiased: both terms are plain moments."""
    m = mx.mean(r_net)
    v = mx.var(r_net)
    return -(m - 0.5 * lam * v)


def sharpe(r_net: mx.array) -> mx.array:
    return mx.mean(r_net) / (mx.sqrt(mx.var(r_net)) + EPS)


def soft_min(x: mx.array, tau: float = 0.1) -> mx.array:
    """Differentiable minimum: -tau * logsumexp(-x/tau)."""
    return -tau * mx.logsumexp(-x / tau)


def robust_sharpe(r_net: mx.array, window: int = 96 * 90, tau: float = 0.1,
                  blend: float = 0.5) -> mx.array:
    """Pooled Sharpe blended with the SoftMin worst-window Sharpe (DeePM).

    Windows are 90-day blocks, matching the fold test length so the worst-case term is
    measured on the same horizon the evaluation uses.
    """
    pooled = sharpe(r_net)
    T = r_net.shape[-1]
    n = max(1, T // window)
    if n < 2:
        return -pooled
    blocks = [sharpe(r_net[..., i * window:(i + 1) * window]) for i in range(n)]
    worst = soft_min(mx.stack(blocks), tau)
    return -((1 - blend) * pooled + blend * worst)


def participation_penalty(w: mx.array, min_exposure: float, weight: float,
                          mask: mx.array | None = None) -> mx.array:
    """Penalise the shortfall below a minimum mean exposure.

    Without this, w == 0 is a GLOBAL OPTIMUM of robust_sharpe whenever nothing is
    profitable: Sharpe of a zero-return series is 0/(0+EPS) = 0, its worst-window term is
    also 0, so inaction scores 0 while any real strategy that loses scores worse. Measured
    2026-09-10 on fold f08 — all six robust_sharpe arms converged to mean exposure 0.000
    and 100% flat, ranking top purely by never taking a position.

    A flat policy is a legitimate ECONOMIC answer but a vacuous EXPERIMENTAL one: it does
    not test whether direct policy optimisation converts signal into P&L. The floor is
    deliberately low so it removes the trivial optimum without dictating an exposure level.

    Quadratic in the shortfall, so the gradient vanishes smoothly as the floor is met
    rather than imposing a cliff at it.

    `mask` is REQUIRED whenever the panel has gaps. Exposure is measured over bars the
    policy can actually trade, i.e. mean(w*mask)/mean(mask), never mean(w) over the whole
    window. Masked bars carry w=0 by construction, so including them understates exposure
    by exactly the invalid fraction and penalises the policy for data gaps it does not
    control. Measured 2026-09-11 on fold f08: the earliest training chunks are ~30% valid
    (only 3 of 10 pairs have 2021 history), so an unnormalised mean demanded true exposure
    of 0.05/0.30 = 0.167 on tradeable bars to clear a 0.05 floor. That distorted even the
    utility arm, which never had the degeneracy the guard exists to prevent.
    """
    if mask is None:
        mean_w = mx.mean(w)
    else:
        denom = mx.maximum(mx.mean(mask), mx.array(1e-6))
        mean_w = mx.mean(w * mask) / denom
    shortfall = mx.maximum(mx.array(min_exposure) - mean_w, mx.array(0.0))
    return weight * mx.square(shortfall)


def turnover_penalty(w: mx.array, lam: float) -> mx.array:
    """lambda * mean|dw| — the explicit alternative to a learned no-trade band."""
    dw = mx.abs(w[..., 1:] - w[..., :-1])
    return lam * mx.mean(dw)


OBJECTIVES = {"utility": utility, "robust_sharpe": robust_sharpe}


def exposure_ce(w: mx.array, y_onehot: mx.array) -> mx.array:
    """Cross-entropy between exposure and the directional label (Arm B).

    The policy emits a scalar exposure, not class probabilities, so the label term is
    framed as: exposure should be HIGH where the label says BUY and LOW otherwise. That is
    a binary cross-entropy against the BUY indicator, which is the only part of the
    3-class label a long-only exposure can express — a long-only policy cannot act on SELL
    beyond going flat, so treating SELL and HOLD alike here is the faithful mapping, not a
    simplification.
    """
    buy = y_onehot[..., 2]
    p = mx.clip(w, 1e-6, 1.0 - 1e-6)
    return -mx.mean(buy * mx.log(p) + (1.0 - buy) * mx.log(1.0 - p))


def aux_vol_loss(w: mx.array, vol_target: mx.array) -> mx.array:
    """Auxiliary head loss for Arm C: predict forward realised volatility.

    Scale-free (the target is standardised within the window) so it cannot dominate the
    profit term merely by being large. Carries NO directional information by construction —
    the target is a magnitude, asserted non-negative and near-zero-correlated with forward
    return in test_purge_embargo.
    """
    t = vol_target
    t = (t - mx.mean(t)) / (mx.sqrt(mx.var(t)) + EPS)
    pred = (w - mx.mean(w)) / (mx.sqrt(mx.var(w)) + EPS)
    return mx.mean(mx.square(pred - t))
