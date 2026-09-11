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


def participation_penalty(w: mx.array, min_exposure: float, weight: float) -> mx.array:
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
    """
    shortfall = mx.maximum(mx.array(min_exposure) - mx.mean(w), mx.array(0.0))
    return weight * mx.square(shortfall)


def turnover_penalty(w: mx.array, lam: float) -> mx.array:
    """lambda * mean|dw| — the explicit alternative to a learned no-trade band."""
    dw = mx.abs(w[..., 1:] - w[..., :-1])
    return lam * mx.mean(dw)


OBJECTIVES = {"utility": utility, "robust_sharpe": robust_sharpe}
