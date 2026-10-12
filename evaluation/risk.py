"""Paired risk verdict: daily Sharpe and maximum drawdown, arm vs control (026 addendum 4).

The P&L verdict (`compare_pooled.compare_data`) asks whether an arm makes MORE money. This asks
whether it makes its money with LESS risk, on the same days. It is reported beside the P&L verdict
and never replaces it.

Series: each side's daily realised P&L in pp (profit_abs x scale_pp_per_abs, summed by close
date), zero-filled over the union of both sides' active span. Realised, not mark-to-market, so
drawdown inside an open trade is not seen -- the same view for both sides.

Statistics (arm minus control):
    sharpe_delta   annualised daily Sharpe (mean / sd x sqrt(365))   higher is better
    maxdd_delta    maximum drawdown of cumulative pp                 LOWER is better

Bootstrap (pre-declared, pinned): PAIRED stationary block bootstrap (Politis & Romano 1994) over
days -- both series resampled with the same indices, blocks keep the autocorrelation a drawdown
depends on. Mean block length 10 days, 2000 resamples, seed 20261011. 95% percentile interval.
Verdict per statistic: IMPROVED / WORSENED when the interval excludes 0 in that direction, else
UNCHANGED. Two-sided p = 2 x min(share <= 0, share >= 0) of the bootstrap deltas.
EXPOSURE: Sharpe is invariant to position size; maximum drawdown is not. An arm holding less
capital on average draws down less for that reason alone (half-size copy -> maxdd IMPROVED, Sharpe
UNCHANGED; see the tests). Read a drawdown verdict beside the arms' mean committed capital
(`evaluation.books.mean_committed`), or compare at matched exposure.
PROVISIONAL under quality gate 2 like every other verdict here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BLOCK_MEAN_DAYS = 10
N_BOOT = 2000
SEED = 20261011
ANNUAL = np.sqrt(365.0)


def daily_pp(trades: list[dict], span: pd.DatetimeIndex | None = None) -> pd.Series:
    """Daily realised pp by close date, zero-filled over `span` (default: own first..last day)."""
    if not trades:
        return pd.Series(0.0, index=span if span is not None else pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(trades)
    pp = df["profit_abs"].fillna(0) * df["scale_pp_per_abs"].fillna(0)
    day = pd.to_datetime(df["close_date"], utc=True).dt.floor("D")
    s = pp.groupby(day).sum()
    if span is None:
        span = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(span, fill_value=0.0)


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * ANNUAL) if sd > 0 else float("nan")


def max_drawdown(x: np.ndarray) -> float:
    eq = np.cumsum(x)
    return float(np.max(np.maximum.accumulate(np.maximum(eq, 0.0)) - eq))


def _stationary_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) stationary-bootstrap index paths: geometric blocks, wrap-around."""
    jumps = rng.random((n_boot, n)) < 1.0 / BLOCK_MEAN_DAYS
    jumps[:, 0] = True
    starts = rng.integers(n, size=(n_boot, n))
    t = np.arange(n)
    # position of the most recent block start at or before t, per row
    last = np.maximum.accumulate(np.where(jumps, t, 0), axis=1)
    start_val = np.take_along_axis(starts, last, axis=1)
    return (start_val + (t - last)) % n


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, x.mean(axis=1) / sd * ANNUAL, np.nan)


def _maxdd_rows(x: np.ndarray) -> np.ndarray:
    eq = np.cumsum(x, axis=1)
    return np.max(np.maximum.accumulate(np.maximum(eq, 0.0), axis=1) - eq, axis=1)


def _verdict(lo: float, hi: float, better_if_positive: bool) -> str:
    if lo > 0:
        return "IMPROVED" if better_if_positive else "WORSENED"
    if hi < 0:
        return "WORSENED" if better_if_positive else "IMPROVED"
    return "UNCHANGED"


def compare(arm_trades: list[dict], ctl_trades: list[dict]) -> dict:
    """The paired risk comparison as data (see module docstring)."""
    both = (*arm_trades, *ctl_trades)
    if not both or any(not t.get("close_date") for t in both):  # no realised dates: no series
        return {"verdict_sharpe": "UNRESOLVABLE", "verdict_maxdd": "UNRESOLVABLE"}
    days = [pd.to_datetime(t["close_date"], utc=True).floor("D") for t in both]
    span = pd.date_range(min(days), max(days), freq="D")
    a = daily_pp(arm_trades, span).to_numpy()
    c = daily_pp(ctl_trades, span).to_numpy()
    n = len(span)
    obs = {"sharpe": sharpe(a) - sharpe(c), "maxdd": max_drawdown(a) - max_drawdown(c)}
    idx = _stationary_indices(n, N_BOOT, np.random.default_rng(SEED))
    A, C = a[idx], c[idx]
    boot = {"sharpe": _sharpe_rows(A) - _sharpe_rows(C), "maxdd": _maxdd_rows(A) - _maxdd_rows(C)}
    out = {
        "days": n,
        "arm": {"sharpe": sharpe(a), "maxdd": max_drawdown(a), "daily_sd": float(a.std(ddof=1))},
        "ctl": {"sharpe": sharpe(c), "maxdd": max_drawdown(c), "daily_sd": float(c.std(ddof=1))},
    }
    for k, better_up in (("sharpe", True), ("maxdd", False)):
        d = boot[k][np.isfinite(boot[k])]
        if len(d) < N_BOOT // 2 or not np.isfinite(obs[k]):
            out[f"verdict_{k}"] = "UNRESOLVABLE"
            continue
        lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
        out[f"{k}_delta"] = obs[k]
        out[f"{k}_ci"] = [lo, hi]
        out[f"{k}_p"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        out[f"verdict_{k}"] = _verdict(lo, hi, better_up)
    return out


def line(r: dict) -> str:
    """One-line summary for decision files."""
    if "sharpe_delta" not in r and "maxdd_delta" not in r:
        return f"risk: Sharpe {r.get('verdict_sharpe')}, max drawdown {r.get('verdict_maxdd')}"
    parts = []
    for k, name in (("sharpe", "Sharpe"), ("maxdd", "max drawdown")):
        if f"{k}_delta" in r:
            lo, hi = r[f"{k}_ci"]
            parts.append(
                f"{name} **{r[f'verdict_{k}']}** {r['arm'][k]:.2f} vs {r['ctl'][k]:.2f} "
                f"(delta {r[f'{k}_delta']:+.2f}, CI [{lo:+.2f}, {hi:+.2f}], p {r[f'{k}_p']:.3f})"
            )
        else:
            parts.append(f"{name} {r.get(f'verdict_{k}')}")
    return f"risk ({r['days']} days, realised daily pp): " + "; ".join(parts)
