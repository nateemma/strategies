"""Statistics: IQM, bootstrap CIs, turnover/hold time, DSR and PBO (T009, T031, T044).

FR-016  >=5 seeds, interquartile mean with stratified bootstrap CI, never a best seed;
        seeds producing identical trades count as one measurement
FR-024  Deflated Sharpe Ratio and PBO (CSCV) over the FULL trial log
FR-025  a delta below the floor reads "not resolved", never "no effect"
"""
from __future__ import annotations

import numpy as np

FLOOR_PP_PER_YEAR = 3.35          # pooled walk-forward resolvable floor
SINGLE_WINDOW_FLOOR_PP = 7.633    # W1 noise floor (AGENT_GUIDE)


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50%."""
    a = np.sort(np.asarray(x, dtype=float))
    if a.size == 0:
        return float("nan")
    lo, hi = np.percentile(a, 25), np.percentile(a, 75)
    m = a[(a >= lo) & (a <= hi)]
    return float(m.mean() if m.size else a.mean())


def dedupe_identical(seed_trades: dict[int, tuple]) -> dict[int, tuple]:
    """Seeds producing identical trades are one measurement, not two (FR-016)."""
    seen, out = {}, {}
    for seed, trades in seed_trades.items():
        key = hash(trades)
        if key not in seen:
            seen[key] = seed
            out[seed] = trades
    return out


def bootstrap_ci(x, n: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def verdict(delta_pp_yr: float, ci: tuple[float, float],
            floor: float = FLOOR_PP_PER_YEAR) -> str:
    """FR-025 / SC-009: below the floor is 'not resolved', never 'no effect'."""
    lo, hi = ci
    if delta_pp_yr >= floor and lo > 0:
        return "accept-candidate"
    if lo <= 0 <= hi or abs(delta_pp_yr) < floor:
        return "not resolved"
    return "not resolved"


def resolvable(delta_pp: float, floor: float = SINGLE_WINDOW_FLOOR_PP) -> bool:
    """T032a: is a single-window difference bigger than the measured noise floor?"""
    return abs(delta_pp) > floor


def deflated_sharpe(sr: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio."""
    from math import erf, log, sqrt
    if n_trials < 2 or n_obs < 2:
        return float("nan")
    e, g = 0.5772156649, 1.0
    z = sqrt(2 * log(n_trials))
    sr0 = (z - (log(log(n_trials)) + log(4 * np.pi)) / (2 * z) + e / z) * g / sqrt(n_obs)
    denom = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    stat = (sr - sr0) * sqrt(n_obs - 1) / denom
    return float(0.5 * (1 + erf(stat / sqrt(2))))


def pbo_cscv(perf: np.ndarray, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2015).

    perf: (n_observations, n_configurations). PBO > 0.2 fails the feature (SC-006).

    CAVEAT — measured 2026-09-10, and it bears on SC-006 directly. CSCV is biased LOW when
    the configuration count is small. On pure noise, where the true PBO is 0.5, this
    implementation returns:

        n_cfg=8   -> 0.148      n_cfg=20 -> 0.385      n_cfg=50 -> 0.521

    (Validated at the other end: 0.000 when one config is genuinely superior, 0.529 for a
    pure in-sample overfit.) This feature's screening grid is **12 configurations**, which
    sits in the biased-low regime, so a raw PBO from the grid alone would clear the 0.2 gate
    more easily than intended. Compute PBO over the FULL trial log — every seed, every
    variant, every abandoned run (FR-024) — which is what pushes the configuration count out
    of the biased regime. Report the configuration count alongside the PBO figure so the
    reader can judge it.
    """
    from itertools import combinations
    m = np.asarray(perf, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return float("nan")
    n = m.shape[0] - (m.shape[0] % n_splits)
    blocks = np.array_split(m[:n], n_splits)
    half = n_splits // 2
    lam = []
    for combo in combinations(range(n_splits), half):
        rest = [i for i in range(n_splits) if i not in combo]
        is_ = np.vstack([blocks[i] for i in combo]).mean(axis=0)
        oos = np.vstack([blocks[i] for i in rest]).mean(axis=0)
        best = int(np.argmax(is_))
        rank = float((oos <= oos[best]).sum()) / len(oos)
        w = max(min(rank, 1 - 1e-9), 1e-9)
        lam.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(lam) < 0))


# ---------------------------------------------------------------------------
# T036 / FR-019 / SC-014 — activity metrics beside EVERY return.
#
# A degenerate arm can post an excellent return by doing almost nothing: it avoids the
# control's losses without ever taking a position. Reporting a return alone cannot
# distinguish that from an edge, so exposure, trades per year and holding time travel
# WITH the return everywhere it is reported.
# ---------------------------------------------------------------------------

BARS_PER_YEAR = 96 * 365          # 15-minute bars
FLAT_EPS = 0.01                   # exposure below this counts as flat


def activity(w: np.ndarray, bars_per_year: int = BARS_PER_YEAR,
             eps: float = FLAT_EPS) -> dict:
    """Activity metrics for an exposure path (FR-019, SC-014).

    `w` is (n_pairs, T) or (T,) of long-only exposure in [0, 1]. A "trade" is one
    contiguous run of non-flat exposure, so holding time is the length of that run --
    the same notion of a trade the backtest archive uses, not a count of weight changes.

    Returns per-pair averages: an arm holding one pair forever and nine flat should not
    read as nine times less active than it is.
    """
    a = np.atleast_2d(np.asarray(w, dtype=float))
    n_pairs, T = a.shape
    if T == 0:
        return {"mean_exposure": float("nan"), "flat_share": float("nan"),
                "trades_per_year": float("nan"), "mean_hold_bars": float("nan"),
                "mean_hold_hours": float("nan"), "turnover_per_year": float("nan")}

    live = a >= eps
    # A trade opens on a flat -> live transition. Prepending False counts a path that
    # starts already live as one open, which is what the archive would record.
    opens = np.diff(np.concatenate([np.zeros((n_pairs, 1), bool), live], axis=1).astype(np.int8),
                    axis=1) == 1
    n_trades = opens.sum(axis=1).astype(float)
    live_bars = live.sum(axis=1).astype(float)

    years = T / bars_per_year
    with np.errstate(divide="ignore", invalid="ignore"):
        hold = np.where(n_trades > 0, live_bars / np.maximum(n_trades, 1), np.nan)

    return {"mean_exposure": float(a.mean()),
            "flat_share": float((~live).mean()),
            "trades_per_year": float((n_trades / years).mean()),
            "mean_hold_bars": float(np.nanmean(hold)) if np.isfinite(hold).any() else float("nan"),
            "mean_hold_hours": (float(np.nanmean(hold)) / 4.0
                                if np.isfinite(hold).any() else float("nan")),
            "turnover_per_year": float((np.abs(np.diff(a, axis=1)).sum(axis=1) / years).mean())}


def format_with_activity(label: str, log_return: float, act: dict) -> str:
    """One line carrying a return and its activity, so neither can be quoted alone."""
    return (f"{label:<26} log_ret={log_return:+.4f}  w={act['mean_exposure']:.3f}  "
            f"trades/yr={act['trades_per_year']:.1f}  hold={act['mean_hold_hours']:.1f}h  "
            f"flat={100 * act['flat_share']:.0f}%")


# ---------------------------------------------------------------------------
# T037 / SC-004 / FR-024 — DSR and PBO over the FULL trial log.
#
# "Full" is load-bearing. The trial count is what deflates the statistic, so every
# abandoned run, every void run and every seed counts. Dropping void runs is precisely
# the selection these statistics exist to correct for.
# ---------------------------------------------------------------------------

def trial_log_deflation(records: list[dict] | None = None) -> dict:
    """DSR and PBO computed over every configuration ever pre-registered.

    Returns the configuration count beside the figures: CSCV is biased LOW at small
    counts (see pbo_cscv), so a PBO is not interpretable without it.
    """
    if records is None:
        from research.labelfree import trials
        records = trials.read()

    pre = [r for r in records if "written_at" in r]
    done = {}
    for r in records:
        if "completed_at" not in r:
            continue
        res = r.get("result") or {}
        if res.get("test_log_return") is not None:
            done[r["trial_id"]] = res            # last write wins: duplicates are one trial

    # A configuration is an arm/objective spec; its seeds are repeat observations of it.
    by_cfg: dict[str, dict[int, float]] = {}
    for tid, res in done.items():
        base = tid.split("@")[0]
        cfg, _, tail = base.rpartition("-s")
        seed = tail.split("-")[0]
        key = cfg or base
        try:
            s = int(seed)
        except ValueError:
            s = 0
        by_cfg.setdefault(key, {})[s] = float(res["test_log_return"])

    seeds = sorted({s for v in by_cfg.values() for s in v})
    cfgs = sorted(k for k, v in by_cfg.items() if len(v) >= 2)
    perf = np.array([[by_cfg[c].get(s, np.nan) for c in cfgs] for s in seeds], dtype=float)
    # CSCV needs a complete matrix; configurations missing a seed are dropped rather
    # than imputed, because an imputed cell invents an observation.
    keep = ~np.isnan(perf).any(axis=0)
    perf, cfgs = perf[:, keep], [c for c, k in zip(cfgs, keep) if k]

    n_trials = len(pre)                       # deflation counts PRE-REGISTRATIONS
    out = {"n_pre_registered": n_trials,
           "n_completed_with_return": len(done),
           "n_configurations": len(cfgs),
           "n_seeds": len(seeds),
           "configurations": cfgs}

    if perf.size and perf.shape[1] >= 2:
        # Sharpe across seeds of the best configuration, the quantity being deflated.
        m = perf.mean(axis=0)
        best = int(np.argmax(m))
        col = perf[:, best]
        sd = col.std(ddof=1)
        sr = float(col.mean() / sd) if sd > 1e-12 else 0.0
        out["best_configuration"] = cfgs[best]
        out["best_sharpe_across_seeds"] = sr
        out["dsr"] = deflated_sharpe(sr, n_obs=len(col), n_trials=n_trials)
        n_splits = min(16, (perf.shape[0] // 2) * 2)
        out["pbo"] = (pbo_cscv(perf, n_splits=n_splits) if n_splits >= 2 else float("nan"))
        out["pbo_n_splits"] = n_splits
        out["pbo_caveat"] = (
            f"CSCV is biased low below ~50 configurations; this is {len(cfgs)}. "
            "Read the figure with the count, never alone.")
    else:
        out["dsr"] = out["pbo"] = float("nan")
        out["reason"] = "fewer than 2 complete configurations"

    # ---- Interpretability gate. Do NOT report a pass that the inputs cannot support.
    #
    # Both statistics are defined over a TIME axis: DSR deflates a Sharpe computed from a
    # return series, CSCV partitions that series into blocks. The trial log stores only
    # summary scalars per run, so the only axis available here is SEEDS -- 5 of them.
    # Computed on that axis, the "Sharpe" measures seed-consistency rather than
    # risk-adjusted return, and CSCV gets 4 blocks instead of 16. Both then read
    # optimistically: a configuration that is reliably mediocre scores a high Sharpe
    # because its across-seed dispersion is tiny.
    #
    # Reporting that as "SC-004 satisfied" would be the exact error the deflation exists
    # to prevent, so the flags below stay False and say why. Satisfying SC-004 as written
    # requires per-bar net returns stored per trial, which is a trial-log schema change.
    reasons = []
    if len(seeds) < 30:
        reasons.append(f"only {len(seeds)} observations (seeds), not a return series; "
                       "DSR and CSCV both need a time axis")
    if len(cfgs) < 50:
        reasons.append(f"{len(cfgs)} configurations is inside the CSCV biased-low regime")
    if out.get("pbo_n_splits", 0) < 8:
        reasons.append(f"CSCV ran with {out.get('pbo_n_splits')} splits, not 16")
    out["interpretable"] = not reasons
    out["not_interpretable_because"] = reasons
    out["sc004_pbo_ok"] = bool(np.isfinite(out.get("pbo", np.nan))
                               and out["pbo"] <= 0.2 and out["interpretable"])
    out["sc004_dsr_ok"] = bool(np.isfinite(out.get("dsr", np.nan))
                               and out["dsr"] > 0.5 and out["interpretable"])
    out["sc004_verdict"] = ("SATISFIED" if out["sc004_pbo_ok"] and out["sc004_dsr_ok"]
                            else "CANNOT BE EVALUATED — see not_interpretable_because")
    return out


# ---------------------------------------------------------------------------
# T033b / SC-008 — in-sample overfitting differential.
# ---------------------------------------------------------------------------

def annualised_pp(log_delta: float, days: float) -> float:
    """A difference of log returns over `days`, expressed as percentage points per year.

    A log difference is a ratio of growth factors, so it annualises by scaling the log and
    then exponentiating once -- not by scaling a percentage. Doing it the other way is the
    error that made the first walk-forward power report call a working instrument a 0.62x
    regression.
    """
    if days <= 0:
        return float("nan")
    return float((np.exp(log_delta * 365.0 / days) - 1.0) * 100.0)


def overfitting_differential(screen_advantage: float, heldout_advantage: float,
                             screen_days: float, heldout_days: float,
                             floor: float = FLOOR_PP_PER_YEAR) -> dict:
    """SC-008: screening-fold advantage minus held-out advantage, over the control.

    Selection happened on the screening fold, so any advantage that does not survive on a
    fold never used for selection is selection gain, not edge. Exceeding the floor is
    recorded as overfitting and the configuration stops.

    Reported in BOTH units. The absolute pp/yr test is the pre-registered one, but on
    configurations whose advantage is itself enormous, an absolute floor trips on almost
    any shrinkage -- so the retention ratio is reported beside it and neither is quoted
    alone.
    """
    s = annualised_pp(screen_advantage, screen_days)
    h = annualised_pp(heldout_advantage, heldout_days)
    diff_log = screen_advantage - heldout_advantage
    retention = (heldout_advantage / screen_advantage
                 if abs(screen_advantage) > 1e-12 else float("nan"))
    exceeds = bool(abs(s - h) > floor)
    return {"screen_advantage_log": screen_advantage,
            "heldout_advantage_log": heldout_advantage,
            "advantage_shrinkage_log": diff_log,
            "screen_advantage_pp_yr": s,
            "heldout_advantage_pp_yr": h,
            "differential_pp_yr": s - h,
            "retention_ratio": retention,
            "floor_pp_yr": floor,
            "exceeds_floor": exceeds,
            "verdict": ("OVERFITTING — configuration stops (SC-008)" if exceeds
                        else "within floor — advantage survives selection")}
