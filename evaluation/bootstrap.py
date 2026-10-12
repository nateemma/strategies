"""The paired bootstrap — ONE statistic for single-window and pooled walk-forward arms.

Lifted verbatim from `regime/tail/compare.py` when the walk-forward harness was promoted out
of `regime/` (docs/WALKFORWARD_TODO.md item 1, step 2). The reuse was deliberate from the
start: single-window arms and pooled fold sets are judged by the SAME statistic, not by two
implementations that drift apart. So it moved here with the harness rather than being copied,
and `regime/tail/compare.py` now imports it back.

NOTHING IN THIS FILE MAY BE RETUNED. `BOOTSTRAP_SEED`, `N_RESAMPLES`, `CI` and
`MIN_DIFF_TRADES` were measured and pinned on 2026-09-09; changing any of them silently
rebases every verdict this repository has already recorded.

Why the bootstrap rather than a flat threshold, from the original module:

  The old POWER_THRESHOLD_PP = 1.0 was a rule of thumb that was never derived. Measured
  2026-09-09, the NNNC W1 control's own bootstrap 95% CI is [2.383%, 17.649%] on 99 trades --
  a ±7.6pp half-width, so the 1.0pp rule was ~7.6x too lenient and passed noise as signal. A
  +1.86pp delta it called IMPROVED bootstraps to [-4.29, +7.93]pp, p=0.54. If the CI spans
  zero, the delta is NOT RESOLVED, whatever its sign or size.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

BT_RESULTS = Path("user_data/backtest_results")

MIN_DIFF_TRADES = 20            # below this the percentile bootstrap is degenerate:
                                # with 1 differing trade it resamples the same value
                                # every time, variance collapses to 0 and it reports
                                # p=0.000 on a +0.01pp delta. Too few items to
                                # estimate a sampling distribution is a POWER limit,
                                # not a null and certainly not a win.
N_RESAMPLES = 20000
BOOTSTRAP_SEED = 0              # fixed: the verdict must be reproducible
CI = (2.5, 97.5)

VERDICTS = ("inconclusive_absorbed", "improved", "worsened",
            "unchanged", "unresolvable", "inconclusive_underpowered")


def _trades(rec: dict) -> list | None:
    """Per-trade rows for an arm record, from the archive it names.

    run_arm.py stores `result_file`; --export-filename is deprecated and silently
    ignored by freqtrade, so the archive under user_data/backtest_results/ is the
    only per-trade source. Returns None if the archive has been pruned.
    """
    rf = rec.get("result_file")
    if not rf:
        return None
    z = BT_RESULTS / rf
    if not z.exists():
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist()
                     if n.endswith(".json") and "config" not in n]
            if not names:
                return None
            data = json.loads(zf.read(names[0]))
        return data["strategy"][rec["strategy"]]["trades"]
    except Exception:
        return None


def _scale(rec: dict, trades: list) -> float:
    """pp per unit of profit_abs, self-calibrated from the record itself.

    Avoids assuming a 10,000 starting balance. Verified exact on NNNC W1
    (983.1 abs -> 9.831%).
    """
    tot = sum(t["profit_abs"] for t in trades)
    if tot and rec.get("wallet_profit_pct"):
        return rec["wallet_profit_pct"] / tot
    return 100.0 / 10000.0


def bootstrap_delta(arm: dict, ctl: dict,
                    arm_trades: list | None = None,
                    ctl_trades: list | None = None,
                    key_fields: tuple = ("pair", "open_date")) -> dict | None:
    """Paired bootstrap of the arm-minus-control return difference.

    Pairs on (pair, open_date). Trades present and IDENTICAL in both contribute
    exactly zero and are excluded, which is what makes this tighter than comparing
    two independent totals -- for an exit-side change sharing every entry, only the
    handful of genuinely differing trades carry variance. Trades unique to one side
    enter with their own sign (the membership/symmetric-difference case, correct
    when the entry set itself moves).
    """
    # 2026-09-09: trade lists may be supplied directly, so a POOLED walk-forward set can be
    # compared with THIS statistic instead of a second, divergent one (FR-013 of feature 007).
    # Pooled sets pair on (fold_id, pair, open_date) because the same pair/time can legitimately
    # recur across folds. Records are still required, for the pp scale.
    at = arm_trades if arm_trades is not None else _trades(arm)
    ct = ctl_trades if ctl_trades is not None else _trades(ctl)
    if at is None or ct is None:
        return None
    scale = _scale(ctl, ct)
    key = lambda t: tuple(t[k] for k in key_fields)
    # summed per key, not overwritten: one backtest never repeats a key, but a union of
    # separate books can (two books entering one pair on one candle, 026 addendum 5)
    a: dict = {}
    c: dict = {}
    for side, trades in ((a, at), (c, ct)):
        for t in trades:
            side[key(t)] = side.get(key(t), 0.0) + t["profit_abs"]
    shared = set(a) & set(c)
    identical = sum(1 for k in shared if abs(a[k] - c[k]) < 1e-9)
    diffs = [a[k] - c[k] for k in shared if abs(a[k] - c[k]) >= 1e-9]
    diffs += [a[k] for k in set(a) - set(c)]
    diffs += [-c[k] for k in set(c) - set(a)]
    if not diffs:
        return {"n_diff": 0, "identical": identical, "observed": 0.0,
                "lo": 0.0, "hi": 0.0, "p": 1.0, "resolved": False}
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = d[rng.integers(0, n, (N_RESAMPLES, n))].sum(axis=1) * scale
    lo, hi = np.percentile(boot, CI)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"n_diff": n, "identical": identical, "observed": float(d.sum() * scale),
            "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0)),
            "resolved": bool(lo * hi > 0),
            "shared": len(shared), "arm_only": len(set(a) - set(c)),
            "ctl_only": len(set(c) - set(a))}
