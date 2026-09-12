"""The four mandatory baselines (T032a, SC-003).

Brief 01 planned this module and never built it, because its Phase 6 was unreached.
Every figure here is produced by the SAME panel loader, cost model and simulator the arms
use, on the IDENTICAL pair universe and fold range -- a baseline computed through a
different path measures the path, not the baseline.

FROZEN. The EMA rule's constants were fixed before any run and no parameter here is
learned from P&L, so this module writes NOTHING to the trial log (SC-003). Changing a
constant below makes it a fitted configuration and it must then be pre-registered.

Why this matters more than it looks: feature 010 measured its arms against a P&L control
that lost 52% of capital in a fold where the market rose 69.5%. Any arm that merely
declines to trade beats that. Buy-and-hold and DO NOTHING are the baselines that catch it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.labelfree import folds as F          # noqa: E402
from research.labelfree import sim                 # noqa: E402
from research.labelfree import stats as ST         # noqa: E402

# --- PINNED. Do not tune. -------------------------------------------------
EMA_FAST = 20          # bars
EMA_SLOW = 60          # bars
EMA_BAND = 0.1         # +/- dead zone around the crossover, in units of the slow EMA
BURN_IN = 96
FROZEN_AT = "2026-09-12"
# --------------------------------------------------------------------------


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA along the time axis of an (n_pairs, T) array."""
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * x[:, t] + (1 - a) * out[:, t - 1]
    return out


def _rolling_std(x: np.ndarray, span: int) -> np.ndarray:
    """Causal rolling std along time; seeded by the expanding std so early bars are usable."""
    c1 = np.cumsum(x, axis=1)
    c2 = np.cumsum(x * x, axis=1)
    T = x.shape[1]
    idx = np.arange(T)
    lo = np.maximum(idx - span + 1, 0)
    n = (idx - lo + 1).astype(float)
    s1 = c1 - np.where(lo > 0, np.take_along_axis(c1, np.maximum(lo - 1, 0)[None, :]
                                                 .repeat(x.shape[0], 0), axis=1), 0.0)
    s2 = c2 - np.where(lo > 0, np.take_along_axis(c2, np.maximum(lo - 1, 0)[None, :]
                                                 .repeat(x.shape[0], 0), axis=1), 0.0)
    var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
    return np.sqrt(var)


def ema_trend_exposure(price: np.ndarray) -> np.ndarray:
    """PINNED 20/60-bar TSMOM rule with a +/-0.1 band, long-only, fully invested.

    Protocol wording is "20/60-bar EMA SIGN, band +/-0.1". The signal is therefore the
    vol-normalised EMA spread, as time-series momentum is conventionally defined -- NOT
    the raw ratio. Measured 2026-09-12: on the raw ratio, +/-0.1 means "fast exceeds slow
    by 10% of slow", which on 15-minute bars with 20/60-bar spans never happens, and the
    rule sat 100% flat on both folds. A baseline that never trades is not a baseline.

    Normalisation uses the rule's OWN slow span (60 bars), causal and rolling, so no new
    constant enters and no future information is used.

    The band is a dead zone, not a smoother: exposure goes to 1 only when the normalised
    spread exceeds +EMA_BAND, back to 0 only below -EMA_BAND, and HOLDS in between --
    which is what stops a rule sitting at its crossover from churning every bar and
    paying the spread each time.
    """
    fast, slow = _ema(price, EMA_FAST), _ema(price, EMA_SLOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(slow > 0, (fast - slow) / slow, 0.0)
    sd = _rolling_std(raw, EMA_SLOW)
    rel = np.where(sd > 1e-12, raw / np.maximum(sd, 1e-12), 0.0)
    w = np.zeros_like(rel)
    state = np.zeros(rel.shape[0])
    for t in range(rel.shape[1]):
        state = np.where(rel[:, t] > EMA_BAND, 1.0,
                         np.where(rel[:, t] < -EMA_BAND, 0.0, state))
        w[:, t] = state
    return w


def _net(w: np.ndarray, P: np.ndarray, C: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Net-of-cost per-bar log returns through the arms' own simulator, hard cap."""
    import mlx.core as mx
    r = sim.net_returns(mx.array(w), mx.array(P), mx.array(C),
                        volume_quote=mx.array(V), soft_cap=False)
    mx.eval(r)
    return np.asarray(r)


def run(fold_id: str = "f08", pairs: list[str] | None = None) -> dict:
    """All four baselines on one fold's held-out range, net of full costs (SC-003)."""
    from research.labelfree.arms import load_panel

    fold = {f.fold_id: f for f in F.folds()}[fold_id]
    pairs = pairs or list(fold.pair_universe)
    _X, P, C, V, _M, _Y, _Vol, _idx, _leak = load_panel(fold, pairs, train=False,
                                                        label_threshold=None)
    T = P.shape[1]
    out: dict = {"fold": fold_id, "test_start": fold.test_start, "test_end": fold.test_end,
                 "n_pairs": len(pairs), "pairs": pairs, "frozen_at": FROZEN_AT,
                 "ema": {"fast": EMA_FAST, "slow": EMA_SLOW, "band": EMA_BAND},
                 "baselines": {}}

    def record(name: str, w: np.ndarray, gross_only: bool = False):
        ww = w[:, BURN_IN:]
        if gross_only:                      # buy & hold pays cost once, at the open
            lr = float(np.log(P[:, -1] / P[:, BURN_IN]).mean())
            net = lr
        else:
            r = _net(w, P, C, V)
            net = float(r[:, BURN_IN:].sum(axis=1).mean())
            lr = net
        act = ST.activity(ww)
        out["baselines"][name] = {"log_return": net, "gross_log_return": lr, **act}

    # 1. DO NOTHING -- the floor any "avoids losses" arm must clear.
    record("do_nothing", np.zeros((len(pairs), T)))
    # 2. Equal-weight buy & hold of the whitelist.
    record("buy_and_hold", np.ones((len(pairs), T)), gross_only=True)
    # 3. PINNED 20/60 EMA trend rule, band +/-0.1, net of full costs.
    record("ema_trend_20_60", ema_trend_exposure(P))
    # 4. Production control at its cost-aware operating point: read from the pooled
    #    walk-forward record rather than re-simulated, because that record IS the
    #    control's audited result (regime/wf/pooled_nnnc.json).
    out["baselines"]["production_control"] = _pooled_control(fold_id)
    return out


def _pooled_control(fold_id: str) -> dict:
    """The NNNC_MLX control's own fold entry from the audited pooled record."""
    p = ROOT / "regime/wf/pooled_nnnc.json"
    if not p.exists():
        return {"status": "unavailable", "path": str(p)}
    d = json.loads(p.read_text())
    per = {f["fold_id"]: f for f in d["per_fold"]}
    f = per.get(fold_id)
    if f is None:
        return {"status": f"fold {fold_id} not in pooled record",
                "folds": sorted(per)}
    return {"status": "ok", "source": "regime/wf/pooled_nnnc.json",
            "return_pct": f["return_pct"], "n_trades": f["n_trades"],
            "test_days": f["test_days"],
            "log_return": (None if f["return_pct"] is None
                           else float(np.log1p(f["return_pct"] / 100.0)))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="f08")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.fold)
    print(f"BASELINES fold {res['fold']}  {res['test_start']} -> {res['test_end']}  "
          f"{res['n_pairs']} pairs   (frozen {res['frozen_at']})")
    for name, b in res["baselines"].items():
        if "log_return" in b and b.get("log_return") is not None:
            print("  " + ST.format_with_activity(name, b["log_return"], b)
                  if "mean_exposure" in b else
                  f"  {name:<26} log_ret={b['log_return']:+.4f}  "
                  f"({b.get('n_trades','?')} trades, {b.get('return_pct')}%)")
        else:
            print(f"  {name:<26} {b.get('status')}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
