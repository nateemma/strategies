"""Pair-restricted comparison of two arms (feature 008).

Compares two strategies on ONE pair's trades. Exists because no current tool does this:
compare.py compares whole arms, trade_forensics.py aggregates by exit reason. Neither is usable
when one arm trades a single pair and the other trades eleven.

WHY THIS SHAPE. LINK contributes 8 trades on W1 against a noise floor of 7.633pp measured on 99.
Total return cannot resolve anything at that count. Two things still can:

  * WHICH trades each model selects -- a set comparison, well determined at n=8.
  * PER-TRADE EXPECTANCY -- at n=8 with sd 1.47 the standard error is ~0.52pp, so an expectancy
    of +2.27% is resolvable as positive even though the total is not.

So this reports selection and expectancy as first-class results, and reports any total-return
delta only alongside the floor that makes it meaningless.

The bootstrap is compare.py's, reached through its trade-list entry point -- NOT reimplemented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "user_data/strategies"))

from regime.tail.compare import BOOTSTRAP_SEED, MIN_DIFF_TRADES, N_RESAMPLES  # noqa: E402

ARMS = ROOT / "user_data/strategies/regime/arms"
ARCHIVES = ROOT / "user_data/backtest_results"
BASELINE_FLOOR_PP = 7.633          # measured 2026-09-09 on 99 W1 trades
MIN_N_FOR_WIN_RATE = 20            # FR-021


def load(tag: str, window: str, pair: str):
    p = ARMS / f"{tag}_{window}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    if rec.get("returncode") != 0:
        return None
    z = ARCHIVES / (rec.get("result_file") or "")
    if not z.exists():
        return None
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and "config" not in n]
        d = json.loads(zf.read(names[0]))
    trades = d["strategy"][rec["strategy"]]["trades"]
    # exit tags carry a TRAILING SPACE (exit_tag += "model_exit ") -- strip before any comparison
    for t in trades:
        t["exit_reason"] = (t.get("exit_reason") or "").strip()
    return rec, [t for t in trades if t["pair"] == pair]


def boot_mean(x: np.ndarray) -> tuple:
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    b = x[rng.integers(0, len(x), (N_RESAMPLES, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_sum(x: np.ndarray, scale: float) -> tuple:
    if len(x) < 2:
        return float(x.sum() * scale), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    b = x[rng.integers(0, len(x), (N_RESAMPLES, len(x)))].sum(axis=1) * scale
    return float(x.sum() * scale), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _scale_for(rec: dict) -> float:
    """pp per unit of profit_abs, from the arm's own totals (all pairs)."""
    z = ARCHIVES / (rec.get("result_file") or "")
    try:
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json") and "config" not in n]
            d = json.loads(zf.read(names[0]))
        tot = sum(t["profit_abs"] for t in d["strategy"][rec["strategy"]]["trades"])
        if tot and rec.get("wallet_profit_pct"):
            return rec["wallet_profit_pct"] / tot
    except Exception:
        pass
    return 100.0 / 10000.0


def report_window(arm_tag, ctl_tag, pair, window, out):
    a = load(arm_tag, window, pair)
    c = load(ctl_tag, window, pair)
    out.append(f"\n=== {window} — {pair} only " + ("[OOS, verdict window]" if window == "W1"
                                                   else "[in-sample]") + " ===")
    if a is None or c is None:
        out.append("  missing arm or control record"); return
    (arec, at), (crec, ct) = a, c
    if not at and not ct:
        out.append("  ZERO-TRADE for both arms — inconclusive, NOT a zero return (FR-011)"); return

    key = lambda t: (t["pair"], t["open_date"])
    A = {key(t): t for t in at}
    C = {key(t): t for t in ct}
    shared = set(A) & set(C)
    identical = sum(1 for k in shared if abs(A[k]["profit_abs"] - C[k]["profit_abs"]) < 1e-9)
    a_only, c_only = set(A) - set(C), set(C) - set(A)

    # ---- 1. selection (US2, FR-006) ----
    out.append(f"  SELECTION   variant {len(at)} trades | control {len(ct)} trades")
    out.append(f"    shared {len(shared)} (identical {identical})  "
               f"variant-only {len(a_only)}  control-only {len(c_only)}")
    if not a_only and not c_only and identical == len(shared):
        out.append("    VERDICT: SAME SELECTION — the two models pick the same opportunities.")
        out.append("             That is a POSITIVE finding (FR-010): cross-pair training")
        out.append("             contributes nothing to selection on this pair.")
    else:
        out.append("    VERDICT: SELECTION DIFFERS")
        for k in sorted(a_only):
            out.append(f"      variant-only  {k[1][:10]}  {A[k]['profit_ratio']*100:+.2f}%")
        for k in sorted(c_only):
            out.append(f"      control-only  {k[1][:10]}  {C[k]['profit_ratio']*100:+.2f}%")

    # ---- 2. per-trade expectancy (US3, FR-007) ----
    out.append("  EXPECTANCY  (per trade, %)")
    for lbl, tr in (("variant", at), ("control", ct)):
        if not tr:
            out.append(f"    {lbl:<8} zero-trade"); continue
        x = np.array([t["profit_ratio"] for t in tr]) * 100
        m, lo, hi = boot_mean(x)
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "(n<2, no interval)"
        wr = (f"  win {100*(x > 0).mean():.0f}%" if len(x) >= MIN_N_FOR_WIN_RATE
              else f"  win rate suppressed (n={len(x)}<{MIN_N_FOR_WIN_RATE}, FR-021)")
        out.append(f"    {lbl:<8} n={len(x):<3} mean {m:+.3f}%  95% CI {ci}{wr}")

    # ---- 3. total-return delta, with the floor that makes it meaningless (FR-008/009) ----
    # pp per unit profit_abs, calibrated on the WHOLE arm, not the pair subset. Dividing by the
    # pair's own abs would force the pair's return to equal the whole window's -- which it did on
    # the first run, printing LINK's W1 contribution as the window's +9.831pp instead of ~1.1pp.
    scale = _scale_for(crec)
    n_diff = len(a_only) + len(c_only) + (len(shared) - identical)
    da = sum(t["profit_abs"] for t in at) * scale
    dc = sum(t["profit_abs"] for t in ct) * scale
    out.append(f"  RETURN      (this pair's contribution to the whole-portfolio return)")
    out.append(f"    variant {da:+.3f}pp | control {dc:+.3f}pp | delta {da-dc:+.3f}pp")
    if n_diff < MIN_DIFF_TRADES:
        out.append(f"    UNRESOLVED — only {n_diff} differing trades (< {MIN_DIFF_TRADES}); the")
        out.append(f"    bootstrap cannot estimate a distribution. Floor on this window is"
                   f" ~{BASELINE_FLOOR_PP}pp measured on 99 trades; this pair has {len(ct)}.")
    else:
        out.append(f"    {n_diff} differing trades; floor ~{BASELINE_FLOOR_PP}pp on 99 trades")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--pair", default="LINK/USDT")
    ap.add_argument("--windows", default="W1,W2,W3")
    a = ap.parse_args()
    out = [f"PAIR-RESTRICTED COMPARISON — {a.pair}",
           f"  arm {a.arm}  vs  control {a.control}",
           "  Success is NOT a P&L win: this pair's trade count cannot resolve returns.",
           "  Read SELECTION and EXPECTANCY; RETURN is shown only with its floor."]
    for w in a.windows.split(","):
        report_window(a.arm, a.control, a.pair, w.strip(), out)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
