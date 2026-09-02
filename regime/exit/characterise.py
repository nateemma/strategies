"""Phase 1 - characterise the exit (T005-T008).

Answers one question: IS THE STOP CUTTING NOISE, OR CUTTING LOSERS?

Method: take the stopped trades that actually happened and replay what price DID
after each exit. No model is loaded, nothing trains, no backtest runs (FR-007).

The two stop populations are reported SEPARATELY throughout (FR-004). 37.4% of
stopped trades exit in profit and 50.6% end with the stop ABOVE entry - those are
a different phenomenon from trades that fell through the stop, and they may want
the opposite remedy. Averaging them would hide that.

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/exit/characterise.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime.exit.baseline import BASELINE, load_trades, verify
from regime.regimedata import load_ohlcv

OUT = Path(__file__).parent
STOP_REASON = "trailing_stop_loss"
BARS_PER_HOUR = 4
DEFAULT_EXTENSIONS = (1, 3, 6, 12, 24)


def annotate(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["age_h"] = (t.close_date - t.open_date).dt.total_seconds() / 3600.0
    t["pnl"] = t.profit_ratio * 100
    t["initial_stop_pct"] = (t.initial_stop_loss_abs / t.open_rate - 1) * 100
    t["final_stop_pct"] = (t.stop_loss_abs / t.open_rate - 1) * 100
    t["stop_above_entry"] = t.final_stop_pct > 0
    t["stop_above_exit"] = t.stop_loss_abs > t.close_rate
    t["ended_in_profit"] = t.profit_ratio > 0
    t["p"] = t.pair.str.split("/").str[0]
    return t


def exit_mix(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (w, r), g in t.groupby(["window", "exit_reason"]):
        rows.append({
            "window": w, "exit_reason": r, "n": len(g),
            "share": round(len(g) / (t.window == w).sum(), 4),
            "mean_pct": round(g.pnl.mean(), 3),
            "median_pct": round(g.pnl.median(), 3),
            "total_pct": round(g.pnl.sum(), 3),
            "median_dur_h": round(g.age_h.median(), 2),
            "win_rate": round((g.pnl > 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows).sort_values(["window", "n"], ascending=[True, False])


def counterfactual(stopped: pd.DataFrame, extensions) -> pd.DataFrame:
    """Replay what price did after each stop, from the actual exit price.

    Excursions are measured FROM THE EXIT PRICE, so this answers "should this
    trade have been held?" rather than a different question about the entry.
    """
    rows, cache = [], {}
    tp_level = BASELINE.take_profit * 100
    roi_level = BASELINE.roi * 100

    for _, tr in stopped.iterrows():
        if tr.p not in cache:
            cache[tr.p] = load_ohlcv(tr.p)
        px = cache[tr.p]
        after = px.loc[px.index > tr.close_date]
        for h in extensions:
            need = int(h * BARS_PER_HOUR)
            seg = after.iloc[:need]
            enough = len(seg) >= need
            if not enough or seg.empty:
                rows.append({"pair": tr.pair, "open_date": tr.open_date,
                             "window": tr.window, "ended_in_profit": tr.ended_in_profit,
                             "extension_h": h, "best_pct": np.nan, "worst_pct": np.nan,
                             "would_reach_take_profit": False, "would_reach_roi": False,
                             "sufficient_history": False})
                continue
            best = (seg.high.max() / tr.close_rate - 1) * 100
            worst = (seg.low.min() / tr.close_rate - 1) * 100
            # Would the ORIGINAL trade have reached its targets if held?
            gain_from_entry = (seg.high.max() / tr.open_rate - 1) * 100
            rows.append({
                "pair": tr.pair, "open_date": tr.open_date, "window": tr.window,
                "ended_in_profit": tr.ended_in_profit, "extension_h": h,
                "best_pct": round(best, 3), "worst_pct": round(worst, 3),
                "would_reach_take_profit": bool(gain_from_entry >= tp_level),
                "would_reach_roi": bool(gain_from_entry >= roi_level),
                "sufficient_history": True,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extensions", default=",".join(map(str, DEFAULT_EXTENSIONS)))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    exts = [float(x) for x in args.extensions.split(",")]

    counts = verify()
    t = annotate(load_trades())
    print(f"baseline verified: {counts}, {len(t)} trades\n")

    # --- T005: exit mix + stop attribution -------------------------------
    mix = exit_mix(t)
    mix.to_csv(args.out / "exit_mix.csv", index=False)

    # FR-005 / research R1: the initial stop must be the static -5%, not the
    # -0.27 grace level. The tolerance only has to SEPARATE THE TWO HYPOTHESES,
    # which are 22pp apart - it is not a precision check. Price-tick rounding on
    # low-precision instruments moves the realised stop by up to ~0.2pp (worst
    # observed: ZEC at open_rate 29.0 with stop 27.6 = -4.83%), so a tolerance
    # tighter than that fails on arithmetic rather than on mechanism.
    STATIC_STOP_PCT, GRACE_STOP_PCT, TOL_PP = -5.0, -27.0, 1.0
    bad = t[(t.initial_stop_pct - STATIC_STOP_PCT).abs() > TOL_PP]
    if len(bad):
        print(f"*** {len(bad)} trades have an initial stop more than {TOL_PP}pp "
              f"from {STATIC_STOP_PCT}% (median {bad.initial_stop_pct.median():.2f}%) ***")
        print("The grace level became reachable; research R1 is invalidated. Halting.")
        return 1
    print(f"initial stop is within {TOL_PP}pp of {STATIC_STOP_PCT}% on all {len(t)} "
          f"trades (observed {t.initial_stop_pct.min():.2f}% to "
          f"{t.initial_stop_pct.max():.2f}%, nearest approach to the "
          f"{GRACE_STOP_PCT}% grace level is "
          f"{(t.initial_stop_pct - GRACE_STOP_PCT).abs().min():.1f}pp)")
    print("-> the grace level is still unreachable; research R1 holds\n")

    stopped = t[t.exit_reason == STOP_REASON]
    stopped.drop(columns=["p"]).to_csv(args.out / "stopped_trades.csv", index=False)

    print("EXIT MIX, pooled:")
    pooled = (t.groupby("exit_reason")
                .agg(n=("pnl", "size"), mean=("pnl", "mean"), median=("pnl", "median"),
                     total=("pnl", "sum"), dur=("age_h", "median"),
                     win=("pnl", lambda s: (s > 0).mean() * 100))
                .round(2).sort_values("n", ascending=False))
    print(pooled.to_string())

    # --- T007: the two populations ---------------------------------------
    print(f"\nSTOPPED TRADES: {len(stopped)} "
          f"({len(stopped)/len(t)*100:.1f}% of all exits)")
    for label, g in (("ended in PROFIT", stopped[stopped.ended_in_profit]),
                     ("ended at a LOSS", stopped[~stopped.ended_in_profit])):
        print(f"  {label:<18} n={len(g):<4} ({len(g)/len(stopped)*100:.1f}%)  "
              f"mean {g.pnl.mean():+.2f}%  median age {g.age_h.median():.2f}h  "
              f"stop above entry {g.stop_above_entry.mean()*100:.1f}%")

    # --- T006: counterfactual --------------------------------------------
    cf = counterfactual(stopped, exts)
    cf.to_csv(args.out / "counterfactual.csv", index=False)

    print(f"\nCOUNTERFACTUAL - what price did after the stop, by population:")
    hdr = f"  {'ext':>5}{'n':>6}{'excl':>6}" + "".join(
        f"{c:>12}" for c in ("best%", "worst%", "reach TP%", "reach ROI%"))
    for label, mask in (("ended in PROFIT", cf.ended_in_profit),
                        ("ended at a LOSS", ~cf.ended_in_profit)):
        print(f"\n  -- {label} --")
        print(hdr)
        for h in exts:
            s = cf[(cf.extension_h == h) & mask]
            ok = s[s.sufficient_history]
            excl = len(s) - len(ok)
            if ok.empty:
                print(f"  {h:>5}{0:>6}{excl:>6}   (no qualifying trades)")
                continue
            print(f"  {h:>5}{len(ok):>6}{excl:>6}"
                  f"{ok.best_pct.median():>12.2f}{ok.worst_pct.median():>12.2f}"
                  f"{ok.would_reach_take_profit.mean()*100:>12.1f}"
                  f"{ok.would_reach_roi.mean()*100:>12.1f}")

    # --- T008: the verdict ------------------------------------------------
    long = cf[(cf.extension_h == max(exts)) & cf.sufficient_history]
    recovered = long.would_reach_roi.mean() * 100
    med_best = long.best_pct.median()
    med_worst = long.worst_pct.median()
    print("\n" + "=" * 68)
    print(f"VERDICT INPUT (at the {max(exts):.0f}h extension, n={len(long)}):")
    print(f"  median best after the stop  : {med_best:+.2f}%")
    print(f"  median worst after the stop : {med_worst:+.2f}%")
    print(f"  would have reached ROI      : {recovered:.1f}%")
    verdict = ("CUTTING NOISE - a large share of stopped trades recover to the "
               "ROI level if held" if recovered >= 50 else
               "CUTTING LOSERS - most stopped trades do NOT recover to the ROI "
               "level; the stop is doing its job")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nwrote exit_mix.csv, stopped_trades.csv, counterfactual.csv to {args.out}")
    print("no model was loaded; nothing trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
