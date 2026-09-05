"""Per-trade forensics over freqtrade backtest archives (feature 004, T005-T010).

READ-ONLY. Never runs a backtest, never writes to regime/arms/. Everything it
reports comes from `user_data/backtest_results/<name>.zip`, which already holds
a complete per-trade list — so the whole US1 diagnosis needs no new measurement
(research.md R1).

The archive carries, per trade: exit_reason, profit_abs, profit_ratio,
trade_duration, fee_open, fee_close, stake_amount, pair, open_rate, close_rate
and -- decisively -- max_rate and min_rate, which are the realised favourable
and adverse excursions. Stop-before-payoff and truncation therefore need no
path recomputation from candles.

Contract: specs/004-oracle-label-exit-alignment/contracts/forensics-cli.md
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BT_RESULTS = ROOT / "user_data" / "backtest_results"

# --------------------------------------------------------------------------
# R2 MECHANISM MAPPING -- mandatory.
#
# `trailing_stop = False` framework-wide (Framework/BaseStrategy.py), yet
# `trailing_stop_loss` is the most common exit reason in these archives. It is
# the ATR ADAPTIVE STOP after it moved: freqtrade reports `stop_loss` when the
# INITIAL stop is hit and `trailing_stop_loss` when a stop that has since moved
# is hit, and `use_atr_adaptive_stoploss` is True for every NN strategy. Both
# are the same mechanism; neither is the native trailing stop.
#
# Reporting the raw label would attribute ~40% of trades to a mechanism that is
# switched off, and would invite "fixing" trailing_stop_positive, already on
# record as inert.
# --------------------------------------------------------------------------
MECHANISM = {
    "stop_loss":           "ATR stop (initial)",
    "trailing_stop_loss":  "ATR stop (moved)",
    "take_profit":         "cexit_take_profit",
    "roi":                 "ROI table",
    "unclog_12h":          "hold window (unclog_12h)",
    "unclog_1d":           "hold window (unclog_1d)",
    "max_hold":            "hold window (max_hold)",
    "metric_overbought":   "profit check (metric_overbought)",
    "rsi_strong_sell":     "profit check (rsi)",
    "rsi_moderate_sell":   "profit check (rsi)",
    "rsi_conservative_sell": "profit check (rsi)",
    "force_exit":          "end of backtest",
    "exit_signal":         "model exit signal",
}
# counted in coverage, excluded from mechanism attribution
NON_MECHANISM = {"end of backtest"}


def mech(reason: str) -> str:
    return MECHANISM.get(reason, f"UNMAPPED:{reason}")


def load(archive: Path, strategy: str | None):
    z = zipfile.ZipFile(archive)
    main = [n for n in z.namelist()
            if n.endswith(".json") and "config" not in n and "market_change" not in n]
    if not main:
        raise SystemExit(f"no result json inside {archive}")
    d = json.loads(z.read(main[0]))
    strats = d.get("strategy", {})
    if strategy and strategy in strats:
        name = strategy
    elif len(strats) == 1:
        name = next(iter(strats))
    else:
        raise SystemExit(f"{archive} holds {list(strats)}; pass --strategy")
    return name, strats[name]


def resolve(args) -> Path:
    if args.archive:
        return Path(args.archive)
    if args.latest:
        marker = BT_RESULTS / ".last_result.json"
        return BT_RESULTS / json.loads(marker.read_text())["latest_backtest"]
    # --match-trades: newest archive whose trade count (and strategy) match
    for p in sorted(BT_RESULTS.glob("*.zip"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            name, s = load(p, args.strategy)
        except Exception:
            continue
        if len(s.get("trades", [])) == args.match_trades:
            if args.strategy is None or name == args.strategy:
                return p
    raise SystemExit(f"no archive with {args.match_trades} trades"
                     + (f" for {args.strategy}" if args.strategy else ""))


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def sec_exits(trades):
    agg = defaultdict(lambda: {"n": 0, "pnl": 0.0, "dur": 0.0})
    for t in trades:
        a = agg[mech(t["exit_reason"])]
        a["n"] += 1
        a["pnl"] += t["profit_abs"]
        a["dur"] += t["trade_duration"]
    total = sum(a["pnl"] for a in agg.values())
    rows = []
    for k, a in sorted(agg.items(), key=lambda kv: kv[1]["pnl"]):
        rows.append({
            "mechanism": k, "n_trades": a["n"], "sum_pnl": a["pnl"],
            "mean_pnl": a["pnl"] / a["n"], "mean_duration_h": a["dur"] / a["n"] / 60.0,
            "share_of_total_pnl": (100.0 * a["pnl"] / abs(total)) if total else 0.0,
        })
    return rows, total


def sec_costs(trades):
    fees = 0.0
    for t in trades:
        entry = t["stake_amount"]
        exit_val = entry * (1.0 + t["profit_ratio"])
        fees += t["fee_open"] * entry + t["fee_close"] * exit_val
    net = sum(t["profit_abs"] for t in trades)
    gross = net + fees
    return {
        "n_trades": len(trades), "total_fees": fees, "net_pnl": net, "gross_pnl": gross,
        "fees_as_share_of_gross": (100.0 * fees / abs(gross)) if gross else float("inf"),
        "loss_survives_costs": gross < 0,
    }


def sec_pairs(trades):
    agg = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        a = agg[t["pair"]]; a["n"] += 1; a["pnl"] += t["profit_abs"]
    n = len(trades)
    return sorted(({"pair": k, "n_trades": v["n"], "sum_pnl": v["pnl"],
                    "share_pct": 100.0 * v["n"] / n} for k, v in agg.items()),
                  key=lambda r: -r["n_trades"])


def sec_excursions(trades, stop_pct, tp_pct):
    """From max_rate/min_rate, normalised to open_rate. No candle recomputation."""
    mfe, mae, stopped_first, truncated = [], [], 0, 0
    for t in trades:
        o = t["open_rate"]
        if not o:
            continue
        up = (t["max_rate"] - o) / o * 100.0
        dn = (t["min_rate"] - o) / o * 100.0
        mfe.append(up); mae.append(dn)
        # adverse excursion reached the operative stop
        if dn <= stop_pct:
            stopped_first += 1
        # favourable excursion ran past the take-profit level
        if up > tp_pct:
            truncated += 1
    n = len(mfe) or 1
    q = lambda xs, p: sorted(xs)[max(0, min(len(xs) - 1, int(p * len(xs))))]
    return {
        "n": len(mfe),
        "mfe_p50": q(mfe, .50), "mfe_p90": q(mfe, .90),
        "mae_p50": q(mae, .50), "mae_p10": q(mae, .10),
        "stop_pct_used": stop_pct, "tp_pct_used": tp_pct,
        "reached_stop_pct": 100.0 * stopped_first / n,
        "exceeded_tp_pct": 100.0 * truncated / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive")
    g.add_argument("--latest", action="store_true")
    g.add_argument("--match-trades", type=int)
    ap.add_argument("--strategy")
    ap.add_argument("--section", choices=["exits", "costs", "pairs", "excursions", "all"],
                    default="all")
    ap.add_argument("--stop-pct", type=float, default=-5.0,
                    help="operative ATR stop band midpoint (NNNC: -6%%..-4%%)")
    ap.add_argument("--tp-pct", type=float, default=2.4,
                    help="cexit_take_profit as a percentage (NNNC: 0.024)")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    archive = resolve(args)
    name, s = load(archive, args.strategy)
    trades = s["trades"]

    # R5: always print what was actually read, so a mis-resolved archive is visible
    print(f"ARCHIVE   {archive.name}")
    print(f"STRATEGY  {name}")
    print(f"TRADES    {len(trades)}")
    print("\nMECHANISM MAPPING (R2) -- trailing_stop is FALSE framework-wide, so")
    print("`trailing_stop_loss` is the ATR ADAPTIVE STOP after it moved, NOT a")
    print("native trailing stop. Both stop labels are the same ATR band.")
    for k, v in sorted(MECHANISM.items()):
        print(f"    {k:<24} -> {v}")

    out = {"archive": archive.name, "strategy": name, "n_trades": len(trades)}

    if args.section in ("exits", "all"):
        rows, total = sec_exits(trades)
        attributed = sum(r["n_trades"] for r in rows)
        print(f"\n== EXITS ==   coverage {attributed}/{len(trades)} "
              f"({100.0*attributed/len(trades):.1f}%)")
        print(f"{'mechanism':<34}{'n':>6}{'sum P&L':>12}{'mean':>9}{'mean h':>9}{'% P&L':>9}")
        for r in rows:
            print(f"{r['mechanism']:<34}{r['n_trades']:>6}{r['sum_pnl']:>12.1f}"
                  f"{r['mean_pnl']:>9.2f}{r['mean_duration_h']:>9.1f}"
                  f"{r['share_of_total_pnl']:>8.1f}%")
        print(f"{'TOTAL':<34}{len(trades):>6}{total:>12.1f}")
        out["exits"] = rows; out["total_pnl"] = total
        out["coverage_pct"] = 100.0 * attributed / len(trades)

    if args.section in ("costs", "all"):
        c = sec_costs(trades)
        print(f"\n== COSTS ==")
        print(f"    trades              {c['n_trades']}")
        print(f"    total fees          {c['total_fees']:.1f}")
        print(f"    net P&L             {c['net_pnl']:.1f}")
        print(f"    GROSS P&L (net+fee) {c['gross_pnl']:.1f}")
        print(f"    fees / |gross|      {c['fees_as_share_of_gross']:.1f}%")
        print(f"    loss survives costs {c['loss_survives_costs']}")
        out["costs"] = c

    if args.section in ("pairs", "all"):
        rows = sec_pairs(trades)
        print(f"\n== PAIRS ==   {len(rows)} instruments")
        print(f"{'pair':<16}{'n':>6}{'share':>8}{'sum P&L':>12}")
        for r in rows[:15]:
            print(f"{r['pair']:<16}{r['n_trades']:>6}{r['share_pct']:>7.1f}%{r['sum_pnl']:>12.1f}")
        out["pairs"] = rows

    if args.section in ("excursions", "all"):
        e = sec_excursions(trades, args.stop_pct, args.tp_pct)
        print(f"\n== EXCURSIONS ==  from max_rate/min_rate, no candle recomputation")
        print(f"    favourable  p50 {e['mfe_p50']:+.2f}%   p90 {e['mfe_p90']:+.2f}%")
        print(f"    adverse     p50 {e['mae_p50']:+.2f}%   p10 {e['mae_p10']:+.2f}%")
        print(f"    reached the {e['stop_pct_used']}% stop        {e['reached_stop_pct']:.1f}% of trades")
        print(f"    exceeded the +{e['tp_pct_used']}% take-profit  {e['exceeded_tp_pct']:.1f}% of trades")
        out["excursions"] = e

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
