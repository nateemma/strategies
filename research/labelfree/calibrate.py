"""Blocking calibration against the arbiter (T017, T018).

SC-001 passes only when ALL THREE hold:
  * every matched trade's profit within 1 basis point
  * total return within 0.10 percentage points
  * trade counts identical

Principle VI (diagnose before tuning): a bare "totals differ" is not a diagnosis. Any
residual is attributed to fee application, fill timing or stake sizing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from regime import regimedata as rd                                    # noqa: E402
from research.labelfree import archive, sim                            # noqa: E402
from research.labelfree.costs import BPS, CostModel                    # noqa: E402

TOL_PER_TRADE_BPS = 1.0
TOL_TOTAL_PP = 0.10


def calibrate(zip_path: Path, strategy: str, window: str,
              portfolio: sim.Portfolio | None = None) -> dict:
    portfolio = portfolio or sim.Portfolio()
    tr = archive.load_trades(zip_path, strategy)
    lo, hi = rd.WINDOWS[window]
    lo = pd.Timestamp(lo, tz="UTC")
    hi = pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1)
    tr = tr[(tr["open_date"] >= lo) & (tr["close_date"] < hi)].reset_index(drop=True)

    ohlcv = {p: rd.load_ohlcv(p.split("/")[0]) for p in tr["pair"].unique()}
    cm = CostModel()

    run = sim.replay(tr, ohlcv, cm, portfolio)
    got = run.to_frame()

    merged = tr.merge(got, on=["pair", "open_date", "close_date"],
                      suffixes=("_arb", "_sim"), how="inner")
    notional = merged["amount_arb"] * merged["open_rate_arb"]
    diff_bps = ((merged["profit_abs_sim"] - merged["profit_abs_arb"]).abs()
                / notional.replace(0, np.nan)) / BPS

    arb_total, sim_total = tr["profit_abs"].sum(), got["profit_abs"].sum()
    pp = lambda v: 100.0 * v / portfolio.wallet

    # --- component attribution (Principle VI) --------------------------------
    fill_ok = int(np.isclose(merged["open_rate_sim"], merged["open_rate_arb"],
                             rtol=1e-9).sum())
    close_ok = int(np.isclose(merged["close_rate_sim"], merged["close_rate_arb"],
                              rtol=1e-9).sum())
    by_reason = (merged.assign(d=diff_bps).groupby("exit_reason_arb")["d"]
                 .agg(["count", "median", "max"]).sort_values("max", ascending=False))

    checks = {
        "counts_equal": len(tr) == len(got) == len(merged),
        "per_trade_within_1bp": bool(np.nanmax(diff_bps.values) <= TOL_PER_TRADE_BPS),
        "total_within_0.10pp": bool(abs(pp(sim_total - arb_total)) <= TOL_TOTAL_PP),
    }
    return {
        "window": window, "strategy": strategy, "archive": zip_path.name,
        "n_trades_arbiter": int(len(tr)), "n_trades_sim": int(len(got)),
        "n_matched": int(len(merged)),
        "total_profit_arbiter": float(arb_total), "total_profit_sim": float(sim_total),
        "total_return_pct_arbiter": float(pp(arb_total)),
        "total_return_pct_sim": float(pp(sim_total)),
        "total_diff_pp": float(pp(sim_total - arb_total)),
        "max_per_trade_diff_bps": float(np.nanmax(diff_bps.values)),
        "median_per_trade_diff_bps": float(np.nanmedian(diff_bps.values)),
        "attribution": {
            "entry_fill_matches_bar_open": f"{fill_ok}/{len(merged)}",
            "exit_fill_matches_bar_open": f"{close_ok}/{len(merged)}",
            "worst_by_exit_reason": json.loads(by_reason.to_json(orient="index")),
        },
        "diagnostics": run.diagnostics,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="NNNC_MLX")
    ap.add_argument("--window", default="W1")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()
    z = Path(args.zip) if args.zip else archive.latest_zip(args.strategy)
    rep = calibrate(z, args.strategy, args.window)
    rep["attempt_number"] = args.attempt
    print(json.dumps(rep, indent=2, default=str))
    print("\nCALIBRATION:", rep["verdict"].upper())
    return 0 if rep["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
