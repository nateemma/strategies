"""Prerequisite check for the label-free direct-policy feature (T003).

Asserts every input the feature depends on is present *before* any modelling, and
records the per-pair data start dates so the distant-window universe shortfall is a
measured input rather than an assumption (research.md R10).

Run:  python -m research.labelfree.preflight
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # user_data/strategies
DATA = ROOT.parent / "data" / "binanceus"           # user_data/data/binanceus
WF_CONFIG = ROOT / "config" / "config_wf.json"
CONTROL = ROOT / "regime" / "wf" / "pooled_nnnc.json"
SAVED = ROOT / "saved_data"

# A 730-day training window before W3 (2023-01-01) needs history from here.
W3_TRAIN_START = "2021-01-01"
W3_TRAIN_END = "2022-12-31"
MIN_TRAIN_DAYS_IN_RANGE = 180
BARS_PER_DAY = 96
EXPECTED_FOLDS = 15


def _load_jsonc(path: Path) -> dict:
    """freqtrade configs allow // comments; json.loads does not."""
    return json.loads(re.sub(r"//.*", "", path.read_text()))


def check() -> tuple[bool, dict]:
    report: dict = {}
    ok = True

    cfg = _load_jsonc(WF_CONFIG)
    pairs = cfg["exchange"]["pair_whitelist"]
    report["portfolio"] = {
        "max_open_trades": cfg["max_open_trades"],
        "stake_amount": cfg["stake_amount"],
        "tradable_balance_ratio": cfg["tradable_balance_ratio"],
        "dry_run_wallet": cfg["dry_run_wallet"],
        "timeframe": cfg["timeframe"],
        "n_pairs": len(pairs),
    }

    # A start date before the range does NOT mean the pair has data in it.
    # Measured 2026-09-10: XRP/USDT starts 2021-01-01 but carries a 911-day gap
    # (Binance.US delisting, Dec 2020 -> Jul 2023), leaving 12 days inside the W3
    # training range. Count bars in range; a first-bar test calls that "covered".
    lo = pd.Timestamp(W3_TRAIN_START, tz="UTC")
    hi = pd.Timestamp(W3_TRAIN_END, tz="UTC")
    starts: dict[str, str | None] = {}
    cov: dict[str, float] = {}
    for p in pairs:
        f = DATA / f"{p.replace('/', '_')}-15m.feather"
        if not f.exists():
            starts[p], cov[p] = None, 0.0
            ok = False
            continue
        d = pd.read_feather(f, columns=["date"])
        idx = pd.to_datetime(d["date"], utc=True)
        starts[p] = str(idx.iloc[0])[:10]
        cov[p] = float(((idx >= lo) & (idx <= hi)).sum()) / BARS_PER_DAY
    report["data_starts"] = starts
    report["w3_coverage_days"] = {k: round(v) for k, v in sorted(cov.items(), key=lambda kv: -kv[1])}

    trainable = [p for p, d in cov.items() if d >= MIN_TRAIN_DAYS_IN_RANGE]
    report["w3_universe"] = {
        "trainable": sorted(trainable),
        "excluded": sorted(p for p in pairs if p not in trainable),
        "n_covered": len(trainable),
        "n_total": len(pairs),
        "rule": f">= {MIN_TRAIN_DAYS_IN_RANGE} days of actual data inside the training range",
    }

    scalers = {n: (SAVED / n).exists() for n in ("main_scaler.pkl", "main_tensor_scaler.pkl")}
    report["scalers"] = scalers
    if not all(scalers.values()):
        ok = False

    if CONTROL.exists():
        ctl = json.loads(CONTROL.read_text())
        n = len(ctl.get("per_fold", []))
        report["control"] = {
            "per_fold": n,
            "n_trades_pooled": ctl.get("n_trades_pooled"),
            "folds_ok": n == EXPECTED_FOLDS,
        }
        if n != EXPECTED_FOLDS:
            ok = False
        # Screening fold: most control trades, NOT highest return (spec.md, analyze.md F1).
        by_trades = sorted(ctl["per_fold"], key=lambda f: -f.get("n_trades", 0))
        report["screening_fold"] = {
            "chosen": by_trades[0]["fold_id"],
            "n_trades": by_trades[0]["n_trades"],
            "return_pct": by_trades[0].get("return_pct"),
            "runner_up": {k: by_trades[1][k] for k in ("fold_id", "n_trades", "return_pct")},
        }
    else:
        report["control"] = None
        ok = False

    return ok, report


def main() -> int:
    ok, report = check()
    print(json.dumps(report, indent=2))
    w3 = report["w3_universe"]
    print(f"\nW3 training universe: {w3['n_covered']}/{w3['n_total']} pairs"
          f"  excluded={w3['excluded']}")
    sf = report.get("screening_fold")
    if sf:
        print(f"Screening fold: {sf['chosen']} ({sf['n_trades']} trades, "
              f"{sf['return_pct']}%); runner-up {sf['runner_up']}")
    print("PREFLIGHT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
