#!/usr/bin/env python3
"""refresh_funding.py — keep FundingCarry's funding feathers fresh for LIVE/dry-run.

FundingCarry re-reads user_data/data/funding/<BASE>_funding.feather every candle.
The historical feathers came from Binance Data Vision (static monthly dumps that
never update live). This sidecar appends the latest SETTLED funding from OKX — the
only funding source reachable from a US IP (Binance-global/Bybit are geo-blocked) —
so the strategy's trailing funding z-score tracks current crowding.

Source note: history is Binance funding, live-appended data is OKX funding. They are
highly correlated (arb-linked); after ~30 days of live running the trailing z-window
(90 settlements) is all-OKX, so the one-time Binance->OKX seam is negligible.

Run every ~1h via cron (funding settles every 8h; hourly is a safe cadence):
    */17 * * * *  cd /path/to/freqtrade && .venv/bin/python user_data/strategies/scripts/refresh_funding.py >> /path/to/refresh_funding.log 2>&1

Idempotent (dedupes by settlement hour), atomic write (temp+rename, never corrupts a
feather), fails safe per-pair (one pair's error doesn't abort the rest; exit code 1
only if EVERY pair failed, so cron alerting can catch a total outage).
"""
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import ccxt

FUND_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "funding"
LOOKBACK_MS = 14 * 24 * 3600 * 1000   # refetch last ~14d each run (cheap overlap buffer)
QUOTE = "USDT"


def okx_client():
    return ccxt.okx({"enableRateLimit": True})


def _fetch_page(ex, sym, since):
    """OKX intermittently throws a ccxt-internal sort TypeError on market load;
    retry a few times before giving up (observed: fails once, succeeds next)."""
    last = None
    for attempt in range(4):
        try:
            return ex.fetch_funding_rate_history(sym, since=since, limit=100)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def fetch_okx_funding(ex, base: str, since_ms: int) -> pd.DataFrame:
    """Fetch settled OKX funding for BASE/QUOTE:QUOTE from since_ms to now."""
    sym = f"{base}/{QUOTE}:{QUOTE}"
    rows, since = [], since_ms
    now = ex.milliseconds()
    for _ in range(20):  # generous page cap; 14d of 8h funding = ~42 rows
        batch = _fetch_page(ex, sym, since)
        if not batch:
            break
        rows += batch
        since = batch[-1]["timestamp"] + 1
        if since >= now or len(batch) < 100:
            break
    if not rows:
        return pd.DataFrame(columns=["dt", "funding"])
    df = pd.DataFrame(
        [{"ts": r["timestamp"], "funding": float(r["fundingRate"])} for r in rows]
    )
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["dt", "funding"]]


def refresh_pair(ex, fpath: Path) -> tuple[bool, str]:
    base = fpath.name.replace("_funding.feather", "")
    try:
        existing = pd.read_feather(fpath)
        existing["dt"] = pd.to_datetime(existing["dt"], utc=True)
    except Exception:
        existing = pd.DataFrame(columns=["dt", "funding"])

    if len(existing):
        last_ms = int(existing["dt"].max().timestamp() * 1000)
        since = last_ms - LOOKBACK_MS
    else:
        since = ex.milliseconds() - 120 * 24 * 3600 * 1000  # cold start: ~120d

    new = fetch_okx_funding(ex, base, since)
    if new.empty and existing.empty:
        return False, f"{base}: no data (existing empty + OKX returned nothing)"

    combined = pd.concat([existing[["dt", "funding"]], new], ignore_index=True)
    combined["dt"] = pd.to_datetime(combined["dt"], utc=True)
    # dedupe by settlement HOUR (Binance ms-offset vs OKX .000); keep newest (OKX on overlap)
    combined["_key"] = combined["dt"].dt.floor("1h")
    combined = (
        combined.sort_values("dt")
        .drop_duplicates("_key", keep="last")
        .drop(columns="_key")
        .sort_values("dt")
        .reset_index(drop=True)
    )
    added = len(combined) - len(existing)

    # atomic write: temp then os.replace (same dir => atomic rename)
    tmp = fpath.with_suffix(".feather.tmp")
    combined.to_feather(tmp)
    os.replace(tmp, fpath)
    latest = str(combined["dt"].iloc[-1])[:16] if len(combined) else "n/a"
    return True, f"{base}: +{added} rows (total {len(combined)}), latest {latest}"


def main() -> int:
    if not FUND_DIR.exists():
        print(f"[refresh_funding] FUND_DIR not found: {FUND_DIR}", file=sys.stderr)
        return 1
    feathers = sorted(FUND_DIR.glob("*_funding.feather"))
    if not feathers:
        print(f"[refresh_funding] no funding feathers in {FUND_DIR}", file=sys.stderr)
        return 1
    ex = okx_client()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    ok = 0
    for f in feathers:
        try:
            good, msg = refresh_pair(ex, f)
            ok += int(good)
            print(f"[refresh_funding {stamp}] {msg}")
        except Exception as e:
            print(f"[refresh_funding {stamp}] {f.stem}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    print(f"[refresh_funding {stamp}] done: {ok}/{len(feathers)} pairs updated")
    return 0 if ok > 0 else 1   # non-zero only on total failure (cron alerting)


if __name__ == "__main__":
    sys.exit(main())
