"""036 T014: daily parity of the live bot against a backtest of the same model (research R4).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.parity --day 2026-10-17 [--no-download]

The tool:
1. downloads the latest 15m candles for the whitelist, unless `--no-download`. It refuses while a
   training or backtest chain is running, because they read the same files.
2. backtests the live bot class CUMULATIVELY, from the live run's `started` day to the end of
   `--day` (strict model mode). Positions held over several days (24 h+ grace) and one trade per
   pair mean a single-day backtest would start from the wrong open-trade state.
3. gets the backtest's p_buy, close and volume per bar from the strategy's own forward pass
   (`research.nnnc_framework.signals.fold_table`).
4. reads the live trades from the bot's sqlite (read-only) and the recorder's signal and order
   logs.
5. matches the DAY's entries on (pair, entry candle) and its exits on (pair, exit candle).
6. classifies each difference by the declared order:

   | order | cause | test |
   |---|---|---|
   | 1 | prediction_difference | live and backtest p_buy differ by more than 0.01 at the signal candle |
   | 2 | data_revision | live and backtest close differ |
   | 3 | gap_fill | zero volume on either side |
   | 4 | timing | the other side entered the pair within one candle, or a live order was not filled |
   | 5 | wallet_state | the other side held the pair at that time, or had all slots full |
   | 6 | UNEXPLAINED | none of the above |

Writes research/live_pnl/parity/<day>.json (data-model ParityReport).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
ROOT = STRAT.parents[1]
for p in (STRAT, STRAT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
OUT = HERE / "parity"
REC_ROOT = STRAT.parent / "data" / "live_nnnc"
DB = ROOT / "tradesv3.live_nnnc.sqlite"
CONFIG = "user_data/strategies/config/config_live_nnnc.json"
BT_DIR = ROOT / "user_data" / "backtest_results" / "live_nnnc"
P_TOL = 0.01
CANDLE = pd.Timedelta(minutes=15)
CAUSES = ("prediction_difference", "data_revision", "gap_fill", "timing", "wallet_state",
          "UNEXPLAINED")


# ------------------------------------------------------------------ pure classification

def _entry_candle(t) -> pd.Timestamp:
    """The SIGNAL candle of an entry: the candle before the one the trade opened in."""
    return pd.Timestamp(t).tz_convert("UTC").floor("15min") - CANDLE


def _held(trades: pd.DataFrame, pair: str, t: pd.Timestamp) -> bool:
    if trades.empty:
        return False
    m = trades[(trades.pair == pair) & (trades.open_date <= t)
               & (trades.close_date.isna() | (trades.close_date > t))]
    return len(m) > 0


def _slots_full(trades: pd.DataFrame, t: pd.Timestamp, max_open: int) -> bool:
    if trades.empty:
        return False
    m = trades[(trades.open_date <= t) & (trades.close_date.isna() | (trades.close_date > t))]
    return len(m) >= max_open


def classify(pair: str, signal_candle: pd.Timestamp, side: str, *, live_sig: pd.DataFrame,
             bt_sig: pd.DataFrame, live_trades: pd.DataFrame, bt_trades: pd.DataFrame,
             unfilled: set, max_open: int) -> tuple[str, dict]:
    """Cause of one difference. `side` is 'live_only' or 'backtest_only'."""
    ls = live_sig[(live_sig.pair == pair) & (live_sig.candle == signal_candle)]
    bs = bt_sig[(bt_sig.pair == pair) & (bt_sig.date == signal_candle)]
    ev = {}
    if len(ls) and len(bs):
        lp, bp = ls.p_buy.iloc[0], bs.p_buy.iloc[0]
        ev.update(live_p_buy=lp, bt_p_buy=bp)
        if lp is not None and bp is not None and abs(float(lp) - float(bp)) > P_TOL:
            return "prediction_difference", ev
        lc, bc = float(ls.close.iloc[0]), float(bs.close.iloc[0])
        ev.update(live_close=lc, bt_close=bc)
        if abs(lc - bc) > 1e-9 * max(abs(lc), 1.0):
            return "data_revision", ev
        if float(ls.quote_volume.iloc[0]) == 0 or float(bs.volume.iloc[0]) == 0:
            return "gap_fill", ev
    elif len(bs) and float(bs.volume.iloc[0]) == 0:
        return "gap_fill", ev
    other = bt_trades if side == "live_only" else live_trades
    near = other[(other.pair == pair)] if not other.empty else other
    if len(near) and ((near.open_date.map(_entry_candle) - signal_candle).abs() <= CANDLE).any():
        return "timing", ev
    if side == "backtest_only" and (pair, signal_candle) in unfilled:
        return "timing", {**ev, "live_order_unfilled": True}
    t = signal_candle + CANDLE
    if _held(other, pair, t) or _slots_full(other, t, max_open):
        return "wallet_state", ev
    return "UNEXPLAINED", ev


def compare_day(day: str, *, live_trades: pd.DataFrame, bt_trades: pd.DataFrame,
                live_sig: pd.DataFrame, bt_sig: pd.DataFrame, unfilled: set = frozenset(),
                max_open: int = 10) -> dict:
    """Match the day's entries and exits; classify every difference."""
    d0 = pd.Timestamp(day, tz="UTC")
    d1 = d0 + pd.Timedelta(days=1)

    def entries(t):
        if t.empty:
            return set()
        x = t[(t.open_date >= d0) & (t.open_date < d1)]
        return {(r.pair, _entry_candle(r.open_date)) for r in x.itertuples()}

    def exits(t):
        if t.empty:
            return set()
        x = t[t.close_date.notna() & (t.close_date >= d0) & (t.close_date < d1)]
        return {(r.pair, pd.Timestamp(r.close_date).floor("15min")) for r in x.itertuples()}

    le, be = entries(live_trades), entries(bt_trades)
    lx, bx = exits(live_trades), exits(bt_trades)
    diffs = []
    for side, keys in (("live_only", le - be), ("backtest_only", be - le)):
        for pair, cand in sorted(keys):
            cause, ev = classify(pair, cand, side, live_sig=live_sig, bt_sig=bt_sig,
                                 live_trades=live_trades, bt_trades=bt_trades,
                                 unfilled=unfilled, max_open=max_open)
            diffs.append({"kind": "entry", "side": side, "pair": pair,
                          "signal_candle": cand.isoformat(), "cause": cause, "evidence": ev})
    # an exit difference on a MATCHED entry is reported as timing (same trade, different close)
    for side, keys in (("live_only", lx - bx), ("backtest_only", bx - lx)):
        for pair, cand in sorted(keys):
            diffs.append({"kind": "exit", "side": side, "pair": pair,
                          "candle": cand.isoformat(), "cause": "timing", "evidence": {}})
    return {"day": day, "matched_entries": len(le & be), "matched_exits": len(lx & bx),
            "live_only": sum(d["side"] == "live_only" for d in diffs),
            "backtest_only": sum(d["side"] == "backtest_only" for d in diffs),
            "differences": diffs,
            "unexplained": sum(d["cause"] == "UNEXPLAINED" for d in diffs)}


# ------------------------------------------------------------------ I/O

def _norm_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["pair", "open_date", "close_date"])
    out = df[["pair", "open_date", "close_date"]].copy()
    out["open_date"] = pd.to_datetime(out.open_date, utc=True)
    out["close_date"] = pd.to_datetime(out.close_date, utc=True)
    return out


def live_trades(db: Path = DB) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql("select pair, open_date, close_date from trades", con)
    finally:
        con.close()
    return _norm_trades(df)


def read_jsonl(kind: str, run_id: str, root: Path = REC_ROOT) -> pd.DataFrame:
    rows = []
    for f in sorted((root / run_id).glob(f"{kind}-*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return pd.DataFrame(rows)


def _busy() -> list[str]:
    from research.candidate_training import train as drv

    return drv.other_chains(drv._ps(), os.getpid())


def _download() -> None:
    busy = _busy()
    if busy:
        raise SystemExit("REFUSED download: a training/backtest chain is running")
    subprocess.run([str(ROOT / ".venv/bin/freqtrade"), "download-data", "-c", CONFIG, "-t", "15m",
                    "--days", "3"], cwd=ROOT, check=True, capture_output=True)


def backtest(rec: dict, day: str) -> pd.DataFrame:
    start = pd.Timestamp(rec["started"]).strftime("%Y%m%d")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    BT_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FT_MODEL_EVAL_STRICT": "1"}
    subprocess.run([str(ROOT / ".venv/bin/freqtrade"), "backtesting", "-c", CONFIG,
                    "--strategy-path", "user_data/strategies/NNNC", "-s", rec["bot_class"],
                    "--timerange", f"{start}-{end}", "--export", "trades",
                    "--backtest-directory", str(BT_DIR)],
                   cwd=ROOT, check=True, env=env, capture_output=True)
    last = json.loads((BT_DIR / ".last_result.json").read_text())["latest_backtest"]
    with zipfile.ZipFile(BT_DIR / last) as z:
        name = [n for n in z.namelist() if n.endswith(".json") and "config" not in n
                and "meta" not in n][0]
        strat = list(json.loads(z.read(name))["strategy"].values())[0]
    return _norm_trades(pd.DataFrame(strat["trades"]))


def run(day: str, download: bool = True) -> dict:
    from research.nnnc_framework.signals import fold_table

    recs = json.loads((HERE / "live_models.json").read_text())
    rec = [r for r in recs if r.get("started")][-1]
    if download:
        _download()
    sig = read_jsonl("signals", rec["run_id"])
    if sig.empty or not (sig.candle.str[:10] == day).any():
        raise SystemExit(f"REFUSED: no live signal log for {day}")
    sig["candle"] = pd.to_datetime(sig.candle, utc=True)
    orders = read_jsonl("orders", rec["run_id"])
    fills = read_jsonl("fills", rec["run_id"])
    filled = set() if fills.empty else set(fills.order_id)
    unfilled = set()
    if not orders.empty:
        for o in orders.itertuples():
            if getattr(o, "rejected", False):
                unfilled.add((o.pair, pd.Timestamp(o.signal_candle)))
    d0 = pd.Timestamp(day)
    bt_sig = fold_table(rec["bot_class"], f"{d0:%Y%m%d}-{d0 + pd.Timedelta(days=1):%Y%m%d}")
    bt_sig["date"] = pd.to_datetime(bt_sig.date, utc=True)
    out = compare_day(day, live_trades=live_trades(), bt_trades=backtest(rec, day),
                      live_sig=sig, bt_sig=bt_sig, unfilled=unfilled)
    out.update(run_id=rec["run_id"], backtest_range=f"{rec['started'][:10]}..{day}",
               filled_orders=len(filled), label="036 parity (PROVISIONAL)")
    OUT.mkdir(exist_ok=True)
    (OUT / f"{day}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "differences"}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", required=True)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args(argv)
    run(a.day, download=not a.no_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
