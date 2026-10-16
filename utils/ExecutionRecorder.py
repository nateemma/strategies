"""Execution recorder for live / dry-run bots (feature 036; contracts/execution_recorder.md).

A strategy-agnostic mixin. While a dry-run or live bot runs, it records what execution would have
cost:
- per order: the signal, the requested and granted stake, and a top-N order-book snapshot;
- fills and exits;
- one signal row per pair and closed candle;
- a scheduled order-book snapshot per pair and candle.

Everything is written as append-only daily JSONL under RECORD_DIR/RECORD_RUN_ID/, with depth
gzipped.

* Default OFF (`RECORD_EXECUTION = False`), and inactive outside live/dry_run runmodes. When
  inactive, every override returns exactly what super() returns and does no I/O, so backtests are
  byte-identical.
* It never changes a decision: super() runs first and its value is returned. Recording errors are
  logged and swallowed.
* It does not change the model key: its attributes are in MODEL_KEY_IGNORED, and none of the
  overridden callbacks is weight-shaping.
* It is idempotent across restarts: each daily file's keys are loaded once, and duplicates are
  skipped.
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)

STRAT_DIR = Path(__file__).resolve().parents[1]
DEPTH_TIMEOUT_S = 3.0
CANDLE_SECONDS = 15 * 60


def _candle(t: datetime, seconds: int = CANDLE_SECONDS) -> datetime:
    ts = int(t.timestamp()) // seconds * seconds
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _iso(t) -> str:
    if hasattr(t, "to_pydatetime"):
        t = t.to_pydatetime()
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).isoformat()


class _DailyWriter:
    """Append-only daily files with per-file dedupe keys (loaded once per file)."""

    def __init__(self, root: Path):
        self.root = root
        self._keys: dict[Path, set] = {}

    def _path(self, kind: str, day: str) -> Path:
        ext = ".jsonl.gz" if kind == "depth" else ".jsonl"
        return self.root / f"{kind}-{day}{ext}"

    def _load(self, path: Path) -> set:
        if path not in self._keys:
            keys = set()
            if path.is_file():
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt") as fh:
                    for line in fh:
                        try:
                            keys.add(json.loads(line)["key"])
                        except (ValueError, KeyError):
                            continue
            self._keys[path] = keys
        return self._keys[path]

    def has(self, kind: str, day: str, key: str) -> bool:
        return key in self._load(self._path(kind, day))

    def write(self, kind: str, day: str, key: str, record: dict) -> bool:
        path = self._path(kind, day)
        keys = self._load(path)
        if key in keys:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"key": key, **record}, default=str) + "\n"
        if path.suffix == ".gz":
            with gzip.open(path, "at") as fh:
                fh.write(line)
        else:
            with open(path, "a") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        keys.add(key)
        return True


class ExecutionRecorderMixin:
    RECORD_EXECUTION: bool = False
    RECORD_DIR: str | None = None
    RECORD_RUN_ID: str | None = None
    RECORD_DEPTH_LEVELS: int = 20
    RECORD_PBUY_COLUMN: str = "%predict_prob_buy"
    MODEL_KEY_IGNORED = ("RECORD_EXECUTION", "RECORD_DIR", "RECORD_RUN_ID", "RECORD_DEPTH_LEVELS",
                         "RECORD_PBUY_COLUMN")

    _rec_writer = None
    _rec_pool = None
    _rec_stake: dict | None = None

    # ------------------------------------------------------------------ plumbing

    def _rec_active(self) -> bool:
        if not self.RECORD_EXECUTION:
            return False
        dp = getattr(self, "dp", None)
        return dp is not None and getattr(dp.runmode, "value", None) in ("live", "dry_run")

    def _rec(self) -> _DailyWriter:
        if self._rec_writer is None:
            if not self.RECORD_DIR or not self.RECORD_RUN_ID:
                raise ValueError(f"{type(self).__name__}: RECORD_DIR and RECORD_RUN_ID are required")
            root = Path(self.RECORD_DIR).resolve()
            if root == STRAT_DIR or STRAT_DIR in root.parents:
                raise ValueError(f"{type(self).__name__}: RECORD_DIR {root} is inside the strategies "
                                 "repository; recorded market data must never be committed")
            self._rec_writer = _DailyWriter(root / self.RECORD_RUN_ID)
            self._rec_stake = {}
        return self._rec_writer

    def _rec_safe(self, fn, *args):
        try:
            fn(*args)
        except ValueError:
            raise
        except Exception as exc:  # recording must never affect trading
            logger.warning("ExecutionRecorder: %s failed: %s", fn.__name__, exc)

    def _rec_depth(self, pair: str, t: datetime, trigger: str) -> str:
        """Take and write one snapshot; return its key (also when the snapshot is missing)."""
        key = f"{trigger}|{pair}|{_iso(t)}"
        if self._rec_pool is None:
            type(self)._rec_pool = cf.ThreadPoolExecutor(max_workers=1)
        bids, asks, missing = [], [], False
        try:
            book = self._rec_pool.submit(self.dp.orderbook, pair, self.RECORD_DEPTH_LEVELS) \
                .result(timeout=DEPTH_TIMEOUT_S)
            bids = [[float(p), float(q)] for p, q, *_ in (book.get("bids") or [])][:self.RECORD_DEPTH_LEVELS]
            asks = [[float(p), float(q)] for p, q, *_ in (book.get("asks") or [])][:self.RECORD_DEPTH_LEVELS]
            missing = not bids or not asks
        except Exception as exc:
            logger.warning("ExecutionRecorder: orderbook %s failed: %s", pair, exc)
            missing = True
        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None
        self._rec().write("depth", _iso(t)[:10], key, {
            "run_id": self.RECORD_RUN_ID, "pair": pair, "time": _iso(t), "trigger": trigger,
            "bids": bids, "asks": asks, "mid": mid, "depth_missing": missing})
        return key

    def _rec_last_candle(self, pair: str):
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if df is None or len(df) == 0:
            return None
        return df.iloc[-1]

    # ------------------------------------------------------------------ callbacks

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, *args, **kwargs):
        granted = super().custom_stake_amount(pair, current_time, current_rate, proposed_stake,
                                              *args, **kwargs)
        if self._rec_active():
            self._rec()
            self._rec_stake[pair] = (float(proposed_stake), float(granted))
        return granted

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time,
                            entry_tag, side, **kwargs):
        ok = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force,
                                         current_time, entry_tag, side, **kwargs)
        if self._rec_active():
            self._rec_safe(self._rec_entry, pair, order_type, amount, rate, current_time,
                           entry_tag, ok)
        return ok

    def _rec_entry(self, pair, order_type, amount, rate, current_time, entry_tag, ok):
        w = self._rec()
        last = self._rec_last_candle(pair)
        candle = _iso(last["date"]) if last is not None else _iso(_candle(current_time))
        key = f"entry_intent|{pair}|{candle}"
        day = _iso(current_time)[:10]
        if w.has("orders", day, key):
            return
        requested, granted = (self._rec_stake or {}).pop(pair, (None, None))
        depth_ref = self._rec_depth(pair, current_time, "order")
        w.write("orders", day, key, {
            "run_id": self.RECORD_RUN_ID, "pair": pair, "signal_candle": candle,
            "signal_close": None if last is None else float(last["close"]),
            "signal_quote_volume": None if last is None else float(last["close"] * last["volume"]),
            "requested_stake": requested, "granted_stake": granted,
            "rate": float(rate), "amount": float(amount), "order_type": order_type,
            "order_time": _iso(current_time), "entry_tag": entry_tag, "rejected": not ok,
            "depth_ref": depth_ref})

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        res = super().order_filled(pair, trade, order, current_time, **kwargs)
        if self._rec_active():
            self._rec_safe(self._rec_fill, pair, trade, order, current_time)
        return res

    def _rec_fill(self, pair, trade, order, current_time):
        key = f"fill|{trade.id}|{order.order_id}"
        self._rec().write("fills", _iso(current_time)[:10], key, {
            "run_id": self.RECORD_RUN_ID, "pair": pair, "trade_id": trade.id,
            "order_id": order.order_id, "side": getattr(order, "ft_order_side", None),
            "fill_time": _iso(current_time), "fill_price": getattr(order, "safe_price", None),
            "filled": getattr(order, "safe_filled", None), "status": getattr(order, "status", None)})

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force, exit_reason,
                           current_time, **kwargs):
        ok = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force,
                                        exit_reason, current_time, **kwargs)
        if self._rec_active():
            self._rec_safe(self._rec_exit, pair, trade, rate, exit_reason, current_time, ok)
        return ok

    def _rec_exit(self, pair, trade, rate, exit_reason, current_time, ok):
        key = f"exit|{trade.id}|{_iso(current_time)}"
        day = _iso(current_time)[:10]
        if self._rec().has("exits", day, key):
            return
        depth_ref = self._rec_depth(pair, current_time, "exit")
        self._rec().write("exits", day, key, {
            "run_id": self.RECORD_RUN_ID, "pair": pair, "trade_id": trade.id,
            "exit_reason": exit_reason, "rate": float(rate), "time": _iso(current_time),
            "confirmed": bool(ok), "depth_ref": depth_ref})

    def bot_loop_start(self, current_time, **kwargs):
        res = super().bot_loop_start(current_time, **kwargs)
        if self._rec_active():
            self._rec_safe(self._rec_schedule, current_time)
        return res

    def _rec_schedule(self, current_time):
        candle = _candle(current_time)
        w = self._rec()
        for pair in self.dp.current_whitelist():
            key = f"schedule|{pair}|{_iso(candle)}"
            if not w.has("depth", _iso(candle)[:10], key):
                self._rec_depth(pair, candle, "schedule")

    def populate_entry_trend(self, dataframe, metadata):
        dataframe = super().populate_entry_trend(dataframe, metadata)
        if self._rec_active() and len(dataframe):
            self._rec_safe(self._rec_signal, dataframe, metadata.get("pair", "?"))
        return dataframe

    def _rec_signal(self, dataframe, pair):
        last = dataframe.iloc[-1]
        candle = _iso(last["date"])
        thr = getattr(self, "prediction_threshold", None)
        thr = getattr(thr, "value", thr)
        col = self.RECORD_PBUY_COLUMN
        self._rec().write("signals", candle[:10], f"{pair}|{candle}", {
            "run_id": self.RECORD_RUN_ID, "pair": pair, "candle": candle,
            "close": float(last["close"]), "quote_volume": float(last["close"] * last["volume"]),
            "p_buy": None if col not in dataframe.columns else float(last[col]),
            "threshold": None if thr is None else float(thr),
            "enter_long": int(last.get("enter_long", 0) or 0)})
