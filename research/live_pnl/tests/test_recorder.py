"""036 T003: ExecutionRecorderMixin contract (contracts/execution_recorder.md)."""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[3]
for p in (STRAT, STRAT / "NNNC"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import ExecutionRecorder as ER  # noqa: E402


T0 = datetime(2026, 10, 16, 12, 7, tzinfo=timezone.utc)


class _DP:
    def __init__(self, runmode="dry_run", book=None, raise_book=False):
        self.runmode = SimpleNamespace(value=runmode)
        self._book = book or {"bids": [[99.0, 1.0], [98.0, 2.0]], "asks": [[101.0, 1.0], [102.0, 2.0]]}
        self._raise = raise_book
        self.calls = 0

    def orderbook(self, pair, maximum):
        self.calls += 1
        if self._raise:
            raise RuntimeError("exchange down")
        return self._book

    def current_whitelist(self):
        return ["AAA/USDT", "BBB/USDT"]

    def get_analyzed_dataframe(self, pair, timeframe):
        df = pd.DataFrame({"date": [pd.Timestamp("2026-10-16 11:45", tz="UTC")],
                           "close": [100.0], "volume": [50.0], "%predict_prob_buy": [0.9],
                           "enter_long": [1]})
        return df, None


class _Parent:
    """Stands in for the strategy: fixed return values, so pass-through is checkable."""
    timeframe = "15m"
    prediction_threshold = SimpleNamespace(value=0.77)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, *a, **k):
        return proposed_stake / 2

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time,
                            entry_tag, side, **k):
        return True

    def order_filled(self, pair, trade, order, current_time, **k):
        return None

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force,
                           exit_reason, current_time, **k):
        return True

    def bot_loop_start(self, current_time, **k):
        return None

    def populate_entry_trend(self, dataframe, metadata):
        return dataframe


def _strategy(tmp_path, runmode="dry_run", on=True, **dp_kw):
    cls = type("S", (ER.ExecutionRecorderMixin, _Parent),
               {"RECORD_EXECUTION": on, "RECORD_DIR": str(tmp_path), "RECORD_RUN_ID": "t1"})
    s = cls()
    s.dp = _DP(runmode, **dp_kw)
    return s


def _exercise(s):
    out = [s.custom_stake_amount("AAA/USDT", T0, 100.0, 900.0, 10.0, 9000.0, 1.0, None, "long"),
           s.confirm_trade_entry("AAA/USDT", "limit", 4.5, 100.0, "GTC", T0, None, "long")]
    trade = SimpleNamespace(id=7, pair="AAA/USDT", enter_tag=None)
    order = SimpleNamespace(order_id="o1", ft_order_side="buy", safe_price=100.5, safe_filled=4.5,
                            status="closed")
    out.append(s.order_filled("AAA/USDT", trade, order, T0))
    out.append(s.confirm_trade_exit("AAA/USDT", trade, "market", 4.5, 103.0, "GTC", "roi", T0))
    out.append(s.bot_loop_start(T0))
    df = _DP().get_analyzed_dataframe("AAA/USDT", "15m")[0]
    out.append(len(s.populate_entry_trend(df, {"pair": "AAA/USDT"})))
    return out


def _files(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("runmode,on", [("backtest", True), ("dry_run", False), ("hyperopt", True)])
def test_inactive_is_pure_pass_through(tmp_path, runmode, on):
    s = _strategy(tmp_path, runmode, on)
    assert _exercise(s) == [450.0, True, None, True, None, 1]
    assert _files(tmp_path) == []
    assert s.dp.calls == 0


def test_active_records_and_returns_parent_values(tmp_path):
    s = _strategy(tmp_path)
    assert _exercise(s) == [450.0, True, None, True, None, 1]
    names = _files(tmp_path)
    for kind in ("orders", "fills", "exits", "signals"):
        assert f"{kind}-2026-10-16.jsonl" in names
    assert "depth-2026-10-16.jsonl.gz" in names
    run = tmp_path / "t1"
    order = json.loads((run / "orders-2026-10-16.jsonl").read_text().splitlines()[0])
    assert order["requested_stake"] == 900.0 and order["granted_stake"] == 450.0
    assert order["signal_close"] == 100.0 and order["depth_ref"] is not None
    sig = json.loads((run / "signals-2026-10-16.jsonl").read_text().splitlines()[0])
    assert sig["p_buy"] == 0.9 and sig["threshold"] == 0.77 and sig["enter_long"] == 1
    depth = [json.loads(x) for x in gzip.open(run / "depth-2026-10-16.jsonl.gz", "rt")]
    assert {d["trigger"] for d in depth} == {"order", "exit", "schedule"}
    sched = [d for d in depth if d["trigger"] == "schedule"]
    assert {d["pair"] for d in sched} == {"AAA/USDT", "BBB/USDT"}
    assert sched[0]["mid"] == 100.0


def test_restart_does_not_duplicate(tmp_path):
    _exercise(_strategy(tmp_path))
    _exercise(_strategy(tmp_path))  # a second instance = a restarted bot
    run = tmp_path / "t1"
    for kind in ("orders", "fills", "exits", "signals"):
        assert len((run / f"{kind}-2026-10-16.jsonl").read_text().splitlines()) == 1, kind
    depth = [json.loads(x) for x in gzip.open(run / "depth-2026-10-16.jsonl.gz", "rt")]
    assert sum(d["trigger"] == "schedule" for d in depth) == 2


def test_schedule_is_once_per_pair_per_candle(tmp_path):
    s = _strategy(tmp_path)
    s.bot_loop_start(T0)
    s.bot_loop_start(T0.replace(minute=9))   # same 15m candle
    s.bot_loop_start(T0.replace(minute=16))  # next candle
    depth = [json.loads(x) for x in gzip.open(tmp_path / "t1" / "depth-2026-10-16.jsonl.gz", "rt")]
    assert len(depth) == 4


def test_failed_depth_is_recorded_missing_and_decision_unchanged(tmp_path):
    s = _strategy(tmp_path, raise_book=True)
    assert _exercise(s) == [450.0, True, None, True, None, 1]
    depth = [json.loads(x) for x in gzip.open(tmp_path / "t1" / "depth-2026-10-16.jsonl.gz", "rt")]
    assert depth and all(d["depth_missing"] and d["bids"] == [] for d in depth)


def test_record_dir_inside_repo_is_refused(tmp_path):
    s = _strategy(STRAT / "research" / "live_pnl")
    with pytest.raises(ValueError, match="inside the strategies repository"):
        s.custom_stake_amount("AAA/USDT", T0, 100.0, 900.0, 10.0, 9000.0, 1.0, None, "long")


def test_mixin_does_not_change_the_model_key():
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cfg = load_config(STRAT / "config" / "config_wf.json")
    base = load_strategy_class("NNNC_A_Raw_Open")
    rec = type("NNNC_A_Raw_Open_REC_T", (ER.ExecutionRecorderMixin, base),
               {"RECORD_EXECUTION": True, "RECORD_DIR": "/tmp/x", "RECORD_RUN_ID": "t"})
    assert MK.key_hash(MK.resolve(rec, cfg)) == MK.key_hash(MK.resolve(base, cfg))


def test_nan_enter_long_never_breaks_analysis(tmp_path):
    """Live regression 2026-10-16: enter_long is NaN on non-signal rows; int(NaN) raised a
    ValueError that _rec_safe re-raised into freqtrade's analyze_ticker for every pair."""
    s = _strategy(tmp_path)
    df = pd.DataFrame({"date": [pd.Timestamp("2026-10-16 11:45", tz="UTC")], "close": [100.0],
                       "volume": [50.0], "%predict_prob_buy": [float("nan")],
                       "enter_long": [float("nan")]})
    out = s.populate_entry_trend(df, {"pair": "AAA/USDT"})
    assert len(out) == 1
    sig = json.loads((tmp_path / "t1" / "signals-2026-10-16.jsonl").read_text().splitlines()[0])
    assert sig["enter_long"] == 0 and sig["p_buy"] is None


def test_recording_errors_are_swallowed_but_config_errors_are_not(tmp_path):
    s = _strategy(tmp_path)
    s.dp.get_analyzed_dataframe = lambda pair, timeframe: (_ for _ in ()).throw(ValueError("boom"))
    assert s.confirm_trade_entry("AAA/USDT", "limit", 4.5, 100.0, "GTC", T0, None, "long") is True
