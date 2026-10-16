"""036 T013: parity matching and the declared cause order (research R4)."""

from __future__ import annotations

import pandas as pd

from research.live_pnl import parity as P


DAY = "2026-10-20"
T = lambda s: pd.Timestamp(f"2026-10-20 {s}", tz="UTC")  # noqa: E731


def _trades(rows):
    df = pd.DataFrame(rows, columns=["pair", "open_date", "close_date"])
    return P._norm_trades(df)


def _sig(rows):
    return pd.DataFrame(rows, columns=["pair", "candle", "p_buy", "close", "quote_volume"])


def _bts(rows):
    return pd.DataFrame(rows, columns=["pair", "date", "p_buy", "close", "volume"])


BASE_SIG = _sig([("A", T("10:00"), 0.9, 100.0, 5e4)])
BASE_BTS = _bts([("A", T("10:00"), 0.9, 100.0, 500.0)])


def _cmp(live, bt, sig=BASE_SIG, bts=BASE_BTS, **kw):
    return P.compare_day(DAY, live_trades=_trades(live), bt_trades=_trades(bt), live_sig=sig,
                         bt_sig=bts, **kw)


def _cause(out):
    assert len(out["differences"]) == 1, out
    return out["differences"][0]["cause"]


def test_matched():
    out = _cmp([("A", T("10:15:05"), T("14:00"))], [("A", T("10:15"), T("14:00"))])
    assert out["matched_entries"] == 1 and out["matched_exits"] == 1 and not out["differences"]


def test_prediction_difference_is_checked_first():
    bts = _bts([("A", T("10:00"), 0.70, 99.0, 0.0)])  # also a revision and a gap; p wins
    assert _cause(_cmp([("A", T("10:15:05"), None)], [], bts=bts)) == "prediction_difference"


def test_data_revision():
    bts = _bts([("A", T("10:00"), 0.9, 99.5, 500.0)])
    assert _cause(_cmp([("A", T("10:15:05"), None)], [], bts=bts)) == "data_revision"


def test_gap_fill():
    bts = _bts([("A", T("10:00"), 0.9, 100.0, 0.0)])
    assert _cause(_cmp([("A", T("10:15:05"), None)], [], bts=bts)) == "gap_fill"


def test_timing_one_candle_apart():
    out = _cmp([("A", T("10:15:05"), None)], [("A", T("10:30"), None)])
    assert {d["cause"] for d in out["differences"]} == {"timing"}


def test_timing_unfilled_live_order():
    out = _cmp([], [("A", T("10:15"), None)], unfilled={("A", T("10:00"))})
    assert _cause(out) == "timing"


def test_wallet_state_carried_position():
    # the live bot still held A from a previous-day difference; the backtest re-entered
    live = [("A", pd.Timestamp("2026-10-19 08:00", tz="UTC"), None)]
    out = _cmp(live, [("A", T("10:15"), None)])
    assert _cause(out) == "wallet_state"


def test_unexplained():
    sig = _sig([("A", T("10:00"), 0.9, 100.0, 5e4)])
    out = _cmp([("A", T("10:15:05"), None)], [], sig=sig)
    assert _cause(out) == "UNEXPLAINED" and out["unexplained"] == 1


def test_only_the_days_entries_are_compared():
    # a backtest-only entry on the previous day is outside the comparison
    out = _cmp([], [("B", pd.Timestamp("2026-10-19 10:15", tz="UTC"),
                     pd.Timestamp("2026-10-19 12:00", tz="UTC"))])
    assert out["differences"] == []


def test_trades_before_the_live_start_are_not_compared():
    """Smoke run 2026-10-16: a backtest entry at 03:30 UTC, before the bot started at 15:56, was
    reported UNEXPLAINED."""
    bt = [("A", T("03:45"), T("10:30")), ("A", T("16:15"), None)]
    live = [("A", T("16:15:04"), None)]
    out = _cmp(live, bt, since=T("15:56"))
    assert out["differences"] == [] and out["matched_entries"] == 1
