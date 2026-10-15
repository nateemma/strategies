"""035 T005: timestamp units, kline and snapshot alignment, gap handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from research.new_inputs import sources as S


def test_ms_and_us_stamps_give_the_same_dates():
    t = pd.Timestamp("2025-03-01 00:15", tz="UTC")
    ms = int(t.value // 10**6)
    assert list(S.to_utc([ms, ms * 1000])) == [t, t]


def test_kline_text_with_and_without_header():
    row = "1735689600000000,1,2,0.5,1.5,10,1735690499999999,15,7,4,6,0"
    a = S.parse_klines_text(row)
    b = S.parse_klines_text("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                            "taker_buy_volume,taker_buy_quote_volume,ignore\n" + row)
    assert a.equals(b) and a["open_time"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert a["trades"].iloc[0] == 7 and a["taker_buy_base"].iloc[0] == 4


def test_same_stamp_klines_align_and_absent_rows_are_nan():
    dates = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    src = pd.DataFrame({"open_time": dates[[0, 2, 3]], "volume": [1.0, 3.0, 4.0]})
    out = S.align_kline(src, dates, ["volume"])
    assert out["volume"].iloc[0] == 1.0 and np.isnan(out["volume"].iloc[1])


def test_snapshot_after_the_bar_close_is_not_used():
    dates = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    close0 = dates[0] + S.BAR
    snaps = pd.DataFrame({"create_time": [close0 - pd.Timedelta(minutes=10),
                                          close0 - pd.Timedelta(minutes=5)
                                          + pd.Timedelta(seconds=1)],
                          "v": [1.0, 2.0]})
    out = S.align_snapshots(snaps, dates, ["v"])
    assert out["v"].iloc[0] == 1.0          # the second becomes usable 1 s after bar 0 closes
    assert out["v"].iloc[1] == 2.0


def test_stale_snapshot_is_absent():
    dates = pd.date_range("2024-01-01 06:00", periods=1, freq="15min", tz="UTC")
    snaps = pd.DataFrame({"create_time": [pd.Timestamp("2024-01-01 00:00", tz="UTC")], "v": [1.0]})
    assert np.isnan(S.align_snapshots(snaps, dates, ["v"])["v"].iloc[0])


def test_truncation_helpers_match_the_alignment_rules():
    cut = pd.Timestamp("2024-01-01 01:00", tz="UTC")
    k = pd.DataFrame({"open_time": [cut, cut + S.BAR]})
    assert len(S.truncate_klines(k, cut)) == 1
    s = pd.DataFrame({"create_time": [cut + pd.Timedelta(minutes=10),
                                      cut + pd.Timedelta(minutes=11)]})
    assert len(S.truncate_snapshots(s, cut)) == 1
