"""032 T018: entry violations are counted on the signal bar (entry bar - 1)."""

from __future__ import annotations

import pandas as pd

from research.candidate_training import instrument as I


def test_violation_is_judged_on_the_signal_bar():
    dates = pd.date_range("2023-01-01", periods=4, freq="15min", tz="UTC")
    table = pd.DataFrame({"date": dates.tz_localize(None), "pair": "X/USDT",
                          "guard_metric": [0.1, 0.9, 0.1, 0.1], "bb_width": [0.05, 0.05, 0.05, 0.01]})
    rec = pd.DataFrame({"fold": "f00", "pair": "X/USDT",
                        "open_date": [dates[1], dates[2], dates[3], dates[1]]})
    bars = lambda pair: (None, pd.DatetimeIndex(dates))  # noqa: E731
    # entry at bar 1 <- signal bar 0 (ok); bar 2 <- bar 1 (guard 0.9: bad); bar 3 <- bar 2 (ok)
    assert I.violations(rec.iloc[:3], {"f00": table}, bars) == 1
    # a signal bar missing from the table counts as a violation
    assert I.violations(rec.iloc[:1], {"f00": table.iloc[1:]}, bars) == 1


def test_achieved_rates_parses_the_pick_log(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "FOLDS_DIR", tmp_path)
    (tmp_path / "thresholds_co_aro_log.txt").write_text(
        "f00: threshold=0.5342  achieved_rate=1.4736%  target_rate=1.4735%  n_scored=440092\n")
    assert I.achieved_rates("aro") == {"f00": {"threshold": 0.5342, "achieved_rate": 0.014736}}
