"""037 T018 / T028: targets -- full-range labels equal the per-fold labels; regime targets are
forward-only, baselines backward-only, cut points from the declared span."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.indicator_screen import windows as W


def test_full_range_labels_equal_the_per_fold_labels(loaded):
    """R4: labels are a function of the price path, so the full-range labels must equal the ones
    the label screen builds on fold f03's training window (except its last 48 bars, which have no
    future inside that window)."""
    from Framework.BaseStrategy import TradingAction
    from research.indicator_screen import targets as T
    from research.label_screen import audit as A

    H = loaded["H"]
    checked = 0
    for pair, strat, d in A._frames(H.ARO_CLASS, "f03"):
        full = H.load_ohlcv(loaded["cfg"], pair)
        buy, sell = T.labels_for(loaded["aro"], full, pair)
        k = np.asarray(strat.get_training_labels(d.copy()))[-len(d):]
        ref = pd.DataFrame({"date": d["date"].to_numpy(),
                            "b": (k == int(TradingAction.BUY)).astype(float),
                            "s": (k == int(TradingAction.SELL)).astype(float)}).iloc[:-W.LABEL_HORIZON]
        got = pd.DataFrame({"date": full["date"].to_numpy(), "b": buy, "s": sell})
        m = ref.merge(got, on="date", suffixes=("_ref", "_got"))
        assert len(m) == len(ref), pair
        assert np.array_equal(m["b_ref"], m["b_got"]), pair
        assert np.array_equal(m["s_ref"], m["s_got"]), pair
        checked += 1
    assert checked == 10


# ---------------------------------------------------------------------------- regime (T028)

from research.indicator_screen import targets as T  # noqa: E402

rng = np.random.default_rng(7)


def _close(n=3000):
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def test_forward_quantities_use_only_bars_after_t_and_baselines_only_bars_up_to_t():
    c = _close()
    t, n = 1500, 48
    for fw, bw in ((T.forward_return, T.trailing_return), (T.forward_vol, T.trailing_vol)):
        q, b = fw(c, n), bw(c, n)
        before = c.copy()
        before[t - 1] *= 1.05                           # a bar strictly before t
        assert fw(before, n)[t] == pytest.approx(q[t], rel=1e-12)
        after = c.copy()
        after[t + n] *= 1.05                            # the last bar inside (t, t+N]
        assert fw(after, n)[t] != q[t]
        assert bw(after, n)[t] == pytest.approx(b[t], rel=1e-12)   # the baseline never sees it (pandas rolling: 1-ulp noise)
        late = c.copy()
        late[t + n + 1] *= 1.05                         # beyond the horizon
        assert fw(late, n)[t] == pytest.approx(q[t], rel=1e-12)
    assert np.isnan(T.forward_return(c, n)[-n:]).all()


def test_event_means_forward_open_and_trailing_closed_only():
    g = 100
    s, cnt = np.zeros(g), np.zeros(g)
    s[10], cnt[10] = 0.02, 1                            # opened at 10
    sc, cc = np.zeros(g), np.zeros(g)
    sc[30], cc[30] = 0.02, 1                            # the same trade closes at 30
    f = T.forward_mean(s, cnt, 12)
    assert f[0] == pytest.approx(0.02) and f[9] == pytest.approx(0.02) and np.isnan(f[10])   # (t, t+N] excludes t itself
    b = T.trailing_mean(sc, cc, 12)
    assert np.isnan(b[30]) and b[31] == pytest.approx(0.02) and b[42] == pytest.approx(0.02) and np.isnan(b[43])


class _FakePanel:
    """One pair, 15m grid 2022-10-01 -> 2023-07-01 (covers folds f00, f01)."""

    def __init__(self):
        self.date = pd.date_range("2022-10-01", "2023-07-01", freq="15min", tz="UTC", inclusive="left")
        n = len(self.date)
        self.pairs, self.lengths = ["AAA/USDT"], [n]
        self.core = {"close": _close(n)}
        self.data_ok = np.ones(n, bool)
        self.candidate = np.zeros(n, bool)

    def __len__(self):
        return len(self.date)


def _regime(panel, trades):
    t0 = pd.Timestamp(W.DATA_START, tz="UTC")
    ev = [(int((o - t0) / W.BAR), int((c - t0) / W.BAR), p) for o, c, p in trades]
    return T.RegimeTargets(panel, a1_events=lambda: ev, a2_trades=lambda: ev)


def test_tercile_cut_points_come_from_the_training_span():
    p = _FakePanel()
    rt = _regime(p, [])
    f0 = W.by_id("f00")
    q = rt.quantity("B-pair:48")
    m = np.asarray((p.date >= f0.train_start) & (p.date < f0.start - 48 * W.BAR)) & np.isfinite(q)
    lo, hi = rt.cuts("B-pair:48", f0)
    assert lo == np.quantile(q[m], 1 / 3) and hi == np.quantile(q[m], 2 / 3)
    y = rt.y("B-pair:48", f0)
    t = f0.mask(p.date)
    assert set(np.unique(y[t][np.isfinite(y[t])])) == {0.0, 1.0}
    rt2 = _regime(p, [])
    rt2._q["B-pair:48"] = q.copy()
    rt2._q["B-pair:48"][t] += 5.0                   # test-window values cannot move the cuts
    assert rt2.cuts("B-pair:48", f0) == (lo, hi)


def test_a2_is_not_computable_outside_the_test_windows():
    p = _FakePanel()
    o = pd.Timestamp("2023-02-01 10:00", tz="UTC")
    rt = _regime(p, [(o, o + pd.Timedelta("6h"), 0.02), (o - pd.Timedelta("60D"), o - pd.Timedelta("59D"), 0.01)])
    q = rt.quantity("A2:48")
    inside = W.by_id("f00").mask(p.date) | W.by_id("f01").mask(p.date)
    assert np.isnan(q[~inside]).all()
    i = int(np.flatnonzero(p.date == o - pd.Timedelta("1h"))[0])
    assert q[i] == pytest.approx(0.02)
    b = rt.baseline("A2:48")
    j = int(np.flatnonzero(p.date == o + pd.Timedelta("7h"))[0])
    assert b[j] == pytest.approx(0.02)              # closed 1 h earlier, inside the trailing 12 h
    k = int(np.flatnonzero(p.date == o + pd.Timedelta("5h"))[0])
    assert np.isnan(b[k])                           # still open at t: excluded
    assert rt.market_wide and all(t.split(":")[0] in T.MARKET_WIDE for t in rt.market_wide)
