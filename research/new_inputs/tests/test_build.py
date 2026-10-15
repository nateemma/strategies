"""035 T007: value ranges, flags, the 026 reuse on cleaned frames, a hand-computed input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from research.new_inputs import build as B
from research.new_inputs import inputs as I
from utils import RegimeState as RS


@pytest.fixture(scope="module")
def built(synth):
    return B.build_all(synth)


def test_every_value_is_bounded_or_nan(built):
    for p, df in built.items():
        for col in I.all_columns():
            v = df[col].to_numpy(float)
            v = v[~np.isnan(v)]
            assert v.size == 0 or (v.min() >= -1 - 1e-12 and v.max() <= 1 + 1e-12), (p, col)


def test_positioning_flags_mark_undefined_values_including_warmup(built, synth):
    df = built["P1/USDT"]
    for inp in I.TABLE:
        col = I.column(inp.tag)
        if inp.missing_flag:
            assert (df[col + "_missing"].to_numpy() == df[col].isna().to_numpy()).all(), inp.tag
        else:
            assert col + "_missing" not in df
    start = synth["metrics"]["P1/USDT"]["create_time"].min()
    before = df["date"] < start
    assert df.loc[before, "ni_po_oi_missing"].eq(1).all()
    after = df["date"] > start + pd.Timedelta(hours=6)
    # z warm-up (960 values) has not been reached ~6 h after the source starts: still flagged
    assert df.loc[after, "ni_po_oi_missing"].head(10).eq(1).all()


def test_market_state_definitions(built, synth):
    """MS_VR: real-bar vol carried <= 24 h, cross median, / its 30-day median (min 960); MS_BR:
    026 breadth on the venue frames unchanged."""
    idx = pd.DatetimeIndex(sorted(set().union(*[set(f["date"]) for f in synth["ohlcv"].values()])))
    cols = {}
    for p, f in synth["ohlcv"].items():
        real = f[f["volume"] > 0]
        v = np.log(pd.Series(real["close"].to_numpy(), index=pd.DatetimeIndex(real["date"])))
        v = v.diff().rolling(96, min_periods=96).std()
        cols[p] = v.reindex(idx).ffill(limit=96)
    w = pd.DataFrame(cols)
    m = w.median(axis=1).where(w.notna().sum(axis=1) >= I.MIN_PAIRS)
    vr = m / m.rolling(2880, min_periods=960).median()
    dates = pd.DatetimeIndex(synth["ohlcv"]["P0/USDT"]["date"])
    got = built["P0/USDT"]["ni_ms_vr"].to_numpy(float)
    assert np.allclose(got, B.z(vr.reindex(dates)).to_numpy(float), equal_nan=True)
    assert np.isfinite(got).sum() > 1000
    br = RS.breadth({p: f[["date", "close"]] for p, f in synth["ohlcv"].items()},
                    min_pairs=I.MIN_PAIRS).reindex(dates)
    assert np.allclose(built["P0/USDT"]["ni_ms_br"].to_numpy(float), B.c(br).to_numpy(float),
                       equal_nan=True)


def test_taker_share_by_hand():
    k = pd.DataFrame({"volume": [10.0, 10, 10, 10, 20], "taker_buy_base": [5.0, 5, 5, 5, 20]})
    s = B.taker_share(k)
    assert np.isnan(s.iloc[2]) and s.iloc[3] == pytest.approx(0.5)
    assert s.iloc[4] == pytest.approx(35 / 50)
    assert B.c(s).iloc[3] == pytest.approx(0.0)


def test_us_order_flow_is_nan_on_gap_bars(built, synth):
    o = synth["ohlcv"]["P0/USDT"]
    gap = o["volume"].to_numpy() == 0
    df = built["P0/USDT"]
    assert df.loc[gap, "ni_of_tbu"].isna().all() and df.loc[gap, "ni_of_tru"].isna().all()
    assert df.loc[~gap, "ni_of_tbu"].notna().sum() > 0
