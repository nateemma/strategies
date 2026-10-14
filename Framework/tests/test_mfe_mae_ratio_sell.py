"""034: labels_mfe_mae_ratio_sell -- the mirror of labels_mfe_mae_ratio, wired into the sell
accessor (before it, the accessor fell back to the BUY rule)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Framework import TrainingSignals as T


def _df(close):
    close = np.asarray(close, float)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})


def test_sell_is_the_buy_rule_on_the_mirrored_path():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, 3000)))
    buy_on_mirror = T.labels_mfe_mae_ratio(_df(1e4 / c), horizon=48, min_gain=0.02).to_numpy()
    sell = T.labels_mfe_mae_ratio_sell(_df(c), horizon=48, min_loss=0.02).to_numpy()
    # 1/c turns a fall into a rise; thresholds differ only at second order, so compare on bars
    # well inside or outside the rule
    assert (sell == 1).sum() > 100
    agree = (sell == buy_on_mirror).mean()
    assert agree > 0.97


def test_hand_paths():
    flat = np.full(60, 98.0)
    down = _df(np.r_[100.0, 100.5, 99.0, 98.0, flat])           # -2% with a +0.5% adverse: ratio 4
    assert T.labels_mfe_mae_ratio_sell(down, horizon=48, min_loss=0.02).iloc[0] == 1
    chop = _df(np.r_[100.0, 101.5, 98.0, flat])                  # -2% with +1.5% adverse: ratio 1.33
    assert T.labels_mfe_mae_ratio_sell(chop, horizon=48, min_loss=0.02).iloc[0] == 0
    small = _df(np.r_[100.0, 99.0, np.full(60, 99.0)])           # -1% only
    assert T.labels_mfe_mae_ratio_sell(small, horizon=48, min_loss=0.02).iloc[0] == 0


def test_accessor_uses_the_sell_variant_and_no_longer_returns_the_buy_rule():
    rng = np.random.default_rng(1)
    df = _df(100 * np.exp(np.cumsum(rng.normal(0, 0.006, 2000))))
    p = {"horizon": 48, "min_gain": 0.02, "min_loss": 0.02, "rvol_threshold": -1.0}
    b = T.get_train_buy_signals(df, method=T.LabelMethod.mfe_mae_ratio, params=p).to_numpy() > 0.5
    s = T.get_train_sell_signals(df, method=T.LabelMethod.mfe_mae_ratio, params=p).to_numpy() > 0.5
    assert b.any() and s.any() and not (b & s).any()
    direct = T.labels_mfe_mae_ratio_sell(df, horizon=48, min_loss=0.02).to_numpy() > 0.5
    assert (s == direct).all()


def test_buy_side_rule():
    rng = np.random.default_rng(2)
    df = _df(100 * np.exp(np.cumsum(rng.normal(0, 0.006, 1500))))
    close = df["close"].to_numpy(float)
    mfe = (T._rolling_max_forward(close, 48) - close) / close
    mae = np.maximum(T._max_adverse_excursion(close, 48), 0.0)
    want = np.where(np.isnan(mfe) | np.isnan(mae), 0,
                    ((mfe >= 0.02) & (mfe / (mae + 1e-12) >= 2.0)).astype(int))
    assert (T.labels_mfe_mae_ratio(df, horizon=48, min_gain=0.02).to_numpy() == want).all()


def test_clean_moves_are_labelled_on_both_sides():
    """A move that never goes against the trade is the cleanest case; before the clamp its
    negative adverse excursion made the ratio negative and the bar unlabelled."""
    up = _df(np.r_[100.0, 100.5, 101.0, 102.5, np.full(60, 102.5)])
    assert T.labels_mfe_mae_ratio(up, horizon=48, min_gain=0.02).iloc[0] == 1
    down = _df(np.r_[100.0, 99.5, 99.0, 97.5, np.full(60, 97.5)])
    assert T.labels_mfe_mae_ratio_sell(down, horizon=48, min_loss=0.02).iloc[0] == 1
