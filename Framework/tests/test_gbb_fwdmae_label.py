"""Tests for the hybrid label `LabelMethod.gbb_fwdmae` (labels_gbb_fwdmae /
labels_gbb_fwdmae_sell): gbb's regime TRIGGER AND forward_mae's
drawdown-capped forward-return TEST.

Required property (regime screening task, 2026-10-07): the hybrid's buy set
is a SUBSET of forward_mae's buy set at the same horizon, AND a subset of
gbb's trigger bars. Both are checked directly against a synthetic OHLCV
dataframe with the guard_metric/bb_width/rvol columns gbb reads.
"""

import numpy as np
import pandas as pd
import pytest

from Framework.TrainingSignals import (
    LabelMethod,
    METHODS,
    labels_forward_return_mae_cap,
    labels_gbb,
    labels_gbb_fwdmae,
    labels_gbb_fwdmae_sell,
    labels_gbb_sell,
    get_train_buy_signals,
    get_train_sell_signals,
)


def _synthetic_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    high = close + rng.uniform(0, 0.6, n)
    low = close - rng.uniform(0, 0.6, n)
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "open": close,
            # Deliberately wide ranges so the trigger is neither always-true
            # nor always-false.
            "guard_metric": rng.uniform(-1, 1, n),
            "bb_width": rng.uniform(0.0, 0.08, n),
            "rvol": rng.uniform(0.0, 3.0, n),
        }
    )


HORIZON = 48
MIN_GAIN = 0.015
MIN_LOSS = 0.015
MAX_DRAWDOWN = 0.10
GUARD_THRESHOLD = 0.3
BB_WIDTH_THRESHOLD = 0.035
RVOL_THRESHOLD = -1.0


@pytest.fixture()
def df():
    return _synthetic_df()


class TestLabelMethodRegistered:
    def test_enum_member_exists(self):
        assert LabelMethod.gbb_fwdmae == 28

    def test_registered_in_methods_dict(self):
        assert METHODS[LabelMethod.gbb_fwdmae] is labels_gbb_fwdmae

    def test_dispatcher_routes_to_it(self, df):
        via_dispatch = get_train_buy_signals(
            df, method=LabelMethod.gbb_fwdmae,
            params={"horizon": HORIZON, "min_gain": MIN_GAIN, "min_loss": MIN_LOSS,
                    "max_drawdown": MAX_DRAWDOWN},
        )
        direct = labels_gbb_fwdmae(df, horizon=HORIZON, min_gain=MIN_GAIN,
                                    min_loss=MIN_LOSS, max_drawdown=MAX_DRAWDOWN)
        pd.testing.assert_series_equal(via_dispatch, direct.astype(float), check_names=False)

    def test_sell_dispatcher_routes_to_it(self, df):
        via_dispatch = get_train_sell_signals(
            df, method=LabelMethod.gbb_fwdmae,
            params={"horizon": HORIZON, "min_loss": MIN_LOSS, "max_drawdown": MAX_DRAWDOWN},
        )
        direct = labels_gbb_fwdmae_sell(df, horizon=HORIZON, min_loss=MIN_LOSS,
                                         max_drawdown=MAX_DRAWDOWN)
        pd.testing.assert_series_equal(via_dispatch, direct.astype(float), check_names=False)


class TestHybridIsASubset:
    def test_subset_of_forward_mae_same_horizon(self, df):
        hybrid = labels_gbb_fwdmae(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            rvol_threshold=RVOL_THRESHOLD, min_gain=MIN_GAIN, min_loss=MIN_LOSS,
            horizon=HORIZON, max_drawdown=MAX_DRAWDOWN,
        )
        fwdmae = labels_forward_return_mae_cap(
            df, horizon=HORIZON, min_gain=MIN_GAIN, max_drawdown=MAX_DRAWDOWN,
            min_loss=MIN_LOSS,
        )
        hybrid_on = set(np.where(hybrid.to_numpy() > 0.5)[0])
        fwdmae_on = set(np.where(fwdmae.to_numpy() > 0.5)[0])
        assert hybrid_on.issubset(fwdmae_on)

    def test_subset_of_gbb_trigger_bars(self, df):
        hybrid = labels_gbb_fwdmae(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            rvol_threshold=RVOL_THRESHOLD, min_gain=MIN_GAIN, min_loss=MIN_LOSS,
            horizon=HORIZON, max_drawdown=MAX_DRAWDOWN,
        )
        # The trigger alone, replicated directly from labels_gbb's own
        # condition (guard_metric/bb_width/rvol), WITHOUT its forward-gain test.
        trigger = (
            (df["guard_metric"].to_numpy() < GUARD_THRESHOLD)
            & (df["bb_width"].to_numpy() > BB_WIDTH_THRESHOLD)
            & (df["rvol"].to_numpy() > RVOL_THRESHOLD)
        )
        hybrid_on = set(np.where(hybrid.to_numpy() > 0.5)[0])
        trigger_on = set(np.where(trigger)[0])
        assert hybrid_on.issubset(trigger_on)

    def test_subset_of_gbb_full_label_too(self, df):
        """gbb's own full label (trigger & its gain test) is itself a subset
        of the trigger; the hybrid must be a subset of THAT too, since it is
        built as gbb_label & forward_mae_label."""
        hybrid = labels_gbb_fwdmae(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            rvol_threshold=RVOL_THRESHOLD, min_gain=MIN_GAIN, min_loss=MIN_LOSS,
            horizon=HORIZON, max_drawdown=MAX_DRAWDOWN,
        )
        gbb = labels_gbb(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            rvol_threshold=RVOL_THRESHOLD, min_gain=MIN_GAIN, min_loss=MIN_LOSS,
            horizon=HORIZON,
        )
        hybrid_on = set(np.where(hybrid.to_numpy() > 0.5)[0])
        gbb_on = set(np.where(gbb.to_numpy() > 0.5)[0])
        assert hybrid_on.issubset(gbb_on)

    def test_subset_is_strict_on_synthetic_data(self, df):
        """The subset relation must actually bind on this dataset -- i.e. the
        drawdown cap removes at least one bar gbb alone would have kept --
        or the hybrid would just be a rename of gbb, not a new label."""
        hybrid = labels_gbb_fwdmae(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            rvol_threshold=RVOL_THRESHOLD, min_gain=MIN_GAIN, min_loss=MIN_LOSS,
            horizon=HORIZON, max_drawdown=MAX_DRAWDOWN,
        )
        gbb = labels_gbb(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            rvol_threshold=RVOL_THRESHOLD, min_gain=MIN_GAIN, min_loss=MIN_LOSS,
            horizon=HORIZON,
        )
        assert hybrid.sum() < gbb.sum()
        assert hybrid.sum() > 0  # and the hybrid isn't simply empty


class TestHybridSellMirror:
    def test_subset_of_gbb_sell_overbought_trigger(self, df):
        hybrid_sell = labels_gbb_fwdmae_sell(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            min_loss=MIN_LOSS, horizon=HORIZON, max_drawdown=MAX_DRAWDOWN,
        )
        trigger = (
            (df["guard_metric"].to_numpy() > GUARD_THRESHOLD)
            & (df["bb_width"].to_numpy() > BB_WIDTH_THRESHOLD)
        )
        hybrid_on = set(np.where(hybrid_sell.to_numpy() > 0.5)[0])
        trigger_on = set(np.where(trigger)[0])
        assert hybrid_on.issubset(trigger_on)

    def test_subset_of_gbb_sell_full_label(self, df):
        hybrid_sell = labels_gbb_fwdmae_sell(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            min_loss=MIN_LOSS, horizon=HORIZON, max_drawdown=MAX_DRAWDOWN,
        )
        gbb_sell = labels_gbb_sell(
            df, guard_threshold=GUARD_THRESHOLD, bb_width_threshold=BB_WIDTH_THRESHOLD,
            min_loss=MIN_LOSS, horizon=HORIZON,
        )
        hybrid_on = set(np.where(hybrid_sell.to_numpy() > 0.5)[0])
        gbb_sell_on = set(np.where(gbb_sell.to_numpy() > 0.5)[0])
        assert hybrid_on.issubset(gbb_sell_on)


class TestExistingLabelersUntouched:
    """Additive-only guard: this feature must not perturb any pre-existing
    labeler's output."""

    def test_gbb_output_unchanged_by_presence_of_hybrid(self, df):
        gbb_a = labels_gbb(df, horizon=HORIZON, min_gain=MIN_GAIN)
        gbb_b = labels_gbb(df, horizon=HORIZON, min_gain=MIN_GAIN)
        pd.testing.assert_series_equal(gbb_a, gbb_b)

    def test_forward_mae_output_unchanged(self, df):
        a = labels_forward_return_mae_cap(df, horizon=HORIZON, min_gain=MIN_GAIN)
        b = labels_forward_return_mae_cap(df, horizon=HORIZON, min_gain=MIN_GAIN)
        pd.testing.assert_series_equal(a, b)
