"""Tests for the generic per-labeler kwarg filter in the label dispatcher
(`Framework.TrainingSignals._filter_label_kwargs`, wired into
`get_train_buy_signals` / `get_train_sell_signals`).

The landmine this fixes: `BaseNNStrategy.get_train_buy_signals` always puts
`rvol_threshold` in its params dict (meaningful only for gbb/gbb_path), and
the dispatcher used to call `fn(df, **params)` with NO filtering, so every
OTHER labeler raised `TypeError` before producing a single label. The fix
must (a) drop GENERIC keys the dispatcher adds to every call silently when a
labeler doesn't accept them, and (b) RAISE if a caller explicitly names a key
(`strict_keys`) that the chosen labeler cannot accept -- silent dropping of
an explicit request is how an earlier run of this study trained on the wrong
label unnoticed.
"""

import numpy as np
import pandas as pd
import pytest

from Framework.TrainingSignals import (
    LabelMethod,
    _filter_label_kwargs,
    get_train_buy_signals,
    get_train_sell_signals,
    labels_forward_return_mae_cap,
    labels_gbb,
    labels_triple_barrier,
)


def _synthetic_df(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "close": close,
            "high": close + rng.uniform(0, 0.5, n),
            "low": close - rng.uniform(0, 0.5, n),
            "open": close,
            "guard_metric": rng.uniform(-1, 1, n),
            "bb_width": rng.uniform(0.0, 0.08, n),
            "rvol": rng.uniform(0.0, 3.0, n),
        }
    )


@pytest.fixture()
def df():
    return _synthetic_df()


# ---------------------------------------------------------------------------
# _filter_label_kwargs unit behaviour
# ---------------------------------------------------------------------------

class TestFilterLabelKwargsUnit:
    def test_drops_unsupported_generic_key_silently(self):
        params = {"horizon": 12, "min_gain": 0.02, "rvol_threshold": -1.0}
        # labels_triple_barrier does not accept rvol_threshold.
        out = _filter_label_kwargs(labels_triple_barrier, params)
        assert out == {"horizon": 12, "min_gain": 0.02}

    def test_keeps_supported_key(self):
        params = {"horizon": 12, "rvol_threshold": -1.0}
        out = _filter_label_kwargs(labels_gbb, params)
        assert out == params

    def test_raises_on_unsupported_explicit_key(self):
        params = {"horizon": 12, "min_gain": 0.02, "bogus_arg": 1.0}
        with pytest.raises(ValueError, match="bogus_arg"):
            _filter_label_kwargs(labels_triple_barrier, params, strict_keys={"bogus_arg"})

    def test_does_not_raise_when_explicit_key_is_supported(self):
        params = {"horizon": 12, "pt": 0.015, "sl": 0.015}
        out = _filter_label_kwargs(labels_triple_barrier, params, strict_keys={"pt", "sl"})
        assert out == params

    def test_strict_keys_none_is_pure_generic_drop(self):
        params = {"horizon": 12, "rvol_threshold": -1.0, "bogus_arg": 1.0}
        # No strict_keys => every key is generic => drop silently, never raise.
        out = _filter_label_kwargs(labels_triple_barrier, params, strict_keys=None)
        assert out == {"horizon": 12}

    def test_excludes_df_from_accepted_set(self):
        # "df" is the first positional parameter of every labeler; must never
        # be treated as a settable kwarg.
        out = _filter_label_kwargs(labels_gbb, {"df": "should never pass through"})
        assert out == {}


# ---------------------------------------------------------------------------
# Dispatcher-level: the bug this closes
# ---------------------------------------------------------------------------

class TestDispatcherNoLongerRaisesOnGenericMismatch:
    """Before the fix, every one of these raised TypeError via
    fn(df, **params) because rvol_threshold (always injected by
    BaseNNStrategy.get_train_buy_signals) is not in these labelers'
    signatures."""

    @pytest.mark.parametrize(
        "method",
        [
            LabelMethod.future_expectancy,
            LabelMethod.future_sharpe,
            LabelMethod.future_sortino,
            LabelMethod.mfe_mae_ratio,
            LabelMethod.risk_adj_future,
            LabelMethod.local_min_followthrough,
        ],
    )
    def test_buy_dispatch_survives_generic_rvol_threshold(self, df, method):
        result = get_train_buy_signals(
            df,
            method=method,
            params={"horizon": 24, "min_gain": 0.02, "min_loss": 0.02,
                    "rvol_threshold": -1.0},
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_triple_barrier_buy_dispatch_survives_generic_rvol_threshold(self, df):
        result = get_train_buy_signals(
            df,
            method=LabelMethod.triple_barrier,
            params={"horizon": 12, "min_gain": 0.015, "min_loss": 0.015,
                    "rvol_threshold": -1.0},
        )
        assert isinstance(result, pd.Series)


class TestExplicitStrictKeyRaisesThroughDispatcher:
    def test_buy_dispatcher_raises_on_unsupported_explicit_key(self, df):
        with pytest.raises(ValueError):
            get_train_buy_signals(
                df,
                method=LabelMethod.future_sharpe,
                params={"horizon": 24, "pt": 0.015},  # future_sharpe has no `pt`
                strict_keys={"pt"},
            )

    def test_buy_dispatcher_accepts_explicit_key_when_supported(self, df):
        # triple_barrier DOES accept pt/sl -- no raise.
        result = get_train_buy_signals(
            df,
            method=LabelMethod.triple_barrier,
            params={"horizon": 12, "min_gain": 0.015, "pt": 0.015, "sl": 0.015},
            strict_keys={"pt", "sl"},
        )
        assert isinstance(result, pd.Series)

    def test_sell_fallback_raises_on_unsupported_explicit_key(self, df):
        # risk_adj_future has no dedicated sell branch -> falls back to the
        # buy accessor, which must still enforce strict_keys.
        with pytest.raises(ValueError):
            get_train_sell_signals(
                df,
                method=LabelMethod.risk_adj_future,
                params={"horizon": 24, "bogus_arg": 1.0},
                strict_keys={"bogus_arg"},
            )


# ---------------------------------------------------------------------------
# Byte-identical guarantee for the 4 methods already used in production
# ---------------------------------------------------------------------------

class TestPreExistingMethodsByteIdentical:
    """The generic filter must be a no-op for every key these 4 methods
    were already called with (that's why they tolerated the unfiltered
    landmine in the first place)."""

    def test_forward_mae_byte_identical(self, df):
        params = {"horizon": 48, "min_gain": 0.02, "min_loss": 0.02,
                  "rvol_threshold": -1.0}
        before = labels_forward_return_mae_cap(df, **params).astype(float)
        after = get_train_buy_signals(df, method=LabelMethod.forward_mae, params=params)
        pd.testing.assert_series_equal(before, after, check_names=False)

    def test_gbb_byte_identical(self, df):
        params = {"horizon": 48, "min_gain": 0.02, "min_loss": 0.02,
                  "rvol_threshold": -1.0}
        before = labels_gbb(df, **params).astype(float)
        after = get_train_buy_signals(df, method=LabelMethod.gbb, params=params)
        pd.testing.assert_series_equal(before, after, check_names=False)

    def test_triple_barrier_byte_identical(self, df):
        params = {"horizon": 12, "min_gain": 0.015, "min_loss": 0.015,
                  "rvol_threshold": -1.0}
        before = labels_triple_barrier(df, horizon=12, min_gain=0.015).astype(float)
        after = get_train_buy_signals(df, method=LabelMethod.triple_barrier, params=params)
        pd.testing.assert_series_equal(before, after, check_names=False)
