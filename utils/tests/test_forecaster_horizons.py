"""Known-answer tests for utils/Forecasters.py (feature 038, research R4: D1-D5, D8).

The defect that matters is the horizon index: forecast(x, h)[-h:] must be the underlying model's own
1..h-step forecast, so a one-candle shift fails. Trend models are also checked on a straight line;
smoothing models are not, because SES/ExpSmoothing/ETS/ARIMA(2,1,2) model no trend and damped Holt
flattens out.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import statsmodels.tsa.api as tsa
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.forecasting.theta import ThetaModel

import Forecasters as FC

warnings.filterwarnings("ignore")
X = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 64))
Y = np.cumsum(np.random.default_rng(5).normal(0, 0.01, 64)) + 0.3
T = FC.ForecasterType

REFERENCE = {
    T.HOLT: lambda x, h: tsa.Holt(x, damped_trend=True, initialization_method="estimated").fit().forecast(h),
    T.EXPONENTAL: lambda x, h: tsa.ExponentialSmoothing(x, initialization_method="estimated").fit().forecast(h),
    T.SIMPLE_EXPONENTAL: lambda x, h: tsa.SimpleExpSmoothing(x, initialization_method="estimated").fit().forecast(h),
    T.ETS: lambda x, h: tsa.ETSModel(x).fit(disp=False).forecast(h),
    T.SS_EXPONENTAL: lambda x, h: tsa.statespace.ExponentialSmoothing(x).fit(disp=False).forecast(h),
    T.ARIMA: lambda x, h: tsa.ARIMA(x, order=(2, 1, 2)).fit().forecast(h),
    T.THETA: lambda x, h: ThetaModel(x, period=1).fit().forecast(h, theta=1.2),
    T.AR: lambda x, h: tsa.AutoReg(x, lags=ar_select_order(x, maxlag=15).ar_lags).fit().predict(len(x), len(x) + h - 1),
}


def _make(t):
    f = FC.make_forecaster(t)
    f.set_detrend(False)
    return f


@pytest.mark.parametrize("t", list(REFERENCE), ids=lambda t: t.name)
@pytest.mark.parametrize("h", [1, 4, 96])
def test_horizon_index_matches_underlying_model(t, h):
    out = np.asarray(_make(t).forecast(X.copy(), h), dtype=float)
    assert out.shape == (len(X),)
    np.testing.assert_allclose(out[-h:] if h < len(X) else out, np.asarray(REFERENCE[t](X, h))[-len(X):],
                               rtol=1e-9, atol=1e-12)
    if h < len(X):
        np.testing.assert_array_equal(out[: len(X) - h], X[h:])  # the head is the input, shifted


@pytest.mark.parametrize("t", [T.ARIMA, T.THETA], ids=lambda t: t.name)
def test_model_is_refitted_on_each_window(t):
    f = _make(t)
    a = np.asarray(f.forecast(X.copy(), 4))[-1]
    b = np.asarray(f.forecast(Y.copy(), 4))[-1]
    np.testing.assert_allclose(b, np.asarray(REFERENCE[t](Y, 4))[-1], rtol=1e-9)
    assert a != b


@pytest.mark.parametrize("t", [T.LINEAR, T.QUADRATIC], ids=lambda t: t.name)
@pytest.mark.parametrize("h", [1, 4, 32])
def test_trend_models_exact_on_a_line(t, h):
    line = 0.01 * np.arange(64)
    out = np.asarray(_make(t).forecast(line.copy(), h), dtype=float)
    assert abs(out[-1] - 0.01 * (63 + h)) < 1e-6


@pytest.mark.parametrize("t", list(REFERENCE) + [T.LINEAR, T.QUADRATIC], ids=lambda t: t.name)
def test_constant_series_forecasts_the_constant(t):
    x = np.full(64, 0.5) + np.r_[np.zeros(63), 1e-9]  # a hair of variance so every fit is defined
    out = np.asarray(_make(t).forecast(x.copy(), 4), dtype=float)
    assert np.allclose(out[-4:], 0.5, atol=1e-6)


def test_arima_failure_raises():
    with pytest.raises(FC.ForecastFailed):
        _make(T.ARIMA).forecast(np.array([np.nan] * 64), 4)


def test_fft_extrapolation_is_unavailable():
    with pytest.raises(FC.ForecastUnavailable):
        _make(T.FFT_EXTRAPOLATION).forecast(X.copy(), 4)


def test_regressor_retrend_uses_the_forecast_windows_own_trend():
    """D4: with detrending on, a trained regressor restores the trend of the window it forecasts,
    not the trend of its training targets."""
    rng = np.random.default_rng(2)
    f = FC.make_forecaster(T.PA)
    f.set_detrend(True)
    train = rng.normal(size=(64, 1))
    f.train(train, rng.normal(size=64) + 5.0, incremental=False)  # training targets far from 0
    w1 = rng.normal(size=(64, 1)) + 1.0
    w2 = rng.normal(size=(64, 1)) - 1.0
    p1 = np.asarray(f.forecast(w1, 4), dtype=float)
    p2 = np.asarray(f.forecast(w2, 4), dtype=float)
    # the restored level follows each window (+1 vs -1), not the training targets (+5)
    assert abs(p1.mean() - 1.0) < 1.0 and abs(p2.mean() + 1.0) < 1.0
