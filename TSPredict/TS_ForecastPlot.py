"""Plot the 038 trend forecasts (research/forecast): damped Holt on several look-back windows, and the
vote C signal. For plotting / inspection only; it is not a trading strategy (038: nothing passed).

At each candle t, Holt (no detrending or smoothing: Holt models the trend itself) is fitted to the
trailing WINDOWS log closes and forecasts close FORECAST_N candles ahead. Two views:
- main plot: `fc_close_<w>` = that forecast drawn at t + N, on top of the actual close it predicts;
- subplot "forecast bp": the forecast N-candle return (basis points) at t, with the 30 bp round-trip
  cost line. Entry markers = vote C: at least VOTE_NEED of the windows forecast > +30 bp.

    zsh user_data/strategies/scripts/plot_strat.sh -c user_data/strategies/config/config_wf.json \\
        -n 14 TSPredict TS_ForecastPlot SOL/USDT

Cost: one Holt fit per window per candle (~4-15 ms), so keep plot ranges to weeks, not years.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
import Forecasters as FC  # noqa: E402


class TS_ForecastPlot(IStrategy):
    timeframe = "15m"
    FORECAST_N = 48  # candles ahead (48 = 12 h, where vote C was best)
    WINDOWS = (32, 64, 128, 256)
    VOTE_NEED = 3
    COST_LOG = 0.003  # 30 bp round trip: 0.10% fee + 5 bp slippage per side
    startup_candle_count = 256

    # illustrative trades only: hold exactly FORECAST_N candles (a -100% ROI target at N x 15 min)
    minimal_roi = {"0": 100.0, str(FORECAST_N * 15): -1.0}
    stoploss = -0.99
    can_short = False
    process_only_new_candles = True

    _colors = ("orange", "crimson", "mediumpurple", "teal")
    plot_config = {
        "main_plot": {"close": {"color": "lightsteelblue"},
                      **{f"fc_close_{w}": {"color": c} for w, c in zip(WINDOWS, _colors)}},
        "subplots": {
            "forecast bp": {**{f"fc_bp_{w}": {"color": c} for w, c in zip(WINDOWS, _colors)},
                            "cost_bp": {"color": "gray"}},
            "vote": {"vote_count": {"color": "black", "type": "bar"}},
        },
    }

    def _holt(self, x: np.ndarray, n: int) -> float:
        f = FC.make_forecaster(FC.ForecasterType.HOLT)
        f.set_detrend(False)
        f.set_smooth(False)
        try:
            return float(np.asarray(f.forecast(x.copy(), n), float)[-1])  # [-1] = exactly n ahead
        except Exception:
            return np.nan

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lc = np.log(dataframe["close"].to_numpy(float))
        n_rows, n = len(lc), self.FORECAST_N
        votes = np.zeros(n_rows, dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for w in self.WINDOWS:
                fc_ret = np.full(n_rows, np.nan)
                for t in range(max(self.WINDOWS) - 1, n_rows):
                    fc_ret[t] = self._holt(lc[t - w + 1: t + 1], n) - lc[t]
                dataframe[f"fc_bp_{w}"] = fc_ret * 1e4
                # the forecast close, drawn at the candle it predicts (t + n)
                dataframe[f"fc_close_{w}"] = (np.exp(lc + fc_ret))
                dataframe[f"fc_close_{w}"] = dataframe[f"fc_close_{w}"].shift(n)
                votes += np.nan_to_num(fc_ret > self.COST_LOG).astype(int)
        dataframe["vote_count"] = votes
        dataframe["cost_bp"] = self.COST_LOG * 1e4
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = (dataframe["vote_count"] >= self.VOTE_NEED).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
