"""Plot the 038 trend forecasts (research/forecast): damped Holt on several look-back windows, and the
vote C signal. For plotting / inspection only; it is not a trading strategy (038: nothing passed).

At each forecast candle t, Holt (no detrending or smoothing: Holt models the trend itself) is fitted to
the trailing WINDOWS log closes and forecasts close FORECAST_N candles ahead. Two views:
- main plot: `fc_close_<w>` = that forecast drawn at t + N, on top of the actual close it predicts;
- subplot "forecast bp": the forecast N-candle return (basis points) at t, with the 30 bp round-trip
  cost line. Entry markers = vote C: at least VOTE_NEED of the windows forecast > +30 bp (only on
  forecast candles, so a rally gives one marker per STRIDE candles, not one per candle).

Speed (one statsmodels fit per window per forecast candle, ~4-15 ms):
- STRIDE: forecasts every STRIDE candles (default 16, the 038 scoring grid) and holds the value in
  between; STRIDE = 1 forecasts every candle (16x slower);
- disk cache: forecasts are kept in user_data/data/forecast_038/plot_cache/, so re-plotting a range is
  near-instant; where the 038 study already computed these forecasts (pred/holt*.feather, 2023-01 to
  2026-10) the cache is seeded from them and nothing is fitted.

    zsh user_data/strategies/scripts/plot_strat.sh -c user_data/strategies/config/config_wf.json \\
        -n 14 TSPredict TS_ForecastPlot SOL/USDT
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

STRAT = Path(__file__).resolve().parents[1]
sys.path.append(str(STRAT / "utils"))
import Forecasters as FC  # noqa: E402

FORECAST_DIR = STRAT.parents[1] / "user_data" / "data" / "forecast_038"
#: 038 prediction files holding Holt forecasts per window (stride-16 grid)
STUDY_PRED = {32: "holt_w32", 64: "holt", 128: "holt_w128", 256: "holt_w256",
              288: "holt_w288", 672: "holt_w672", 1344: "holt_w1344"}
BAR = pd.Timedelta(minutes=15)
EPOCH = pd.Timestamp(0, tz="UTC")


class TS_ForecastPlot(IStrategy):
    timeframe = "15m"
    FORECAST_N = 48  # candles ahead (48 = 12 h, where vote C was best)
    WINDOWS = (32, 64, 128, 256)
    VOTE_NEED = 3
    COST_LOG = 0.003  # 30 bp round trip: 0.10% fee + 5 bp slippage per side
    STRIDE = 16
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

    _cache: dict = {}

    # ------------------------------------------------------------------ forecasts + cache

    def _holt(self, x: np.ndarray, n: int) -> float:
        f = FC.make_forecaster(FC.ForecasterType.HOLT)
        f.set_detrend(False)
        f.set_smooth(False)
        try:
            return float(np.asarray(f.forecast(x.copy(), n), float)[-1])  # [-1] = exactly n ahead
        except Exception:
            return np.nan

    def _cache_path(self, pair: str, w: int) -> Path:
        return FORECAST_DIR / "plot_cache" / f"{pair.replace('/', '_')}_w{w}_n{self.FORECAST_N}.feather"

    def _load_cache(self, pair: str, w: int) -> pd.Series:
        """date -> forecast log return; from the plot cache, else seeded from the 038 predictions."""
        key = (pair, w, self.FORECAST_N)
        if key in self._cache:
            return self._cache[key]
        s = pd.Series(dtype=float)
        p = self._cache_path(pair, w)
        if p.exists():
            d = pd.read_feather(p)
            s = pd.Series(d["fc"].to_numpy(float), index=pd.DatetimeIndex(d["date"]))
        elif w in STUDY_PRED and (FORECAST_DIR / "pred" / f"{STUDY_PRED[w]}.feather").exists():
            col = f"f_{self.FORECAST_N}"
            try:
                d = pd.read_feather(FORECAST_DIR / "pred" / f"{STUDY_PRED[w]}.feather", columns=["pair", "date", col])
                d = d[d["pair"] == pair].dropna(subset=[col])
                s = pd.Series(d[col].to_numpy(float), index=pd.DatetimeIndex(d["date"]))
            except Exception:  # the study does not have this horizon: compute instead
                pass
        self._cache[key] = s
        return s

    def _save_cache(self, pair: str, w: int, s: pd.Series) -> None:
        p = self._cache_path(pair, w)
        p.parent.mkdir(parents=True, exist_ok=True)
        s = s.sort_index()
        pd.DataFrame({"date": s.index, "fc": s.to_numpy()}).to_feather(p)

    def _grid(self, dates: pd.DatetimeIndex, pair: str) -> np.ndarray:
        """Forecast candles: every STRIDE-th candle on a fixed date grid (aligned to the 038 study's
        grid for this pair when its predictions exist, so cached study forecasts are reused)."""
        k = ((dates - EPOCH) // BAR).to_numpy()
        phase = 0
        seed = self._load_cache(pair, self.WINDOWS[0])
        if len(seed):
            phase = int(((seed.index[0] - EPOCH) // BAR) % self.STRIDE)
        return (k % self.STRIDE) == phase

    # ------------------------------------------------------------------ freqtrade

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "pair")
        dates = pd.DatetimeIndex(dataframe["date"])
        lc = np.log(dataframe["close"].to_numpy(float))
        n_rows, n = len(lc), self.FORECAST_N
        on_grid = self._grid(dates, pair) & (np.arange(n_rows) >= max(self.WINDOWS) - 1)
        votes = np.zeros(n_rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for w in self.WINDOWS:
                cache = self._load_cache(pair, w)
                have = cache.reindex(dates).to_numpy(float).copy()
                todo = np.flatnonzero(on_grid & np.isnan(have))
                for t in todo:
                    have[t] = self._holt(lc[t - w + 1: t + 1], n) - lc[t]
                if len(todo):
                    new = pd.Series(have[todo], index=dates[todo])
                    cache = pd.concat([cache[~cache.index.isin(new.index)], new])
                    self._cache[(pair, w, n)] = cache
                    self._save_cache(pair, w, cache)
                fc_ret = pd.Series(np.where(on_grid, have, np.nan)).ffill().to_numpy()
                dataframe[f"fc_bp_{w}"] = fc_ret * 1e4
                # the forecast close, drawn at the candle it predicts (t + n)
                dataframe[f"fc_close_{w}"] = pd.Series(np.exp(lc + fc_ret)).shift(n).to_numpy()
                votes += np.nan_to_num(fc_ret > self.COST_LOG)
        dataframe["vote_count"] = votes
        dataframe["on_grid"] = on_grid
        dataframe["cost_bp"] = self.COST_LOG * 1e4
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = ((dataframe["vote_count"] >= self.VOTE_NEED) & dataframe["on_grid"]).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
