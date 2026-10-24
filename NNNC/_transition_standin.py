# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""The 040 regime-transition stand-in, computed INSIDE the strategy (backtest and live alike).

Walk-forward first read the stand-in's predictions from per-fold tables (research/nnmt/transition.py
screen). A default must also run live, so this module rebuilds the stand-in's inputs from what a bot has
and scores the fitted models itself:

  base   the strategy's own 24 indicator columns (the deployed model's inputs, = the screen's `base:`)
  raw8   research.clfstudy.data.raw8's eight columns on the pair's candles (96-bar windows: fit in the
         ~920-bar live buffer)
  btc    the same raw8 on BTC/USDT (an informative pair), joined by date
  z      z_L of the pair's and of BTC's close, L in STATE_LENGTHS, with a 2880-bar volatility window --
         longer than the live buffer, so these come from utils/CloseHistoryStore (local files + live
         candles), never from the dataframe alone

A bar is scored only if it is TRADEABLE, as in the screen: >= 95 % of its trailing 1000 bars are real
candles (the store's presence flag) and its own volume is > 0. Elsewhere the probability is NaN and the
scaler leaves the stake unchanged.

The models are fit per cutoff by `research.nnmt.transition fit` (the screen's fit rule) and saved under
saved_data/transition/<YYYYMMDD>/: <cell>.joblib and meta.json (feature order, tail percentiles). A
fold class uses its MODEL_TRAIN_TIMERANGE end as the cutoff; a live class sets TR_STANDIN_CUTOFF. A
missing artifact raises: an unfitted stand-in is a deployment error, not a neutral signal.

Constants mirror research/nnmt/preregistration_transition.json (tests/test_transition_standin.py
asserts it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[1]
ARTIFACTS = STRAT / "saved_data" / "transition"
CONTEXT_PAIR = "BTC/USDT"

INDICATORS = ("adx_scaled", "aroonosc_scaled", "atr_norm", "bb_position", "bb_width", "cci_scaled",
              "di_diff_scaled", "ema_fast_norm", "fast_diff", "fastk_scaled", "fisher_ss", "cg_ss", "gain_norm",
              "guard_metric_pos", "guard_metric_neg", "macd_pos", "macd_neg", "macdhist_norm", "mfi_scaled",
              "rsi_scaled", "sar_ratio", "spread_ma", "vwap_pos", "vwap_neg")
RAW8 = ("ret1", "ret4", "log_hc", "log_lc", "log_oc", "range", "logvol_z", "vol_ratio")
RAW8_WINDOW = 96
STATE_LENGTHS = (24, 72, 96, 168, 288, 672)
SIGMA_WINDOW, SIGMA_MIN = 2880, 960
WARMUP_BARS, DATA_OK_SHARE = 1000, 0.95


def feature_names() -> list[str]:
    """The screen's column order (research.nnmt.transition._features)."""
    own = [f"base:{c}" for c in INDICATORS] + [f"raw8:{c}" for c in RAW8]
    btc = [f"btc:raw8:{c}" for c in RAW8]
    z = [n for L in STATE_LENGTHS for n in (f"z_pair_{L}", f"z_btc_{L}")]
    return own + btc + z


def raw8(df: pd.DataFrame) -> pd.DataFrame:
    """research.clfstudy.data.raw8 (the screen's raw8), on date/OHLCV columns."""
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["ret1"] = np.log(close / close.shift(1))
    out["ret4"] = np.log(close / close.shift(4))
    out["log_hc"] = np.log(df["high"].astype(float) / close)
    out["log_lc"] = np.log(df["low"].astype(float) / close)
    out["log_oc"] = np.log(df["open"].astype(float) / close)
    out["range"] = (df["high"].astype(float) - df["low"].astype(float)) / close
    vol = df["volume"].astype(float)
    logvol = np.log(vol + 1.0)
    out["logvol_z"] = (logvol - logvol.rolling(RAW8_WINDOW, min_periods=RAW8_WINDOW).mean()) / \
        logvol.rolling(RAW8_WINDOW, min_periods=RAW8_WINDOW).std()
    out["vol_ratio"] = np.log((vol + 1.0) / (vol.rolling(RAW8_WINDOW, min_periods=RAW8_WINDOW).mean() + 1.0))
    return out[list(RAW8)].fillna(0.0)


def zscore(close: pd.Series, L: int) -> pd.Series:
    """research.nnmt.transition.zscore with the pre-registered sigma window."""
    lc = np.log(close.astype(float))
    sma = close.astype(float).rolling(L, min_periods=L).mean()
    sig = lc.diff().rolling(SIGMA_WINDOW, min_periods=SIGMA_MIN).std()
    return (lc - np.log(sma)) / (sig * np.sqrt(L))


def data_ok(has: pd.Series) -> pd.Series:
    """harvest.data_flags' data_ok: the bar is real and >= DATA_OK_SHARE of the trailing WARMUP_BARS are."""
    share = has.astype(float).rolling(WARMUP_BARS, min_periods=WARMUP_BARS).mean()
    return (has > 0) & (share >= DATA_OK_SHARE)


def features(dataframe: pd.DataFrame, btc: pd.DataFrame, hist_pair: pd.DataFrame,
             hist_btc: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """(X in feature_names() order, tradeable mask), row-aligned with `dataframe`."""
    dates = pd.to_datetime(dataframe["date"], utc=True)
    X = pd.DataFrame(index=dataframe.index)
    for c in INDICATORS:
        X[f"base:{c}"] = dataframe[c].astype(float).to_numpy()
    r = raw8(dataframe)
    for c in RAW8:
        X[f"raw8:{c}"] = r[c].to_numpy()
    rb = raw8(btc).set_index(pd.to_datetime(btc["date"], utc=True))
    for c in RAW8:
        X[f"btc:raw8:{c}"] = dates.map(rb[c]).astype(float).to_numpy()
    hp = hist_pair.set_index("date")
    hb = hist_btc.set_index("date")
    for L in STATE_LENGTHS:
        X[f"z_pair_{L}"] = dates.map(zscore(hp["close"], L)).astype(float).to_numpy()
        X[f"z_btc_{L}"] = dates.map(zscore(hb["close"], L)).astype(float).to_numpy()
    ok = dates.map(data_ok(hp["has"])).fillna(False).to_numpy(bool)
    tradeable = ok & (dataframe["volume"].to_numpy(float) > 0)
    return X[feature_names()], tradeable


class StandIn:
    """The fitted stand-in for one cutoff: per cell a classifier and its tail percentiles."""

    def __init__(self, cutoff: str, root: Path = ARTIFACTS):
        d = Path(root) / cutoff
        meta_p = d / "meta.json"
        if not meta_p.is_file():
            raise FileNotFoundError(f"transition stand-in not fitted for cutoff {cutoff}: {meta_p} "
                                    f"(run: research.nnmt.transition fit --cutoff {cutoff})")
        import joblib

        self.meta = json.loads(meta_p.read_text())
        if self.meta["features"] != feature_names():
            raise ValueError(f"{meta_p}: feature order differs from the strategy's")
        self.models = {c: joblib.load(d / f"{c}.joblib") for c in self.meta["cells"]}

    def threshold(self, cell: str, q: float) -> float:
        return float(self.meta["tail_percentiles"][cell][str(int(q))])

    def predict(self, cell: str, X: pd.DataFrame, tradeable: np.ndarray) -> np.ndarray:
        p = np.full(len(X), np.nan)
        if tradeable.any():
            p[tradeable] = self.models[cell].predict_proba(X.to_numpy(np.float32)[tradeable])[:, 1]
        return p
