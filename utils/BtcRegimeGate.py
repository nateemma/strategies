# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0325, C0411, C0413
# type: ignore
# pylint: disable=import-error
"""
BtcRegimeGate — opt-in mixin that gates alt entries by BTC's market regime.

Motivated by the btc_corr study (Exp 4): alt longs are strongly BTC-regime
dependent — Weak-Up is the sweet spot, Sideways is a trap. This blocks new
entries while BTC is NOT in a favorable regime.

Favorable (default) = BTC uptrend AND trending:
    btc_close > EMA(btc_close, ema_len)  AND  ADX(btc, 14) >= adx_min

Reuses the standard cross-pair pattern: declare BTC in informative_pairs(),
pull it with dp.get_pair_dataframe() at the strategy timeframe, merge on date
(same timeframe -> no lookahead), and expose a boolean mask a strategy ANDs
into its entry conditions. Fail-open: if BTC data is unavailable the gate does
not filter (reverts to baseline behaviour).

Default is OFF (btc_gate_enable = False) so mixing it in is a no-op until a
concrete strategy opts in. NOTE: adx_min was calibrated on 1h data in the
study; re-tune it at the strategy's actual timeframe.
"""
import numpy as np
import pandas as pd
import talib

BTC_GATE_COLS = ["btcg_close", "btcg_ema50", "btcg_adx", "btcg_favorable"]


class BtcRegimeGate:

    btc_gate_enable: bool = False       # opt-in; no behaviour change when False
    btc_gate_mode: str = "uptrend"      # "uptrend" (require BTC up) | "crash" (block only BTC crashes)
    btc_gate_use_adx: bool = False      # (uptrend mode) add ADX trend-strength floor?
    btc_gate_adx_min: float = 20.0      # ADX floor when use_adx (re-tune per TF; 20 is 1h)
    btc_gate_ema_len: int = 50          # BTC trend EMA length
    # crash mode: block entries only when BTC is below trend AND has dropped
    # sharply (falling knife) — keeps ~all normal dip-buys, respects buy-low models.
    btc_crash_lookback: int = 96        # bars over which to measure the drop (24h @15m)
    btc_crash_drop: float = 0.05        # min log-drop over lookback to count as a crash

    @property
    def btc_pair(self) -> str:
        stake = self.config.get("stake_currency", "USDT") if hasattr(self, "config") else "USDT"
        return f"BTC/{stake}"

    def btc_informative_pairs(self):
        return [(self.btc_pair, self.timeframe)]

    def add_btc_regime(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Merge BTC regime columns onto the (same-timeframe) alt dataframe."""
        if not self.btc_gate_enable:
            return dataframe

        btc = self.dp.get_pair_dataframe(self.btc_pair, self.timeframe)
        if btc is None or len(btc) == 0:
            dataframe["btcg_favorable"] = 1        # fail-open
            return dataframe

        c = btc["close"].astype(float).values
        h = btc["high"].astype(float).values
        l = btc["low"].astype(float).values
        feat = btc[["date"]].copy()          # keep tz-aware date to match the alt df
        feat["btcg_close"] = c
        feat["btcg_ema50"] = talib.EMA(c, self.btc_gate_ema_len)
        feat["btcg_adx"] = talib.ADX(h, l, c, 14)
        dataframe = dataframe.merge(feat, on="date", how="left")
        for col in ("btcg_close", "btcg_ema50", "btcg_adx"):
            dataframe[col] = dataframe[col].ffill()

        if self.btc_gate_mode == "crash":
            # block only genuine BTC crashes: below trend AND a sharp recent drop.
            ret = np.log(dataframe["btcg_close"]) - np.log(
                dataframe["btcg_close"].shift(self.btc_crash_lookback)
            )
            crash = (dataframe["btcg_close"] < dataframe["btcg_ema50"]) & (
                ret < -self.btc_crash_drop
            )
            fav = ~crash.fillna(False)
        else:                                                        # "uptrend"
            fav = dataframe["btcg_close"] > dataframe["btcg_ema50"]
            if self.btc_gate_use_adx:
                fav = fav & (dataframe["btcg_adx"] >= self.btc_gate_adx_min)
        dataframe["btcg_favorable"] = fav.fillna(False).astype(int)
        return dataframe

    def btc_favorable_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        if not self.btc_gate_enable or "btcg_favorable" not in dataframe.columns:
            return pd.Series(True, index=dataframe.index)   # no filtering
        return dataframe["btcg_favorable"] > 0
