# type: ignore
# pylint: disable=import-error
"""FundingCarry — slow long-only funding-reversion strategy (Study 7 spin-off).

*** RESEARCH ARTIFACT (2026-07) — NOT deploy-ready as a standalone edge. ***
The funding-reversion signal is REAL and era-persistent, but with realistic
(liquidity-aware) fills it is MARGINAL on Binance.US — see "Reality check" below.
Kept fully built out (validated exit / circuit-breaker / horizon, live funding
sidecar, phantom-fill sizing) so it's ready if a more-liquid venue or futures
access ever becomes available.

Study 7 found a real, era-persistent funding reversal edge (low/negative funding =
shorts crowded -> price squeezes UP over 8-24h; +0.6..0.8pp extreme-decile fwd-24h
spread stable across 2024/25/26). The fast gbb NN strategy can't harvest it (exits
in 2h, before the squeeze). This strategy is built AT the funding horizon: enter
long when funding is extreme-NEGATIVE, hold to the funding horizon, exit.

Design (validated in-sample + across 3 non-overlapping eras):
  - Entry: trailing funding z-score < ENTRY_Z (crowded shorts).
  - Exit: PATIENT pure-time-exit at MAX_HOLD_H=16h (ROI + stop OFF). A stop whipsaws
    this slow mean-reversion — a tight ROI/stop turned +30% into -90%. 12-24h is a
    robust plateau; >=36h degrades hard.
  - Down-market CIRCUIT-BREAKER: only enter when BTC > SMA(MARKET_SMA=500, ~21d).
    Long-only spot inherits market beta over the hold, so it bleeds in a sustained
    bear; the ~21d trend breaker sits it out. (Drawdown-from-high / fast-EMA gates
    did NOT work — too laggy / too whippy.)
  - Phantom-fill sizing (custom_stake_amount + confirm_trade_entry): cap orders to
    <=10% of candle quote volume, reject dust — so the backtest books realistic fills.

*** Reality check (the important part): ~73% of the raw-backtest trades are
un-fillable on Binance.US. With phantom-fill sizing ON, the full-period result is
+6.6% / Sharpe 0.89 over 2024-2026 (down from a PHANTOM +23.7%), and only the
2024-25 era stays positive (flat-to-neg since). The funding signal fires in ILLIQUID
moments (crowded-short = thin grind-downs), so the biggest apparent edges (ZEC
+10.3%->+2.5%) were the most phantom. Same "edge lives where you can't cheaply trade"
wall as the cross-sectional-reversion and spread studies. Real edge, marginal capture
on US spot.

Long-only spot captures only the low-funding leg; the clean market-neutral long/short
spread would hedge beta AND be far more capturable on a liquid venue, but requires
shorting (perps/futures) US spot can't do. Rule-based, no NN. Funding history from
Binance Data Vision, refreshed live from OKX via user_data/strategies/scripts/refresh_funding.py
(user_data/data/funding/<PAIR>_funding.feather; last SETTLED rate, causal).
"""
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)
FUND_DIR = Path(__file__).parent.parent.parent / "data" / "funding"


class FundingCarry(IStrategy):
    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 550   # cover the MARKET_SMA window

    # --- exit stack: patient pure-time-exit tuned to the 8-24h funding horizon ---
    minimal_roi = {"0": 10.0}    # OFF — patient pure-time-exit is the validated design
    stoploss = -0.99             # OFF — a stop whipsaws this slow mean-reversion (tight stop => -90%)
    use_custom_stoploss = False
    trailing_stop = False

    # --- tunables ---
    ENTRY_Z = -1.5       # enter long when trailing funding z-score < this (crowded shorts)
    FUND_WIN = 90        # trailing settlements (~30d at 8h) for the z-score
    MAX_HOLD_H = 16      # time-exit cap; 12-24h is a robust plateau (best Sharpe at 16h,
                         #   lowest DD at 12h, max return at 24h; >=36h degrades hard)
    BREAKER_ENABLE = True
    MARKET_SMA = 500     # BTC ~21d SMA — down-market circuit-breaker

    # --- liquidity-aware sizing (ported from the NN family: reduce a thin-candle
    #     order to <=10% of the candle's quote volume; reject dust pairs). Runs in
    #     backtest too, for live-parity — so the backtest doesn't book phantom fills. ---
    MIN_QUOTE_VOLUME = 1000
    QUOTE_VOLUME_HEADROOM_MULT = 10.0

    def _funding_z(self, pair: str, dates: pd.Series) -> pd.Series:
        base = pair.split("/")[0]
        fpath = FUND_DIR / f"{base}_funding.feather"
        if not fpath.exists():
            return pd.Series(0.0, index=dates.index)
        f = pd.read_feather(fpath)
        f["dt"] = pd.to_datetime(f["dt"], utc=True)
        f = f.sort_values("dt").reset_index(drop=True)
        m = f["funding"].rolling(self.FUND_WIN, min_periods=10).mean()
        s = f["funding"].rolling(self.FUND_WIN, min_periods=10).std()
        f["fz"] = ((f["funding"] - m) / s).fillna(0.0)
        left = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)})
        merged = pd.merge_asof(
            left.sort_values("date"),
            f[["dt", "fz"]].rename(columns={"dt": "date"}).sort_values("date"),
            on="date", direction="backward",
        )
        return merged["fz"].fillna(0.0)

    def informative_pairs(self):
        return [("BTC/USDT", self.timeframe)] if self.BREAKER_ENABLE else []

    def _market_risk_on(self, dataframe: pd.DataFrame) -> pd.Series:
        if not self.BREAKER_ENABLE:
            return pd.Series(1, index=dataframe.index)
        btc = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        if btc is None or len(btc) == 0:
            return pd.Series(1, index=dataframe.index)
        btc = btc.copy()
        btc["risk_on"] = (
            btc["close"] > btc["close"].rolling(self.MARKET_SMA, min_periods=24).mean()
        ).astype(int)
        btc["date"] = pd.to_datetime(btc["date"], utc=True)
        left = dataframe.copy(); left["date"] = pd.to_datetime(left["date"], utc=True)
        m = pd.merge_asof(
            left[["date"]].sort_values("date"),
            btc[["date", "risk_on"]].sort_values("date"),
            on="date", direction="backward",
        )
        return pd.Series(m["risk_on"].fillna(1).values, index=dataframe.index)

    def populate_indicators(self, dataframe, metadata):
        dataframe["funding_z"] = self._funding_z(metadata["pair"], dataframe["date"]).values
        dataframe["risk_on"] = self._market_risk_on(dataframe).values
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe["funding_z"] < self.ENTRY_Z)
            & (dataframe["risk_on"] == 1)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe  # exits via time (custom_exit); ROI/stop off

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        held_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if held_h >= self.MAX_HOLD_H:
            return "time_exit"
        return None

    # --- phantom-fill protection (ported from Framework/BaseStrategy) ---
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        """Cap the stake at <=1/HEADROOM of the candle's quote volume so the order
        doesn't dominate a thin candle."""
        if self.dp.runmode.value in ("plot", "other"):
            return proposed_stake
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last = df.iloc[-1].squeeze()
        quote_volume = last["volume"] * last["close"]
        fillable = quote_volume / self.QUOTE_VOLUME_HEADROOM_MULT
        return min(proposed_stake, fillable)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        """Reject if the candle can't absorb the (already-reduced) order with
        headroom, or if it's a dust pair below MIN_QUOTE_VOLUME."""
        if self.dp.runmode.value in ("plot", "other"):
            return True
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last = df.iloc[-1].squeeze()
        quote_volume = last["volume"] * last["close"]
        required = max(self.MIN_QUOTE_VOLUME, self.QUOTE_VOLUME_HEADROOM_MULT * amount * rate)
        if quote_volume < required:
            logger.info("FundingCarry reject %s: quote_vol %.0f < required %.0f",
                        pair, quote_volume, required)
            return False
        return True
