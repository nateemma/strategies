"""MomentumRegimeBasket — cross-sectional TOP-N momentum + BTC-regime circuit-breaker.

*** RESEARCH ARTIFACT (2026-07) — NOT deploy-ready. The apparent edge is a
concentrated, illiquid PUMP LOTTERY, not a robust diversified strategy. Read the
caveats before running this with real money. ***

Hold an EQUAL-WEIGHT basket of the top-N trailing-return coins WHILE BTC is in an
uptrend (risk-on); go fully to CASH when BTC is below its trend (risk-off). Long-only
spot, DISCRETE signal-based rotation. Design choices (validated): (1) DISCRETE top-N
concentration (N=2-3 — N>=5 dilutes the edge, Sharpe ~1.3->0.8); (2) BTC-REGIME cash
gate, which flips the 2026 bear from ~-40% to positive and ~halves max drawdown.

Headline: 20-coin universe reproduces frictionless research under real execution
(+125% / Sharpe 1.17). BUT on a broad 75-coin universe the frictionless +329%
collapses to ~+59% in freqtrade — the meme-coin pumps that drove it are
UN-EXECUTABLE (daily next-candle fills miss a coin that pumps 100% in a day), and
what survives is still ZEC-dominated. So the scrutiny below dismantles the headline.

*** CAVEATS (why this is a research artifact, not a strategy) ***
  - RETURN CONCENTRATION: the entire profit is ~2 MONTHS (Oct 2025 + Jun 2026 = 123%
    of profit; the other 17 months net NEGATIVE). Positive-skew momentum on n~=2
    payoff events => magnitude AND sign are fragile.
  - IT'S ZEC: remove ZEC and the recent-period edge collapses +120% -> -13%. The
    "cross-sectional momentum edge" is really ZEC-momentum + a regime filter. Same
    ZEC-dependence as every other strategy family in this codebase.
  - BROADENING (75-coin universe) de-concentrates it (ZEC 82%->31%) but does NOT
    improve it: no more profit, deeper DD (-52%), WORSE realistic-fill capture
    (40-60% kept vs 66-100% at 20 coins — new contributors are thin memes), and
    WORSE survivorship (meme pumpers are pump-and-SURVIVE listings).
  - SAMPLE: ~2yr / one bull cycle / SURVIVOR universe. Un-validatable to the degree
    a real strategy needs.
  If run at all: tiny convex sleeve, size for a pump-lottery payoff distribution,
  expect long bleeds. See us_spot_market_study.md + project_xsectional_momentum_regime.

Config: max_open_trades == TOP_N, stake_amount "unlimited" (freqtrade splits capital
equally across the open slots => equal-weight top-N).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class MomentumRegimeBasket(IStrategy):
    timeframe = "1d"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 100   # momentum needs 90; higher excludes recent listings
    stoploss = -0.99          # rotation is via signals, not stops
    minimal_roi = {"0": 100}  # ROI off
    trailing_stop = False
    use_exit_signal = True

    MOM_LOOKBACK = 90    # trailing-return window (days)
    TOP_N = 3            # concentration — keep 2-3 (== config max_open_trades)
    REGIME_SMA = 100     # BTC trend window (days)
    REGIME_REF = "BTC/USDT"

    _xs = None       # cached (top-N membership matrix, risk-on series)
    _xs_key = None   # cache key: (latest candle date, whitelist) — refresh when either changes

    def _compute_xs(self):
        """Causal cross-sectional top-N membership + BTC risk-on, per date.

        Cache keyed on (latest candle date, whitelist) so it recomputes as new
        candles arrive AND when a dynamic pairlist changes the universe. In
        backtest the latest date is constant (full history present up front) so
        this computes once; in live/dry-run it refreshes every new candle.
        """
        wl = tuple(sorted(self.dp.current_whitelist()))
        ref = self.dp.get_pair_dataframe(self.REGIME_REF, self.timeframe)
        asof = ref["date"].iloc[-1] if ref is not None and len(ref) else None
        key = (asof, wl)
        if self._xs is not None and self._xs_key == key:
            return self._xs
        closes = {}
        for p in wl:
            df = self.dp.get_pair_dataframe(p, self.timeframe)
            if df is not None and len(df):
                s = df.copy(); s["date"] = pd.to_datetime(s["date"], utc=True)
                closes[p] = s.set_index("date")["close"]
        P = pd.DataFrame(closes).sort_index()
        mom = P.pct_change(self.MOM_LOOKBACK)                      # causal trailing return
        member = mom.rank(axis=1, ascending=False, method="first") <= self.TOP_N
        btc = P.get(self.REGIME_REF)
        risk_on = (btc > btc.rolling(self.REGIME_SMA).mean()) if btc is not None \
            else pd.Series(True, index=P.index)
        self._xs = (member, risk_on.rename("risk_on"))
        self._xs_key = key
        return self._xs

    def _hold_flag(self, pair: str, dates: pd.Series) -> pd.Series:
        member, risk_on = self._compute_xs()
        want = (member[pair] & risk_on) if pair in member.columns else pd.Series(False, index=member.index)
        left = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)})
        m = pd.merge_asof(left.sort_values("date"),
                          want.rename("hold").reset_index().rename(columns={"index": "date"}).sort_values("date"),
                          on="date", direction="backward")
        return m["hold"].fillna(False).astype(bool)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["hold"] = self._hold_flag(metadata["pair"], dataframe["date"]).values
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["hold"] & (dataframe["volume"] > 0), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[~dataframe["hold"], "exit_long"] = 1   # exit when out of top-N or risk-off
        return dataframe
