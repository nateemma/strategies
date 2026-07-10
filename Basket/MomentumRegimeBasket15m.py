"""MomentumRegimeBasket15m — 15m-data / hourly-rebalance momentum with ACCUMULATING fills.

*** RESEARCH ARTIFACT (2026-07) — the real-execution test of a vectorized finding. ***

Same signal as MomentumRegimeBasket (cross-sectional top-N momentum on a 90-day
lookback + BTC>SMA100 daily regime, long-only spot), plus a per-coin trend filter
(the drawdown fix — see TREND_FILTER_ENABLE), run on 15m candles and rebalanced
hourly. The point of this variant is to test — with freqtrade's real next-candle
fills — a vectorized result that overturned an earlier "wall":

  Vectorized (conservative VWAP fills, $50k, per-year contributions):
    CORE-20  full +223% [2024 +98/2025 +90/2026 +35],  ex-ZEC +158%
    BROAD-77 full +155% [2024 +43/2025 +83/2026 +29],  ex-ZEC +116%
  i.e. fast execution is NOT catastrophic (an earlier "-17%" was a fill-model
  artifact) and is diversified BEYOND ZEC (positive every year ex-ZEC).

*** THE KEY MECHANIC — why this isn't just "the daily strategy on 15m". ***
A single next-candle fill capped to one 15m candle's liquidity captures almost
nothing (that IS the -17% failure). The edge only survives because you ACCUMULATE:
a 90-day ranking is sticky, so a coin stays in the top-N for many candles, and you
fill a little each candle (<=10% of that candle's quote volume) until the position
reaches its equal-weight target. That is implemented via position adjustment
(position_adjustment_enable + adjust_trade_position adds toward target every candle),
NOT a single entry. Phantom-fill protection (custom_stake_amount caps every add to
<=10% of candle quote volume; confirm_trade_entry rejects dust) runs in backtest too,
so fills are realistic. Full exit (populate_exit_trend) when a coin leaves the top-N
or the regime turns risk-off; a runaway winner is partial-trimmed back toward the
equal-weight cap (see MAX_POSITION_WEIGHT — a return/risk-adjusted improvement).

*** CAVEATS (unchanged from the vectorized study) ***
  - SURVIVORSHIP BIAS inflates the MAGNITUDE (dead pump-and-die coins are absent,
    worst for the broad meme set). Trust the SIGN + multi-year robustness, not the %.
  - Short ~2yr / one-cycle sample.
  - This freqtrade run is the honest execution check; divergence from the vectorized
    numbers is expected (order pricing, fee accounting, one-add-per-candle cadence).

*** lookahead-analysis reports "bias detected" — it is a PROVEN FALSE POSITIVE. ***
`freqtrade lookahead-analysis` detects bias by re-running on cut timeranges, but this
strategy reads the daily feathers DIRECTLY off disk (_daily_closes, the workaround for
the startup-candle cap), bypassing the DataProvider the tool truncates — so it can't
reason about the inputs and flags heuristically (a documented limitation for external-
data / cross-sectional strategies). The `hold` signal is CAUSAL by construction
(momentum = current 15m close / Pd.shift(1).shift(90); regime + trend off Pd.shift(1);
membership floored to the hour, all ffill-mapped) and this was VERIFIED empirically: a
truncation-invariance test (recompute `hold` with future data removed) found ZERO
changed cells across 76,867 candles x 75 pairs at 4 cut points, in BOTH the cut-all and
the freqtrade-exact (daily-full / 15m-cut) scenarios. Test: /tmp/bias_check.py.

Config: config/config_mom_15m.json (max_open_trades == TOP_N, stake "unlimited").
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

# Daily OHLCV feathers — read directly for the 90d ranking + 100d regime SMA, which
# need ~100 days of history that freqtrade's 15m warmup (capped at 5x candle limit,
# ~52 days) cannot supply. Same direct-feather pattern as Funding/FundingCarry.
FEATHER_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "binanceus"


class MomentumRegimeBasket15m(IStrategy):
    timeframe = "15m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200   # 15m frames only need current price; history comes from daily feathers
    stoploss = -0.99          # rotation is via signals, not stops
    minimal_roi = {"0": 100}  # ROI off
    trailing_stop = False
    use_exit_signal = True
    position_adjustment_enable = True   # REQUIRED — accumulate fills toward target

    MOM_LOOKBACK_DAYS = 90   # trailing-return window (daily close 90d ago is the reference)
    TOP_N = 3                # == config max_open_trades
    REGIME_SMA = 100         # BTC trend window, in DAILY candles
    REGIME_REF = "BTC/USDT"
    REBALANCE_HOURLY = True  # only change top-N membership on the hour (matches the test)

    # Per-coin trend filter — the drawdown fix. The BTC>SMA100 regime is a RISK-ON
    # gate that doesn't protect against alt-specific bleeds (the 52% drawdown accrued
    # while BTC held above its SMA100). Requiring each held coin to be above its OWN
    # daily SMA drops it as soon as it rolls over — exits faders, refuses freshly
    # dumping pumps, and holds <TOP_N (more cash) when few coins trend. Vectorized:
    # cuts maxDD ~40%->23% while RAISING return (it removes losing tail trades).
    TREND_FILTER_ENABLE = True
    PER_COIN_SMA = 50        # a held coin must be above its own DAILY SMA(this)

    # Max-position-weight cap — trim a runaway winner back toward this fraction of the
    # portfolio (equal weight is 1/TOP_N ~= 0.33). Banks the excess into cash so a
    # retracing winner has less at risk, attacking the UNREALIZED give-back (wallet DD
    # > closed DD). 0.0 = off.
    MAX_POSITION_WEIGHT = 0.45

    # liquidity-aware sizing (same discipline as FundingCarry / the NN family)
    MIN_QUOTE_VOLUME = 1000
    QUOTE_VOLUME_HEADROOM_MULT = 10.0   # fill <= 1/10 of a candle's quote volume

    _xs = None       # cached membership matrix (bool DataFrame, per pair)
    _xs_key = None   # cache key: (latest candle date, whitelist)

    def _daily_closes(self, pairs) -> DataFrame:
        """Full-history daily close panel, read straight from the feathers."""
        out = {}
        for p in pairs:
            f = FEATHER_DIR / f"{p.split('/')[0]}_USDT-1d.feather"
            if f.exists():
                d = pd.read_feather(f)
                d["date"] = pd.to_datetime(d["date"], utc=True)
                out[p] = d.set_index("date")["close"]
        return pd.DataFrame(out).sort_index()

    def _compute_xs(self) -> DataFrame:
        """Causal top-N membership AND-ed with BTC daily risk-on, per 15m date.

        Momentum = current 15m close / daily close 90d ago (intraday-responsive, so a
        coin pumping mid-day can enter the top-N that hour — the "catch fast pumps"
        edge). Regime = daily SMA100 on BTC. Both daily inputs are lagged one day
        (yesterday's close is what's known intraday) => causal, no lookahead.
        Membership floored to the hour so the basket rebalances hourly, not every 15m.
        Cached on (latest 15m date, whitelist).
        """
        wl = tuple(sorted(self.dp.current_whitelist()))
        ref = self.dp.get_pair_dataframe(self.REGIME_REF, self.timeframe)
        asof = ref["date"].iloc[-1] if ref is not None and len(ref) else None
        key = (asof, wl)
        if self._xs is not None and self._xs_key == key:
            return self._xs

        # --- daily inputs (full history from disk), lagged 1 day to stay causal ---
        Pd = self._daily_closes(wl)
        known = Pd.shift(1)                                        # yesterday's close, known intraday
        ref90 = known.shift(self.MOM_LOOKBACK_DAYS)               # daily close ~90d ago
        btc_d = known.get(self.REGIME_REF)
        if btc_d is not None:
            ron_d = (btc_d > btc_d.rolling(self.REGIME_SMA).mean())
        else:
            ron_d = pd.Series(True, index=Pd.index)

        # --- current 15m close panel (freqtrade-loaded) ---
        closes = {}
        for p in wl:
            df = self.dp.get_pair_dataframe(p, self.timeframe)
            if df is not None and len(df):
                s = df.copy(); s["date"] = pd.to_datetime(s["date"], utc=True)
                closes[p] = s.set_index("date")["close"]
        P15 = pd.DataFrame(closes).sort_index()

        # map daily inputs onto the 15m index (ffill = as-of the latest known day)
        ref90_15 = ref90.reindex(columns=P15.columns).reindex(P15.index, method="ffill")
        risk_on = ron_d.reindex(P15.index, method="ffill").fillna(False)
        mom = P15 / ref90_15 - 1                                  # intraday-responsive 90d momentum
        member = mom.rank(axis=1, ascending=False, method="first") <= self.TOP_N
        if self.TREND_FILTER_ENABLE:                              # drop coins below their own trend
            trend_ok = (known > known.rolling(self.PER_COIN_SMA).mean())
            trend_ok_15 = trend_ok.reindex(columns=P15.columns).reindex(P15.index, method="ffill").fillna(False)
            member = member & trend_ok_15
        want = member.apply(lambda col: col & risk_on)            # bool DataFrame
        if self.REBALANCE_HOURLY:
            hourly = want[want.index.minute == 0]                 # decision at each :00
            want = hourly.reindex(want.index, method="ffill").fillna(False)
        self._xs = want
        self._xs_key = key
        return self._xs

    def _hold_flag(self, pair: str, dates: pd.Series) -> pd.Series:
        want = self._compute_xs()
        w = want[pair] if pair in want.columns else pd.Series(False, index=want.index)
        left = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)})
        m = pd.merge_asof(left.sort_values("date"),
                          w.rename("hold").reset_index().rename(columns={"index": "date"}).sort_values("date"),
                          on="date", direction="backward")
        return m["hold"].fillna(False).astype(bool)

    def _portfolio_value(self) -> float:
        """Mark-to-market total = free cash + Σ open position values (for the
        equal-weight target). Present-state only (current-candle closes)."""
        stake_ccy = self.config["stake_currency"]
        pv = self.wallets.get_free(stake_ccy)
        for ot in Trade.get_trades_proxy(is_open=True):
            df, _ = self.dp.get_analyzed_dataframe(ot.pair, self.timeframe)
            price = df["close"].iloc[-1] if df is not None and len(df) else ot.open_rate
            pv += ot.amount * price
        return pv

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["hold"] = self._hold_flag(metadata["pair"], dataframe["date"]).values
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["hold"] & (dataframe["volume"] > 0), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[~dataframe["hold"], "exit_long"] = 1   # exit when out of top-N or risk-off
        return dataframe

    # --- liquidity-aware sizing: cap the INITIAL fill to the equal-weight target
    #     AND to <=10% of the candle's quote volume ---
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return proposed_stake
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last = df.iloc[-1].squeeze()
        fillable = (last["volume"] * last["close"]) / self.QUOTE_VOLUME_HEADROOM_MULT
        target = self._portfolio_value() / self.TOP_N
        stake = min(proposed_stake, target, fillable, max_stake)
        if min_stake and stake < min_stake:
            return 0.0
        return max(stake, 0.0)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return True
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last = df.iloc[-1].squeeze()
        return (last["volume"] * last["close"]) >= self.MIN_QUOTE_VOLUME   # reject dust

    # --- ACCUMULATION: add toward the equal-weight target each candle, capped to
    #     <=10% of the candle's quote volume, while the coin is still in the basket ---
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return None
        df, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if df is None or len(df) == 0:
            return None
        last = df.iloc[-1].squeeze()
        pv = self._portfolio_value()
        current_value = trade.amount * current_rate
        # max-position-weight cap: trim a runaway winner first (banks profit into cash)
        if self.MAX_POSITION_WEIGHT and pv > 0 and current_value > self.MAX_POSITION_WEIGHT * pv:
            trim = self.MAX_POSITION_WEIGHT * pv - current_value   # negative => reduce
            if not min_stake or abs(trim) >= min_stake:
                return trim
        if not bool(last["hold"]):
            return None   # leaving the basket -> full exit is handled by the exit signal
        target = pv / self.TOP_N
        if current_value >= target * 0.98:
            return None   # already at target weight
        fillable = (last["volume"] * last["close"]) / self.QUOTE_VOLUME_HEADROOM_MULT
        add = min(target - current_value, fillable, max_stake)
        if add <= 0 or (min_stake and add < min_stake):
            return None
        return add
