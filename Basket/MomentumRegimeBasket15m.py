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

*** MEASURED (freqtrade, 15mFast, 2024-08-31..2026-08-20, config_mom_15m.json) ***
  596 trades, win rate 32.7%, +444.20% vs market -20.61%, PF 1.64, CAGR 136%.
  Sharpe 1.28 / Calmar 11.48 (daily wallet balance). MAX DRAWDOWN 61.68% (wallet;
  58.48% closed) over ~152 days, Dec-2024 -> May-2025.

  NOTE the drawdown. It is consistent with the earlier persistence study, which
  also measured ~58% closed on the full window -- the "26% DD" in commit a9093ac
  refers specifically to the LATE window (W2, 2025-06->2026-08), where Fast doubled
  base's return at equal drawdown. Do not read that 26% as full-period risk: the
  deep drawdown sits in the 2024 window, and over the whole sample it is 58-62%.

  Diversification DOES hold here, unlike on the 11-pair config.json: ZEC is 28% of
  net (not 81%), and removing ANY single pair still leaves $33-39k of $44k at
  PF~1.5. Net by year 2024 $15.8k / 2025 $13.2k / 2026 $15.4k.

  BUT the profit is tail-carried at the TRADE level: the top 5 trades of 596 are
  107% of net, i.e. trades 6..596 are collectively negative, and the top 3 PAIRS
  (ZEC/PENGU/TROLL) are 76%. Median win +1.22% vs median loss -1.51% -- the typical
  winner is SMALLER than the typical loser; the payoff ratio (2.75 vs 2.12 needed at
  this win rate) comes entirely from the right tail. Judge this strategy on that
  tail surviving, not on the trade count.

*** EXIT_RANK_N (exit hysteresis) -- SWEPT, NOT YET DEFAULTED ***
  Diagnosis: bleed is a HOLDING-PERIOD effect. On 15mFast every duration bucket
  under 24h loses (<2h -$5.7k / 2-6h -$3.3k / 6-24h -$10.3k, 467 of 596 trades),
  while the 32 trades held >7d make +$57.7k. With entry and exit sharing one
  threshold, rank oscillation across TOP_N is itself the churn.

  Sweep (15mFast, MULT=10, FILL_VOLUME_LAG=1). ex-top5 = net excluding the 5
  biggest trades, i.e. whether the BODY pays for itself:

    N     trades  net%    PF   maxDD  ret/DD   ex-top5   illiq%
    3(=)     596  444.2  1.64  58.5%   7.60    -3,425     55%
    4        264  551.8  1.92  57.2%   9.64    +2,953     48%
    5        196  388.8  1.84  57.9%   6.71      +903     51%
    6        156  711.0  2.68  43.7%  16.28   +12,804     46%
    7        140  625.3  2.64  34.0%  18.42    +8,616     43%
    8        133  554.6  2.43  33.4%  16.63    +6,130     43%
    9        120  714.8  2.83  32.6%  21.94   +14,381     41%
    11       111  661.1  3.13  36.1%  18.29   +14,893     43%
    15       102  508.7  2.78  45.3%  11.24    +4,806     59%

  PLATEAU 6..11 (not a spike): PF 2.4-3.1, ret/DD 16-22, all years positive,
  trades/duration/win-rate all monotonic in N. N=5 is an isolated dip between two
  strong neighbours -- noise. Falls off at 15 (DD back up, illiquid share 59%).
  Pick from the MIDDLE of the plateau (~8-9), never the peak.

  Not a liquidity artifact: at MULT=50, N=7 gives 411.9% / PF 2.47 / DD 27.0% vs
  N=3's 291.5% / PF 1.59 / DD 44.8% -- still better on every axis.

  DEFAULT REMAINS None (no hysteresis). This is ONE in-sample window; per the
  house rule an edge must show in BOTH halves of a persistence split before it is
  promoted. Run W1/W2 the way MOM_LOOKBACK_DAYS=14 was validated, then flip.

*** CAVEATS (unchanged from the vectorized study) ***
  - SURVIVORSHIP BIAS inflates the MAGNITUDE (dead pump-and-die coins are absent,
    worst for the broad meme set). Trust the SIGN + multi-year robustness, not the %.
    Acute here: TROLL earns ~$11k on TWO trades, and PENGU/BONK/FLOKI/PEPE are in
    the carrying set.
  - Short ~2yr / one-cycle sample.
  - This freqtrade run is the honest execution check; divergence from the vectorized
    numbers is expected (order pricing, fee accounting, one-add-per-candle cadence).
  - Fill sizing rests on QUOTE_VOLUME_HEADROOM_MULT. SWEPT (2026-08, 15mFast,
    FILL_VOLUME_LAG=1); net / PF / maxDD and the carrying names' contribution:

      MULT   net %   PF    maxDD    ZEC     PENGU   TROLL   HBAR    XLM
        10   444.2  1.64   58.5%   12,316  10,007  11,347   7,695   7,444
        20   364.9  1.58   50.8%   11,343   8,800   6,181   6,648   6,715
        50   291.5  1.59   44.8%    9,620   6,459   2,150   6,253   6,587
       100   285.2  1.83   36.3%    7,209   4,396   1,067   6,349   6,952

    READ THIS AS: TROLL scales ~linearly with the cap (11.3k -> 1.1k for a 10x
    tighter fill) -- its contribution is a SIZING ASSUMPTION, not an edge, and any
    result leaning on it should be discounted. PENGU is ~half assumption. ZEC decays
    gently (1.7x). HBAR and XLM are FLAT across the whole sweep -- those are real.
    The edge survives a 10x tighter cap (+285%), and tightening IMPROVES PF
    (1.64->1.83) and maxDD (58%->36%): the marginal fill at MULT=10 was buying
    return with drawdown, largely in illiquid names. MULT=100 has the best
    return/maxDD (7.85 vs 7.60). Default stays 10, but quote the sweep, not the
    headline, when judging this family.

*** lookahead-analysis reports "bias detected" — it is a STRUCTURAL FALSE POSITIVE. ***
freqtrade/optimize/analysis/lookahead.py builds every comparison run per trade with a
SINGLE-pair whitelist:

    self.prepare_data(entry_varHolder, [result_row["pair"]])
    prepare_data_config["exchange"]["pair_whitelist"] = pairs_to_load

Under that substitution _compute_xs degenerates two ways: mom.rank(axis=1) <= TOP_N
ranks across one column, so the top-N constraint is vacuously true on every candle;
and known.get(REGIME_REF) finds no BTC/USDT column, so the `ron_d = pd.Series(True)`
fallback silently disables the risk-on gate. `hold` MUST therefore differ from the
full-whitelist reference run, however causal the strategy is — the tool cannot
validate ANY cross-sectional or regime-gated strategy, with any config.
(NB: reading the daily feathers directly off disk is NOT the cause; that is what
keeps the daily inputs identical across runs. The 15m whitelist collapse is.)

The `hold` signal is CAUSAL by construction (momentum = current 15m close /
Pd.shift(1).shift(MOM_LOOKBACK_DAYS); regime + trend off Pd.shift(1); membership
floored to the hour, all ffill-mapped), and that is now PROVEN BY A COMMITTED TEST
rather than prose: Basket/test_momentum_regime_bias.py asserts zero changed
membership cells when future data is removed, at 4 cut points x {cut-all,
freqtrade-exact} x {hourly, per-candle}, and separately pins the single-pair
collapse above. It is mutation-tested — a one-candle peek (P15.shift(-1)) and a
full-sample normalisation both turn it red.

  CAVEAT ON COVERAGE: in the "freqtrade-exact" scenario the daily panel is never
  truncated (it comes off disk), so that scenario cannot catch lookahead introduced
  on the DAILY inputs; the "cut-all" scenario is what covers those. Keep both.

  NB: run check_bias.sh with -c config/config_mom_15m.json. Its default config.json
  is an 11-pair whitelist with NO BTC/USDT (regime gate off), max_open_trades 10 vs
  TOP_N 3, and a fixed 900 stake — that measures a different strategy entirely.

Config: config/config_mom_15m.json (max_open_trades == TOP_N, stake "unlimited").
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
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

    # Exit hysteresis. Entry is always rank <= TOP_N; a HELD coin is kept until its
    # rank passes EXIT_RANK_N. None => TOP_N => no hysteresis (original behaviour).
    # Rationale: with a short MOM_LOOKBACK_DAYS the rank oscillates across the TOP_N
    # boundary, and every oscillation is a round-trip. Measured on 15mFast: 467 of
    # 596 trades were held <24h and lost $19.3k in aggregate, while the 32 trades
    # held >7d made $57.7k. A buffer converts boundary churn into continuous holds
    # and stops winners being shaken out by a one-hour dip to rank TOP_N+1.
    EXIT_RANK_N = None

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

    # Which candle's quote volume bounds a fill. In BACKTEST get_analyzed_dataframe()
    # is sliced to the CURRENT candle, so iloc[-1] is that candle's COMPLETED volume --
    # which is not knowable at the moment the order is placed. Sizing against it is
    # optimistic exactly where it matters most (an illiquid name's pump candle).
    # 1 = last completed candle (causal, the default). 0 = old behaviour, for A/B only.
    FILL_VOLUME_LAG = 1

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
        rank = mom.rank(axis=1, ascending=False, method="first")

        # Hard gates (NO hysteresis on these -- they are the risk controls, so a
        # failing trend or a risk-off regime drops the coin immediately).
        gate = pd.DataFrame(True, index=P15.index, columns=P15.columns)
        if self.TREND_FILTER_ENABLE:                              # drop coins below their own trend
            trend_ok = (known > known.rolling(self.PER_COIN_SMA).mean())
            trend_ok_15 = trend_ok.reindex(columns=P15.columns).reindex(P15.index, method="ffill").fillna(False)
            gate = gate & trend_ok_15
        gate = gate.apply(lambda col: col & risk_on)

        exit_n = self.TOP_N if self.EXIT_RANK_N is None else self.EXIT_RANK_N
        enter_ok = (rank <= self.TOP_N) & gate
        stay_ok = (rank <= exit_n) & gate

        # Decision points: hourly if rebalancing hourly, else every candle.
        idx = P15.index[P15.index.minute == 0] if self.REBALANCE_HOURLY else P15.index
        held = self._hysteresis_membership(rank.loc[idx], enter_ok.loc[idx], stay_ok.loc[idx])
        want = held.reindex(P15.index, method="ffill").fillna(False).astype(bool)
        self._xs = want
        self._xs_key = key
        return self._xs

    def _hysteresis_membership(self, rank: DataFrame, enter_ok: DataFrame,
                               stay_ok: DataFrame) -> DataFrame:
        """Slot-based forward scan over the decision index.

        TOP_N slots. A slot is vacated only when its occupant stops satisfying
        stay_ok (rank > EXIT_RANK_N, or a hard gate fails); vacant slots are then
        filled by the best-ranked pairs satisfying enter_ok. With EXIT_RANK_N None
        (or == TOP_N) stay_ok == enter_ok and this reduces EXACTLY to the plain
        "top-N each decision" behaviour, so it is a no-op by default.

        CAUSAL: state at row i depends only on row i-1 and row i -- never forward.
        Guarded by test_momentum_regime_bias.py.
        """
        r = rank.to_numpy()
        e = enter_ok.to_numpy()
        st = stay_ok.to_numpy()
        n_rows, n_cols = e.shape
        out = np.zeros((n_rows, n_cols), dtype=bool)
        held = np.zeros(n_cols, dtype=bool)
        for i in range(n_rows):
            held &= st[i]                                  # vacate slots that failed
            free = self.TOP_N - int(held.sum())
            if free > 0:
                cand = np.flatnonzero(e[i] & ~held)
                if cand.size:
                    cand = cand[np.argsort(r[i][cand], kind="stable")]
                    held[cand[:free]] = True
            out[i] = held
        return DataFrame(out, index=enter_ok.index, columns=enter_ok.columns)

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

    def _quote_volume(self, df) -> float:
        """Quote volume of the candle FILL_VOLUME_LAG bars back (0 => current)."""
        i = -1 - self.FILL_VOLUME_LAG
        if df is None or len(df) < abs(i):
            return 0.0
        bar = df.iloc[i].squeeze()
        return float(bar["volume"]) * float(bar["close"])

    # --- liquidity-aware sizing: cap the INITIAL fill to the equal-weight target
    #     AND to <=10% of the candle's quote volume ---
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        if self.dp.runmode.value in ("plot", "other"):
            return proposed_stake
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        fillable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
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
        return self._quote_volume(df) >= self.MIN_QUOTE_VOLUME   # reject dust

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
        fillable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
        add = min(target - current_value, fillable, max_stake)
        if add <= 0 or (min_stake and add < min_stake):
            return None
        return add
