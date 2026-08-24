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
  NB: these are EXIT_RANK_N=None (pre-hysteresis) numbers, kept because the tail
  and liquidity analysis below refers to them. For the CURRENT default (9) see
  the EXIT_RANK_N block: P3 is +798% at 32.6% maxDD over the same window.
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

*** EXIT_RANK_N (exit hysteresis) -- DEFAULT 9, VALIDATED ACROSS 3 REGIMES ***
  Diagnosis: the bleed is a HOLDING-PERIOD effect. On 15mFast every duration
  bucket under 24h loses (<2h -$5.7k / 2-6h -$3.3k / 6-24h -$10.3k, 467 of 596
  trades), while the 32 trades held >7d make +$57.7k. With entry and exit sharing
  one threshold, rank oscillation across TOP_N IS the churn.

  Total return %, N=3 (no hysteresis) vs widened, each window a standalone
  backtest from a fresh 10k, config_mom_15m.json, 75 pairs:

    N     P1 2021-05..2022-12   P2 2023-01..2024-08   P3 2024-09..2026-08
    3(=)        -28.3               -21.3                 +483
    5           -34.6              +110.6                 +476
    7            +2.0               +51.1                 +714
    9           +34.0              +163.1                 +834   <- default
    11          +38.4               +96.8                 +773
    15         +112.3              +133.9                 +579
  (re-measured at FILL_VOLUME_LAG=0 after that flag was reverted; the earlier
  lag=1 table differed by <5pp everywhere and changed no conclusion.)

  THE ROBUST RESULT is not any single N: it is that EVERY N >= 7 beats N=3 in
  EVERY window -- 12 of 12 arm-window comparisons at lag=0 (15/15 at lag=1). N=5 is WORSE than baseline in
  both P1 and P3 and N=6 only scrapes it in P1, so the band must be substantially
  wider than TOP_N, not marginally.

  DO NOT RE-TUNE ON ONE WINDOW. The per-window optimum drifts (P1 rises
  monotonically to 15; P2 is erratic, adjacent N swinging 2-4x; only P3 shows the
  tidy 6..11 plateau). An earlier note claiming a general "plateau at 6..11" was
  derived from P3 alone and was wrong. Anything in 8..11 is equivalent within
  regime noise; 9 is the default. 15 wins P1 but is the weakest qualifying arm in
  P3 with its illiquid share up at 59%.

  BIGGEST FINDING: the pre-hysteresis default LOSES MONEY in two of three regimes
  (-29.7%, -19.8%) and only worked in P3. This is not a tuning improvement -- it
  is the difference between a one-regime strategy and a three-regime one.

  STILL TRUE AT N=9: drawdown is severe (44% / 58% / 32% across P1/P2/P3), and
  P1/P2 carry heavy survivorship bias (today's 75-name whitelist applied back to
  54 survivors), so their ABSOLUTE returns are flattered. The N-vs-baseline
  comparison is fair -- both arms see an identical universe -- but do not quote
  the absolute P1/P2 figures.

*** JOINT lb x EXIT_RANK_N GRID (2026-08-24) -- the two DO interact ***
  The two sweeps above each fixed the other parameter, so neither could see the
  interaction. 12 combinations x 3 windows. WORST-of-3-windows total return %
  (the robustness objective; every cell is POSITIVE in all three windows):

              N=7      N=9     N=11     N=15
     lb=14    +2.0    +34.0    +38.4   +112.3
     lb=21   +40.5    +96.1    +53.1    +25.4
     lb=30   +57.0    +69.9    +35.8    +79.2

  READ IT COLUMN-WISE, NOT BY BEST CELL.
  - ALL 12 cells beat the N=3 baseline (-28% P1 / -21% P2) in every window. The
    DIRECTION (widen the exit band) is robust to the lookback. That is the result.
  - DO NOT pick lb=14/N=15 for its best worst-case (+112.3): its neighbours are
    +38.4 and +25.4, i.e. an isolated spike on a noisy surface. Exactly the trap
    the retracted "plateau at 6..11" was.
  - N=9 is the only column with no weak cell (+34.0/+96.1/+69.9), which supports
    the default INDEPENDENTLY of lb -- a better argument than the original one,
    which assumed lb=14.
  - lb=30 is the most robust to getting N wrong (worst cell +35.8 vs lb=14's
    +2.0) but surrenders most of P3 (173-249% vs 579-835%). lb=21/N=9 is the
    best-supported middle. This SHARPENS the lb=21 open decision, it does not
    settle it.

*** CURRENT PRODUCTION BASELINE (2026-08-24, capped exits ON) ***
  MomentumRegimeBasket15mFast (lb=14, EXIT_RANK_N=9, FILL_VOLUME_LAG=0,
  EXIT_LIQUIDITY_CAP=True), config_mom_15m.json, standalone per window:

    P1 2021-05..2022-12   66 trades   +38.8%   maxDD 46.4%
    P2 2023-01..2024-08  160 trades  +197.1%   maxDD 51.4%
    P3 2024-09..2026-08  102 trades  +401.7%   maxDD 42.1%

  These SUPERSEDE every free-exit figure above (P3 was +834.5% / 31.8% DD before
  the cap). Quote these.

  OPEN: EXIT_RANK_N=9 was selected under FREE exits. The cap slows rotation --
  which is why it helped short lookbacks in P1/P2 -- i.e. the same medicine
  hysteresis provides, so the two are now partly redundant. The 12/12 validation
  behind N=9 assumed instant exits. RE-SWEEP N under capped exits before treating
  9 as settled.

*** ACTUAL CRASH FREQUENCY (2026-08-24) -- calibrates the synthetic-death test ***
  Single-candle drops among SURVIVING pairs (volume-bearing candles only):

    year   <=-50%   <=-80%   <=-90%
    2022      3        0        0
    2023      9        1        0
    2024      3        2        2
    2025     13        3        2
    2026     21        4        2

  True one-candle rugs are ~0 before 2024 and ~2/yr since -- RARE. By contrast 42
  of 75 pairs fell >=90% from peak and never recovered half of it, but those are
  COMP/FIL/ONT/BCH/UNI/VET/ETC/CRV/LTC/AVAX/ADA clustered in May-Nov 2022: the
  bear market, playing out over months. So reality sits close to the SOFT profile,
  where lb=14 improves (+38.8 -> +67.2), NOT the hard-rug case that inverted it.
  The fragility argument for lb=21 rests on a TAIL scenario, not a base case.

  CAVEATS: survivors only -- delisted coins are precisely those most likely to have
  rugged hard, so 0-2/yr is a FLOOR not an estimate. Some -50% candles are likely
  data artifacts (freqtrade flags "Price jump" on ZIL 100.6%, XNO 75.3%). To get a
  real number: Binance US delisting announcements give the death arrival rate;
  Binance global's public archives give the delisted coins' profiles so each death
  could be classified soft vs hard.

*** SYNTHETIC-DEATH BOUND (2026-08-24) -- survivorship cuts BOTH ways ***
  P1's universe is missing the coins that pumped and then delisted. Does that
  understate a SHORT lookback (which would have caught those pumps)? Injected 30
  synthetic pump-then-die coins into P1 (carried on real exchange tickers we hold
  no data for, so pairlist validation passes), EXIT_LIQUIDITY_CAP on, profile
  calibrated on SHIB/APE/GALA (4-25x run-up, dies to 2-12% of peak, VOLUME TRACKS
  THE PUMP -- a coin that never trades is harmless and would make the test
  toothless). Total return %:

    lb    control   soft deaths     hard rugs (3 seeds)        mean   worst
    14     +38.8      +67.2      -16.9 / -20.4 / -63.7        -33.7   -63.7
    21     +79.6      +74.0      +44.0 / -34.8 / +19.6         +9.6   -34.8
    30     +54.1     +161.0      -25.2 / -38.2  / +2.9        -20.2   -38.2

  SOFT (decay over weeks, the exit can fire): every lookback improves; lb=14
  nearly doubles. Momentum rides the pump and leaves -- as designed.
  HARD (90% gone in ONE candle, no exit can outrun it): lb=14 is negative in 3/3
  seeds with the worst mean and worst single outcome. A short lookback chases
  pumps hardest, so it eats the most rugs.

  CONCLUSION: the survivorship question resolves in OPPOSITE directions depending
  on how the missing coins died -- which is exactly what the data cannot show. So
  lb=14's 5th-of-6 ranking in P1 can be neither excused as survivorship nor
  confirmed as real. The usable finding is FRAGILITY, not expected return: lb=14
  swings ~130pp across death profiles it cannot see coming; lb=21 swings less and
  centres higher. That is the strongest argument for the longer lookback so far,
  and the only one not resting on in-sample return.

  CAVEATS: magnitudes are heavily seed-dependent (lb=14 spans -17% to -64%); only
  the ORDERING is stable. The arrival rate (30 coins/window) is a guess and scales
  the effect. Do not quote the levels; quote the ordering and the spread.

*** EXIT LIQUIDITY (2026-08-24) -- entries were capped, EXITS WERE NOT ***
  populate_exit_trend dumped the WHOLE position with no volume check. Measured on
  the P3 production run (128 trades, $83,449 net):

    exits into a ZERO-volume candle      33 trades   $17,437 of profit
    exit > 100% of the candle's volume   84 trades   $45,319
    zero-vol OR >100%                   117 of 128   $62,755 = 75% OF NET PROFIT

  Worst: PENGU exiting $19,602 into a ZERO-volume candle (+$14,544); BONK $18,245
  into $156 (117x); PUMP $35,074 into $1,210 (29x). Context: TROLL trades in 4.6%
  of candles and its ENTIRE quote volume over P3 is $611k (~$850/day), yet the
  strategy booked $11,347 from it.

  EXIT_LIQUIDITY_CAP (DEFAULT ON since 2026-08-24) makes exits symmetric with entries:
  confirm_trade_exit refuses a full exit the candle cannot absorb, and
  adjust_trade_position shaves the position down over subsequent candles.

  lb sweep, total return %, free exit -> capped exit:

    lb      P1              P2                P3
     7    +21.7 -> +15.6   +73.9 -> +61.1   +243.7 -> +206.0
    14    +35.6 -> +38.8  +163.1 -> +197.1  +834.5 -> +401.7   <- -52% in P3
    21    +95.8 -> +79.6  +186.0 -> +144.9  +345.7 -> +177.0
    30    +76.3 -> +54.1  +163.7 ->  +99.2  +189.2 -> +184.6
    60    +48.1 -> +42.1   +87.9 ->  +24.1   +94.4 ->  +79.4
    90    +67.2 -> +66.7  +118.6 ->  +78.9  +263.8 -> +121.1

  - The cap bites ONLY in P3 (neutral-to-POSITIVE in P1/P2) -- the signature of a
    real liquidity correction, not a blanket tax. In liquid windows it helps short
    lookbacks by slowing rotation, the same medicine EXIT_RANK_N administers.
  - lb=14 loses 52% of its P3 return but STILL WINS P3 (+401.7 vs +206 next).
    Its edge is reduced, not manufactured.
  - Worst-of-3 ranking is UNCHANGED by the cap: lb=21 best (+79.6 capped / +95.8
    free), lb=14 fourth (+38.8). Realistic exits shrink magnitudes without
    resolving the short-vs-long trade-off.
  - COST: capped lb=14 in P3 carries 42.1% maxDD vs 31.8% uncapped -- being unable
    to exit means wearing more downside. Real, not an artifact.

*** UNIVERSE CONFOUND (2026-08-24) -- partial, does NOT overturn lb=14 ***
  Is P3's preference for a short lookback just the meme cohort that did not exist
  earlier? Re-ran P3 on the 55 pairs that existed BEFORE P3 began (excluding PENGU,
  TROLL, BONK, PEPE, HYPE, PUMP + 14 others). Total return %:

    lb        7      14      21      30      60      90
    full   243.7   834.5   345.7   189.2    94.4   263.8
    pre-P3 278.2   440.2   183.9   211.2   134.8   161.3

  New listings account for ~HALF of lb=14's advantage (834 -> 440), but lb=14 still
  WINS OUTRIGHT on the pre-existing universe, by a wider relative margin. So P3's
  short-lookback preference is a genuine regime property, NOT a new-listing
  artifact. The magnitude was inflated ~2x; the direction is real.

  OPEN: whether prior windows had pump-and-dump names (now delisted, hence absent)
  that a short lookback would also have caught. Cannot be tested by removing coins
  from P3 -- it needs missing coins RESTORED to P1/P2, i.e. delisted data we do not
  have. Exchanges do not serve delisted symbols; partial routes are Binance global's
  public archives, paid vendors, or daily-only sources (which would suffice, since
  the ranking panel is daily). Nearest cheap proxy: inject synthetic pump-then-die
  coins and check whether the short lookback still wins.

*** PER-WINDOW HYPEROPT (2026-08-24) -- lb NOT identifiable, N=9 confirmed ***
  Ran hyperopt SEPARATELY per window (identifiability probe, not optimisation),
  60 epochs each, WalletCalmarHyperOptLoss, same random-state. Winners:

    P1 2021-05..2022-12   lb=33  N=9     P2 ...   lb=28  N=11     P3 ...  lb=13  N=7

  Each winner then evaluated in ALL three windows (diagonal = where it was tuned):

    config                    P1        P2        P3      WORST
    P1 win  lb=33/N=9      100.1%     58.2%    151.4%    +58.2%   <- most robust
    P2 win  lb=28/N=11      47.9%    340.4%    264.0%    +47.9%
    P3 win  lb=13/N=7      -31.1%     75.0%    724.4%    -31.1%   <- LOSES in P1
    PRODUCTION lb=14/N=9    34.0%    163.1%    834.5%    +34.0%

  - lb is NOT identifiable: winners 33/28/13 are monotonic in recency (old regimes
    want LONG, recent wants SHORT) -- a regime property, not a tunable constant.
  - A P3-ONLY hyperopt returns lb=13/N=7 with a compelling +724% objective, and
    that config LOSES 31% in P1. This is the concrete cost of single-window tuning.
  - N=9 confirmed a 4th independent way (P1's winner, production, and the exit band
    of the best-worst-case config). Every method lands in 7..11.
  - Long lookbacks (28-33) are the transferable family. lb=21 was NOBODY's winner --
    if the goal is robustness the evidence points to ~30, not to splitting the
    difference. Production lb=14 is not dominated; it is a deliberate bet on the
    current regime (+834% P3 vs +151% for lb=33).

  HARNESS GOTCHAS -- hyperopt of these params is silently INERT without both:
   1. `--analyze-per-epoch`. freqtrade computes indicators ONCE and pickles them
      (hyperopt_optimizer.py: `if not self.analyze_per_epoch`), re-running only
      populate_entry/exit_trend per epoch. MOM_LOOKBACK_DAYS and EXIT_RANK_N act
      entirely through `hold` in populate_indicators, so without this flag they are
      frozen at defaults. Proof: 60 distinct param combos -> 1 distinct trade count.
   2. Delete <Strategy>.json between runs. hyperopt WRITES it, and the next run
      LOADS it as fixed values ("Strategy Parameter: ..." at startup).
   3. They also need the _xs cache key to include them -- see _xs_params().
  All three failure modes look identical: a flat objective that reads as "this
  parameter does not matter". DIAGNOSE VIA THE .fthypt FILE (every epoch's params +
  metrics); console output cannot distinguish "did not vary" from "did not improve".

*** LOOKAHEAD AUDIT (2026-08-23) -- CLEAN, three independent checks ***
  1. Signal path: test_momentum_regime_bias.py, 74 tests, truncation-invariance
     over 4 cut points x {cut-all, ft-exact} x {hourly, per-candle} x {no-hyst,
     hyst9}. Mutation-tested (a one-candle peek and a full-sample normalisation
     both turn it red).
  2. Within-candle: MEASURED that df.iloc[-1] inside custom_stake_amount /
     confirm_trade_entry / adjust_trade_position is the PREVIOUS COMPLETED candle
     -- lag exactly one timeframe, 8000/8000 + 200/200 + 100/100 calls. So
     _portfolio_value's mark-to-market, adjust's last["hold"], and _quote_volume
     are all causal as written. See FILL_VOLUME_LAG.
  3. End-to-end: two backtests differing ONLY in end date (2026-08-23 vs
     2025-12-31); of the trades closing before 2025-12-01, all 111 are IDENTICAL
     on pair / open+close date / stake / amount / open+close rate / profit / exit
     reason. Strategy file md5 verified unchanged across both runs.

  These cover DIFFERENT failure modes and none is sufficient alone: (3) is
  structurally blind to within-candle leakage (it looks the same in both runs),
  which is exactly what (2) covers; (1) is the only one that localises a fault.
  NB freqtrade's own `lookahead-analysis` cannot be used here -- see the
  false-positive section above.

*** MOM_LOOKBACK_DAYS -- 14 IS A P3 ARTIFACT, NOT VALIDATED CROSS-REGIME ***
  Same sweep protocol, EXIT_RANK_N=9. Total return %, rank in brackets:

    lb     P1              P2              P3
     7     +21.7 (6)       +73.9 (6)      +243.7 (4)
    14     +35.6 (5)      +166.2 (2)      +834.5 (1)   <- 15mFast's defining value
    21     +95.8 (1)      +186.0 (1)      +345.7 (2)
    30     +76.3 (2)      +163.7 (3)      +189.2 (5)
    60     +48.1 (4)       +87.9 (5)       +94.4 (6)
    90     +67.2 (3)      +118.6 (4)      +263.8 (3)

  lb=14 ranks FIFTH OF SIX in P1. Its dominance is a P3 phenomenon (there it is
  2.4x the runner-up). lb=21 is 1st/1st/2nd -- the only value strong everywhere.
  The original lb=14 "persistence" study split 2024-05..2026-08 in half, which is
  ENTIRELY INSIDE P3, so both halves shared one regime and it could not have
  detected this. NOT CHANGED: lb=14 is the identity of MomentumRegimeBasket15mFast
  and there is a real trade-off (lb=14 ret/DD 26.2 in P3 vs lb=21's 7.8). Decide
  deliberately; do not treat 14 as validated.

*** CAVEATS (unchanged from the vectorized study) ***
  - SURVIVORSHIP BIAS inflates the MAGNITUDE (dead pump-and-die coins are absent,
    worst for the broad meme set). Trust the SIGN + multi-year robustness, not the %.
    Acute here: TROLL earns ~$11k on TWO trades, and PENGU/BONK/FLOKI/PEPE are in
    the carrying set.
  - Short ~2yr / one-cycle sample.
  - This freqtrade run is the honest execution check; divergence from the vectorized
    numbers is expected (order pricing, fee accounting, one-add-per-candle cadence).
  - Fill sizing rests on QUOTE_VOLUME_HEADROOM_MULT. SWEPT (2026-08, 15mFast,
    FILL_VOLUME_LAG=1 -- since reverted to 0, see below); net / PF / maxDD and
    the carrying names' contribution:

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

    # Exit hysteresis. Entry is always rank <= TOP_N; a HELD coin keeps its slot
    # until its rank passes EXIT_RANK_N. None => TOP_N => no hysteresis.
    # Rationale: with a short MOM_LOOKBACK_DAYS the rank oscillates across the TOP_N
    # boundary, and every oscillation is a round-trip. Measured on 15mFast: 467 of
    # 596 trades were held <24h and lost $19.3k in aggregate, while the 32 trades
    # held >7d made $57.7k. A buffer converts boundary churn into continuous holds
    # and stops winners being shaken out by a one-hour dip to rank TOP_N+1.
    # 9 is validated across three regimes (see the block above). The exact value is
    # NOT identifiable -- anything in 8..11 is equivalent within regime noise. Do
    # NOT re-tune it on a single window; the per-window optimum drifts (15/8/9).
    EXIT_RANK_N = 9

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

    # Cap EXITS to the same share of a candle's volume as entries. ON by default
    # since 2026-08-24: leaving it off models exits that cannot happen. Set False
    # to reproduce any result recorded before that date -- every absolute figure in
    # this docstring predating the switch was measured with FREE exits.
    #
    # WHY IT EXISTS: entries were liquidity-capped but populate_exit_trend dumped the
    # WHOLE position with no volume check. Measured on the P3 production run: 33 of
    # 128 exits landed in a ZERO-volume candle, 84 exceeded 100% of their candle's
    # volume, and trades that were zero-vol-or->100% carried $62,755 of $83,449 --
    # 75% of net profit. Worst: PENGU exiting $19,602 into a zero-volume candle,
    # BONK $18,245 into $156 (117x), PUMP $35,074 into $1,210 (29x).
    #
    # CONSEQUENCE when enabled: a position that leaves the basket is unwound over
    # many candles instead of instantly, so it keeps occupying one of the TOP_N
    # slots and blocks rotation. That is the real constraint, not a bug.
    EXIT_LIQUIDITY_CAP = True

    # liquidity-aware sizing (same discipline as FundingCarry / the NN family)
    MIN_QUOTE_VOLUME = 1000
    QUOTE_VOLUME_HEADROOM_MULT = 10.0   # fill <= 1/10 of a candle's quote volume

    # Extra lag, in candles, on the quote volume that bounds a fill. 0 = the frame
    # get_analyzed_dataframe() returns, i.e. df.iloc[-1].
    #
    # MEASURED (do not "fix" this again without re-measuring): in BACKTEST,
    # df.iloc[-1] inside custom_stake_amount / confirm_trade_entry /
    # adjust_trade_position is the PREVIOUS, COMPLETED candle -- lag exactly one
    # timeframe (15.0 min in 8000/8000 adjust calls, 200/200 stake, 100/100
    # confirm). backtesting.py bumps row_index BEFORE _set_dataframe_max_index and
    # the slice is exclusive-end, which reads like it includes the current candle;
    # it does not. So lag 0 is ALREADY CAUSAL and is the correct default.
    #
    # A previous commit (be80e63) defaulted this to 1 on the false premise that
    # iloc[-1] was the in-progress candle. That cost ~4.3% of return for no
    # correctness reason. Retained only as a deliberate conservatism / stress
    # lever: 1 sizes off a candle two bars old.
    FILL_VOLUME_LAG = 0

    _xs = None       # cached membership matrix (bool DataFrame, per pair)
    _xs_key = None   # cache key: (latest candle date, whitelist, *_xs_params())

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

    def _xs_params(self) -> tuple:
        """Every attribute that changes the membership matrix, for the cache key.

        MUST include anything _compute_xs reads. freqtrade's hyperopt mutates
        parameters IN PLACE on a single reused strategy instance (see
        hyperopt_optimizer.generate_optimizer: `attr.value = params_dict[...]`),
        while `asof` and the whitelist are constant within a backtest. With a key
        of only (asof, whitelist), epoch 2 onward would hit the cache and silently
        reuse epoch 1's matrix -- and since these attributes affect NOTHING else,
        every epoch would score identically and hyperopt would conclude the
        parameters are inert. It would not error. Keep this in sync.
        """
        return (
            int(self.MOM_LOOKBACK_DAYS),
            int(self.TOP_N),
            int(self.REGIME_SMA),
            str(self.REGIME_REF),
            bool(self.REBALANCE_HOURLY),
            bool(self.TREND_FILTER_ENABLE),
            int(self.PER_COIN_SMA),
            None if self.EXIT_RANK_N is None else int(self.EXIT_RANK_N),
        )

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
        key = (asof, wl) + self._xs_params()
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
        """Quote volume of the fill-reference candle, plus FILL_VOLUME_LAG extra bars.

        NOTE: lag 0 is the last COMPLETED candle in backtest, not the in-progress
        one -- see FILL_VOLUME_LAG. It is causal as-is.
        """
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

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force,
                           exit_reason, current_time, **kwargs):
        """Refuse a full exit the candle cannot absorb; adjust_trade_position then
        shaves the position down over subsequent candles until it can."""
        if not self.EXIT_LIQUIDITY_CAP or self.dp.runmode.value in ("plot", "other"):
            return True
        if exit_reason in ("force_exit", "stop_loss", "liquidation"):
            return True                      # never block a forced exit
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        sellable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
        return (amount * rate) <= sellable

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
            if not self.EXIT_LIQUIDITY_CAP:
                return None   # leaving the basket -> full exit handled by the exit signal
            # Liquidity-capped unwind: release only what this candle can absorb. When
            # the remainder fits, confirm_trade_exit lets the exit signal finish it.
            sellable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
            if sellable <= 0 or sellable >= current_value:
                return None
            reduce = -min(sellable, current_value)
            if min_stake and abs(reduce) < min_stake:
                return None
            return reduce
        target = pv / self.TOP_N
        if current_value >= target * 0.98:
            return None   # already at target weight
        fillable = self._quote_volume(df) / self.QUOTE_VOLUME_HEADROOM_MULT
        add = min(target - current_value, fillable, max_stake)
        if add <= 0 or (min_stake and add < min_stake):
            return None
        return add
