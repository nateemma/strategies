# Predictability & Tradeability of Crypto Alts on US Spot — A Study Log

*A rigorous, mostly-negative research program on whether — and how much — an ML-driven
retail trader can predict and profitably trade a basket of crypto alts on a **US spot-only
venue (Binance.US)**, and where the real limits are.*

**Setup:** freqtrade; a ~11-to-75 alt universe (core: ZEC, XRP, SOL, LINK, NEAR, AAVE, SUI,
AVAX, LTC, BCH, DOT); 15m/1h timeframes for the intraday NN work and 1d for the longer-horizon
momentum work; MLX/Keras NN classifiers (conv1d→LSTM) plus rule-based strategies. All backtests are
walk-forward (train 240d / test 55–180d, multiple eras 2024–2026) with realistic fees and (where
noted) liquidity-aware fill modeling.

---

## TL;DR

Markets here are **weakly predictable, not unpredictable** — and the honest conclusion is more
precise than "you can/can't beat the market":

1. **A real edge exists and is deployed.** A mean-reversion NN strategy (buy statistical dips,
   "gbb") is profitable out-of-sample. So prediction *to a degree* is demonstrated, not hoped-for.
2. **But the signal in OHLCV is small and near a ceiling.** Every method — hand-crafted indicators,
   an 85-indicator battery, and learned CNNs on raw OHLCV — converges on the same predictability
   ceiling (Spearman ρ ≈ 0.15 for forward returns). That number is an **information limit**, not a
   modeling failure. You cannot out-model your way past it on the same inputs.
3. **Most remaining edges are real but *un-capturable* on US spot — with one exception.** The persistent
   intraday signals (funding-rate reversion, market-neutral spreads, cross-sectional reversion) are blocked
   by structural constraints (**no shorting, thin liquidity, OHLCV-only**), and *three* of them reduce to the
   **same one illiquid coin (ZEC)**. **The exception is longer-horizon cross-sectional momentum with fast
   (hourly) execution:** confirmed in freqtrade at **+114%** across 2024–2026 against a *falling* market
   (−28.5%), with **real** (if partial) diversification beyond ZEC — the one branch where the "it's all ZEC /
   edge lives where you can't trade" pattern breaks, because *accumulating* fills over a hold window captures
   pumps that single-shot fills miss. Not yet cleanly deployable, though: ZEC still ≈69% of net return and
   the drawdown is severe (52–64%). Details below.
4. **So the frontier is structural, not algorithmic:** new *information* (order flow, funding, on-chain)
   or a venue/instrument (leverage/shorting, deeper liquidity) — not a better model.

This is the textbook signature of an **"efficiently inefficient" market**: efficient enough that cheap
edges are gone, inefficient enough that a small, hard-won edge remains for whoever pays the price in
work, capital, information, or access.

---

## What works

- **gbb mean-reversion NN family (deployed).** A conv1d→LSTM 3-class classifier trained on a "guard
  metric < threshold" dip label. Profitable walk-forward. It operates *at* the OHLCV information
  ceiling and is already well-tuned.
- **Caveat — concentration.** Over a full year, ~70% of trades / nearly all P&L came from **one pair
  (ZEC)**, an illiquid, high-volatility alt. Half the whitelist never traded. The strategy is really a
  2–4-pair high-volatility-alt harvester; its guards *are* an implicit, volatility-based pair selector.

## The central finding: an information ceiling, not a modeling problem

The strongest, most-repeated result. The "purest" trade target — a **triple-barrier** label (did price
hit the profit target before the stop) — is essentially **unpredictable from OHLCV** (MCC ≈ 0.007). And
this is a *data* problem, confirmed four independent ways:

| test | result |
|---|---|
| Hard triple-barrier (binary) | MCC ≈ 0.00 |
| Soft/continuous forward-return regression | R² < 0 (worse than the mean), ρ ≈ 0.15 |
| **85-indicator battery** (3× the project's features) + RF/MLP interactions | **no new signal**; OOS R² got *worse* (overfit), ρ invariant |
| **Learned 1D-CNN on raw OHLCV** | ρ ≈ 0.04–0.06 — *worse* than hand-crafted indicators |

A CNN *can represent* any indicator, so if representation were the bottleneck it should have won. It
lost. The ceiling is the **mutual information between an OHLCV window and the forward return** on a
near-efficient market — and both hand-crafted and learned methods hit it. The ~ρ 0.15 that *does* exist
is exactly what the deployed gbb strategy already monetizes.

## What was tried and rejected (the map of dead ends)

Every "reshape or re-extract from OHLCV" lever was tested and found not to add robust edge:

- **Feature / label engineering** — breakout/momentum labels (learnable but negative-EV: alts
  mean-revert at 15m, so a breakout is the local top); session/temporal features (signal 0.01–0.11,
  OOS-neutral across MLX *and* Keras, multi-seed); Fibonacci retracement features (in-sample lift,
  OOS noise). All rejected.
- **Model capacity** — CNNs on raw OHLCV (worse than hand-crafted); GAN data augmentation (net noise;
  only a manifold-aware autoencoder filter marginally helped in one config).
- **Universe / pair selection** — expanding to liquid majors (BTC/ETH/…): they never trade (too calm
  for the vol guards); forcing them in loses −34% (no gbb edge there). Guards aren't a hyperopt-loss
  artifact — they select the edge *and* pace capital.
- **Position sizing** — volatility-targeting *reduced* return and Sharpe: it sizes the high-vol ZEC
  entries (the edge) down and up-sizes no-edge majors. Fixed stake wins.
- **Timing** — day-of-week / session effects were an artifact: a skew that looked persistent across
  two *adjacent* windows *flipped* in a distant era. No temporal edge.
- **Exit tuning** — the deployed exit is already well-calibrated; an apparent "tighten the stop" win
  was pair-mix overfit that flipped in a distant era.

## Recurring structural walls (why edges can't be captured here)

Three walls explain nearly every negative result:

1. **The OHLCV information ceiling** (ρ ≈ 0.15) — you can't extract signal that isn't in price data.
2. **Illiquidity — "the edge lives where you can't cheaply trade."** The best signals fire in illiquid
   moments. Concretely: enforcing realistic fills (cap orders to ≤10% of candle volume, reject dust)
   removed **~73%** of the funding strategy's backtested trades and cut its return from +23.7% to
   **+6.6%**. The cross-sectional reversion edge was ~92% one illiquid coin (ZEC); a liquid-only subset
   *lost* money. Phantom fills flatter every backtest that ignores them.
3. **No shorting.** The genuinely attractive edges are **market-neutral** — long/short spreads whose
   alpha is hidden by market beta. On long-only spot you inherit the beta and the thin alpha drowns.

## Methodological lessons (reusable by anyone)

These are worth as much as the findings:

- **Learnability ≠ trading edge.** A label the model predicts well (high MCC/ρ) is *not* the same as a
  profitable trade after stops, fees, and execution. Judge on walk-forward P&L, never on prediction
  metrics. (Confirmed repeatedly — the highest-learnability configs traded *worse*.)
- **Two adjacent walk-forward windows share a regime and are NOT independent.** A pattern can look
  "persistent" across them and *flip* in a temporally-distant era. Always validate on a distant window
  before believing an edge. (This killed the day-of-week effect and a stop-tuning "win" — and it's what
  gave the funding signal credibility, because funding *didn't* flip.)
- **Model phantom fills.** Standard backtests fill at the candle price regardless of volume. Port
  liquidity-aware sizing (reduce-to-fillable + reject-dust) *before* trusting any result on thin pairs.
- **Stage your gates.** Cheap signal-check → cheap no-GAN retrain → full expensive chain, in that order.
  Don't spend a scaler+GAN+classifier retrain on a feature that fails a 5-minute signal check.
- **Capacity ≠ signal.** More features / bigger models made OOS *worse* here. On a low-SNR target,
  flexibility buys overfitting, not edge.

## The one real new signal: funding-rate reversion

The single most promising find, and the only signal to survive distant-era validation.

- **What it is.** Funding rate is *derivatives positioning* — orthogonal to OHLCV. Extreme funding
  (crowded longs/shorts) precedes squeezes/reversals. Extreme-decile forward-24h spread ≈ **+0.6 to
  +0.8pp, stable across 2024, 2025, and 2026** (where DOW, stop-tuning, and cross-sectional all flipped).
- **Data, free.** Binance's historical funding dumps (`data.binance.vision`) are a static CDN, *not*
  geo-blocked; live it's refreshed from OKX (Binance-global/Bybit are geo-blocked from US IPs). Binance.US
  itself has **no funding** — it's spot-only, so funding is inherently a *cross-venue signal* here.
- **Why it can't be harvested on US spot.** (a) It lives at an **8–24h horizon**; the fast gbb strategy
  exits in ~2h, before the squeeze, and at 2h the effect *inverts*. A dedicated slow strategy
  (`FundingCarry`: enter on extreme-negative funding, 16h patient time-exit, ~21d BTC circuit-breaker)
  captures it — but (b) **long-only** inherits market beta (loses in bear), and (c) realistic fills
  show ~73% of the entries are un-fillable illiquid moments, leaving a **marginal ~+6.6% / Sharpe 0.89**.
  Real edge, marginal capture. `FundingCarry.py` is committed as a documented research artifact.

## The longer-horizon detour: cross-sectional momentum — and how it reduces to the same wall

Everything above is intraday (2–24h). The obvious escape is to change the *horizon*, not the model:
maybe a **weeks-to-months cross-sectional momentum** strategy — hold the top-N trailing-return alts,
rebalance slowly — sidesteps the intraday information ceiling and the fee drag entirely. It's long-only
and US-spot-legal. For a while it looked like the first genuinely deployable, all-weather US edge:

- **The apparent win.** Rank the universe by 90-day trailing return, hold an equal-weight top-3 basket,
  and — critically — **go fully to cash when BTC is below its 100-day SMA** (a regime circuit-breaker).
  This flipped the 2026 bear from ~−40% to positive, roughly **halved max drawdown**, and beat
  buy-and-hold across the sample (frictionless +205% vs −11%, Sharpe ~1.3). It looked *all-weather*.
- **Then the same four walls closed in.** Under scrutiny the headline dismantled itself:
  1. **It's ZEC — again.** Remove ZEC and the recent-period edge collapses **+120% → −13%.** The
     "cross-sectional momentum edge" is really ZEC-momentum with a regime filter — the *same single
     illiquid coin* that carries gbb, the cross-sectional-reversion study, and half of funding.
  2. **Concentration.** ~**2 months** (Oct 2025 + Jun 2026) account for **123% of profit**; the other
     17 months net *negative*. It's a positive-skew pump lottery with n≈2 payoff events — magnitude
     *and sign* are fragile.
  3. **Broadening the universe doesn't help *at daily rebalance*.** Expanding 20 → 75 coins
     de-concentrates it (ZEC 82% → 31%) but the new contributors are thin meme-coins whose pumps are
     *un-executable* on daily next-candle fills: frictionless **+329% → +59%** in freqtrade, and what
     survives is still ZEC.

At this point the arc looked like every other family — *it's just ZEC, un-capturable*. **That conclusion
was wrong, and finding out why is the most interesting result of the momentum work.**

**The correction — execution *timeframe*, honestly modeled, breaks the wall.** The "daily is the only
capturable version" claim rested on a flawed liquidity model. A first pass suggested 15m-data/hourly
rebalancing was *catastrophic* (−17%) — but that model judged fillability on a **single 15m candle** while
holding for hours, and re-drew the whole position each hour, booking phantom turnover fees that manufactured
the negative sign. Rebuilt honestly — a **persistent, accumulating** position (you don't fill in one candle;
a sticky 90-day ranking lets you *build the position up* over the whole hold window, and you fill *something*
every period), fills priced at the **volume-weighted price** over the window (so pump-chasing is penalized),
fees only on real trades — the picture inverts:

  | universe | total | 2024 | 2025 | 2026 | ex-ZEC |
  |---|---:|---:|---:|---:|---:|
  | Core 20 | **+223%** | +98 | +90 | +35 | **+158%** |
  | Broad 77 | **+155%** | +43 | +83 | +29 | **+116%** |

  (conservative VWAP fills, $50k account, per-year contributions — the vectorized *estimate*). Three earlier
  conclusions flip: **(a)** finer timeframe is *not* the wall — hourly action on a slow, sticky signal is the
  *best* execution, because you accumulate fills across many thin candles instead of needing one fat one;
  **(b)** it is *not* just ZEC — the *winning side* is genuinely diversified; **(c)** the broad meme universe
  that was un-executable at daily (+59%) *is* capturable at 15m/hourly.

**Confirmed in freqtrade (real next-candle fills) — the honest numbers, which supersede the vectorized
estimate above.** Built `MomentumRegimeBasket15m` (15m data, hourly rebalance, position-adjustment
*accumulation* toward an equal-weight target — the essential mechanic; a single capped fill *is* the −17%
failure) and backtested the broad 75-coin universe, 2024-08 → 2026-07:

  | run | total | ex-ZEC | vs market | Sharpe / Calmar / maxDD |
  |---|---:|---:|---:|---:|
  | Broad 75 | **+114%** | **+36%** | market −28.5% (ex-ZEC alts −47%) | 0.88 / 4.8 / **52% (64% wallet)** |

  So the core correction **holds under real execution** (+114%, decisively not −17%), and the diversification
  is **real but partial**: the winning side is spread (SUI/TROLL/DOGE/PENGU collectively rival ZEC on gross),
  ex-ZEC is still positive and beats a −47% alt market — *but* ZEC is still ≈**69% of net** return, and real
  fills **deflate the meme diversification hard** (vectorized ex-ZEC +116% → real **+36%**): thin meme pumps
  fill far worse in reality than the VWAP model assumed. Survivorship bias compounds this (dead pump-and-die
  names absent).

**Verdict:** changing the *execution timeframe* — not the model — genuinely broke the −17% wall and added
real diversification; fast-execution momentum beats a falling market across multiple regimes and is the
program's best US-spot-legal candidate. But it is **not a clean deployable edge yet**: ZEC still carries the
majority of the net return, the meme breadth deflates under real fills, and — now the binding problem — the
**drawdown is severe (52–64%, a 205-day underwater stretch)**. `MomentumRegimeBasket.py` (daily) and
`MomentumRegimeBasket15m.py` (15m/hourly, accumulating) are both committed as research artifacts.

*Drawdown investigation.* The 52% drawdown turned out to be a **risk-on** drawdown — the whole Dec 2024→Jun
2025 bleed accrued *while BTC held above its SMA100*, so the regime filter never engaged; the failure mode is
buying alt/meme pumps that have already rolled over (not "least-bad losers" — an absolute-momentum gate does
nothing). The lever that helps is a **per-coin trend filter** (drop any held coin below its own 50d SMA):
closed-trade DD 52→44%, wallet DD 64→58%, Calmar 4.8→4.9, at an ~8pp return cost (+114%→+105%). Faster
windows and a faster BTC regime both whipsaw and do worse. It's a real but *modest* fix — the residual DD is
largely **structural to the convex/lottery payoff** (27% win rate; you bleed for months awaiting the rare
blow-off), and sizing down to shrink it just scales the return down with it. The filter is on by default.

---

## What might be worth pursuing on a leverage venue (outside the US)

The program's biggest takeaway for anyone *not* constrained to US spot: **the hard work of finding the
edges is done — the walls that blocked them (no shorting, thin liquidity) are exactly what a leverage/
derivatives venue removes.** On an offshore perp exchange (Binance-global, Bybit, OKX, Hyperliquid, …)
the following go from "real but un-capturable" to "directly tradeable." Roughly in order of how much the
evidence here already supports them:

1. **Market-neutral funding-reversion (long low-funding / short high-funding).** *This is the #1
   candidate* — the program proved the funding signal is real and era-persistent; the only reason it
   underperformed was long-only beta contamination and spot illiquidity. Go long the most
   extreme-negative-funding names and short the most extreme-positive, sized to be beta-neutral. This
   **hedges the market beta and isolates the ~+0.33pp/24h funding alpha** — and does it on deep perp
   books where the ~73% phantom-fill problem largely disappears. The infrastructure (funding data
   pipeline, z-score signal, horizon, circuit-breaker) is already built.
2. **Funding carry (harvest the payment, not the reversion).** Hold the *receiving* side of funding
   (short when funding is positive, long when negative) and collect the periodic payment, delta-hedged.
   A distinct, well-known income stream that only exists on a perp venue.
3. **Cash-and-carry / perp-spot basis.** When a perp trades at a premium to spot (positive basis),
   short perp + long spot to earn the convergence *plus* funding, delta-neutral. Classic, capacity-heavy,
   low-drama.
4. **Add the short leg to the existing gbb signal.** The NN mean-reversion model predicts *direction*;
   US spot could only act on "buy the dip." A leverage venue lets you also *short the rip* (the
   symmetric overbought signal), roughly doubling the edge's reach and letting it trade in down markets.
5. **Cross-sectional long/short reversion — *carefully*.** The spot version was an illiquidity mirage
   (~92% one coin; liquid-only subset lost money). With shorting *and* a liquid universe it *might*
   work, but treat it as unproven — the spot result was a genuine warning, not just a US-access artifact.
6. **Modest leverage on the deployed gbb edge.** The mean-reversion edge is real but thin; leverage
   amplifies it — *and* the drawdowns and liquidation risk. Only with hard risk limits; leverage magnifies
   a small edge in both directions and does not create new signal.
7. **Deeper liquidity makes *everything already found* more capturable.** Simply running the same
   strategies against Binance-global's books (vs Binance.US's thin ones) reclaims much of the phantom
   fraction — the edges shrink less under realistic fills.

**Caveats for the leverage path (don't skip these):** funding costs eat carry and can flip against you;
liquidation risk turns a normal drawdown into a wipeout; leverage magnifies the *thin* edge both ways, so
position sizing and kill-switches matter more than the signal; and the OHLCV information ceiling still
applies — leverage changes *capturability and instruments*, not *predictability*. The move that actually
raises the ceiling is still **new information** (order flow / CVD, funding as an input, liquidations,
on-chain), on *any* venue.

---

## Bottom line

This program didn't prove the market is unpredictable — it **mapped the box.** For an OHLCV-only,
US-spot operator, prediction is near its ceiling and the deployed mean-reversion edge is about as good as
this input/venue combination allows. Of the two escape routes tested, one held and one broke: a *different
model* (learned CNNs) can't beat the information ceiling — but a *different horizon* (cross-sectional
momentum) **did**, once execution was modeled honestly and confirmed in freqtrade (+114% across three years
vs a −28.5% market). It's the program's best US-spot-legal candidate — genuinely diversified on the winning
side — though not yet cleanly deployable (still ~69% ZEC on net, 52–64% drawdown), so the open work there is
taming risk, not finding return. For the intraday OHLCV problem, though, the remaining upside is entirely
structural:
**new information, or a venue that supports shorting and deep liquidity.** Knowing exactly where the walls
are — and that they're made of constraints, not randomness — is the precondition for spending the next
effort on something that can actually move the ceiling instead of polishing a model against it.
