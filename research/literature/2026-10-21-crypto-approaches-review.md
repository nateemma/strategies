# Crypto approaches review: untried sources of out-of-sample, cost-aware edge

Date: 2026-10-21. Scope: classifier/signal study for 15m long-only spot alts (SOL, LINK, NEAR, AAVE, AVAX, LTC, BCH, DOT, XRP, ZEC), ~30 bp round trip. Not a P&L discussion.

## Verification and limits (read first)

- Built from ~20 searches/fetches. **[F]** = I fetched the abstract/page this session. **[S]** = search-result snippet only; details unread. Full texts were not read for most items, so effect sizes are as the abstracts report them. Nothing here was checked against this project's data.
- Already covered in the repo and NOT re-reviewed: `research/label-free-study/01-literature-review.md` (direct policy, RL, DFL, generative, TSFM), `research/forecast/literature.md` (Guo et al. 2024, Fischer-Krauss-Deinert 2019, Hudson-Urquhart, Shen-Urquhart-Wang, Cont-Kukanov-Stoikov, Kolm et al.). Where I cite those it is to add newer evidence or to say how a candidate differs.
- **Tested here since (2026-10-21):** items 1 and 2 FAILED as standalone rank signals, and item 3 (Donchian ensemble, `research/donchian_screen`) FAILED vs equal weight in both halves; the 1-day reversal also failed as a gbb selector (`research/reversal_selector`) on the 10 pairs (`research/xsec_screen`, 0/30; momentum is 1-3 day reversal, the reversal book loses to costs). See the dead-ends list in `us_spot_market_study.md`.
- Structural caveat that applies to nearly every cross-sectional paper below: they report long-SHORT portfolios on hundreds or thousands of coins, with alpha concentrated in small illiquid coins. This project is long-only on ~10 liquid large caps, so the usable part is (i) ranking within the 10 and (ii) the long leg. Effect sizes will not transfer one-for-one.

## Ranked candidates (credibility x fit x novelty)

### Strong-to-moderate evidence

**1. Liquidity-conditioned daily momentum/reversal plus residual ("past alpha") momentum, as a within-universe rank signal**
- (a) Claim: short-term reversal dominates the crypto cross-section, but "the handful of largest and most tradeable coins exhibit daily momentum rather than a reversal" (Zaremba et al. 2021) [F]. In an ML cross-section study, the predictive signal "comes mostly from a few simple characteristics, such as market price, past alpha, illiquidity, and momentum", model complexity adds little, and most ML long-short strategies stay profitable after costs, but alphas sit in small, illiquid, volatile coins (Cakici et al. 2024, IRFA) [S]. Dobrynskaya: momentum for sort/hold up to ~2 weeks, reverses beyond ~4 weeks; notes weekly rebalancing costs are material [S, press release/secondary; the 70%/-1000% p.a. figures are unverified].
- (b) Zaremba: >3,600 coins, daily. Cakici: broad coin cross-section. Dobrynskaya: all coins >$1M cap, 2014-2020.
- (c) Costs: Cakici reports post-cost profitability for most ML strategies; Zaremba/Dobrynskaya are mostly gross.
- (d) Replication: reversal-vs-momentum split by liquidity is the reconciling result across conflicting earlier papers. Contrary: Cakici's own caveat that alpha lives in untradeable coins.
- (e) Fit: high on data (15m bars aggregate to any horizon); this project's cross-sectional momentum basket is the best performer, so this is a refinement not a new family. DIFFERENT from what exists: (1) **residual** return after removing BTC beta ("past alpha") rather than raw return rank, (2) **illiquidity** (Amihud from 15m bars, taker-buy share) as a rank key, (3) explicit lookback/hold grid capped at ~2 weeks, which tests whether the basket's lookback sits in the momentum regime.
- (f) Afternoon screen: on the 10 pairs, daily/weekly rebalanced rank IC (Spearman) of {raw 1d, 3d, 7d, 14d return; BTC-residual return; Amihud; 15m-derived range/volume} vs next 1d/3d/7d return, per fold, with a two-half persistence split. Pass bar: |IC| t>3 (Harvey-Liu-Zhu) and same sign in both halves; then check the long-top-tercile spread against 30 bp x turnover.

**2. Realized variance and signed-jump cross-sectional predictors computed from 15m bars**
- (a) Higher realized variance predicts lower returns over the following weeks; the negative predictability comes from the positive (upside) jump and jump-robust components; strongest in small, illiquid, retail-heavy coins, interpreted as lottery demand (Lee & Wang, JFQA 2025) [S: [IDEAS](https://ideas.repec.org/a/cup/jfinqa/v60y2025i4p1859-1890_9.html)]. Zhang & Zhao: signed jump variation is ~18% of quadratic variation and realized signed jumps predict cross-sectional excess returns; premium survives standard factor controls [S: [SSRN 3910202](https://papers.ssrn.com/abstract=3910202), also published as Finance Research/ IRFA 2023 "Good volatility, bad volatility..." [IDEAS](https://ideas.repec.org/a/eee/finana/v89y2023ics1057521923002284.html)].
- (b) Intraday data, many coins; weekly horizon.
- (c) Costs/OOS: not confirmed from abstracts. Lee & Wang is JFQA (peer-reviewed). The effect's concentration in illiquid coins is the main threat to transfer.
- (d) Contrary: lottery/MAX literature is mixed (positive MAX premium in Ozdamar et al. 2021 [S]; a Liege thesis finds a TMAX strategy loses [S, unreviewed]). Signs differ across papers, so sign must be tested, not assumed.
- (e) Fit: high. Differs from existing work: the repo's `bb_width` / `atr_pct_roll` are volatility LEVEL features inside a reversion trigger; this is cross-sectional ranking on realized jump components at weekly horizon, used as a long-only underweight/avoid filter or sizer.
- (f) Screen: weekly rank IC of RV, positive/negative semivariance, signed jump (Barndorff-Nielsen-Shephard style from 15m returns) vs next-week return across the 10 pairs; compare with the existing width-sort baseline (every model change must beat it per project notes).

**3. Donchian multi-lookback ensemble with volatility sizing and a turnover-reducing portfolio technique**
- (a) Rotating portfolio of the 20 most liquid coins, net-of-fee Sharpe "above 1.5", 10.8% annualised alpha vs Bitcoin (Zarattini, Pagani, Barbon, SFI RP 25-80, 2025) [F: [IDEAS](https://ideas.repec.org/p/chf/rpseri/rp2580.html)].
- (b) Survivorship-bias-free all-coin dataset from 2015; end date, fee assumption and OOS design not visible on the abstract page.
- (c) Abstract says costs are assessed and a mitigation is proposed; numbers unverified. Authors are practitioner-affiliated; treat as unreplicated.
- (d) A 2026 preprint on vol-scaled crypto trend-following reports OOS Sharpe 1.83 vs 0.65 for TSMOM-1M over 2022-2024 with 4 bp fee, calibrated slippage and funding, but its table columns look misaligned [S: [arXiv 2602.11708](https://arxiv.org/pdf/2602.11708)]. Monash note: shorter lookbacks give high Sharpe but costs bite via rebalancing [S].
- (e) Fit: medium. The repo already has breakout momentum and a cross-sectional momentum basket; what is new is the ensemble of lookbacks, volatility-parity sizing, and the cost-reduction rule. Daily horizon matches the 30 bp budget (the one regime where cost is not binding).
- (f) Screen: daily Donchian-ensemble signal on the 10 pairs, long-only, inverse-vol weights, with/without a no-trade band; compare Sharpe and turnover with the live basket over the same folds. Use the project's existing walk-forward harness, no new infrastructure.

**4. Cross-coin lagged returns (Guo, Sang, Tu, Wang) - mostly tried; only a narrow variant is new**
- (a) Past returns of other coins predict a focal coin; long-short OOS return after transaction costs; mechanism = information diffusion with limited attention (JEDC 163, 2024) [F: [IDEAS](https://ideas.repec.org/a/eee/dyncon/v163y2024ics0165188924000551.html)]. The existing repo note says the effect lasts ~10 minutes and quotes a 2.16% daily working-paper figure not verified in the journal version.
- (c)/(d): peer-reviewed; but the horizon is shorter than a 15m bar, so the 15m project can capture at most the tail. 038 follow-ups already tested cross-asset and got FAIL/INCONCLUSIVE.
- (e) Only new variant: pooled across the 10 pairs (not per-asset), BTC/ETH lag as a cross-sectional rank key at 1-4 bar horizons. Low expected value given cost. Screen: IC of lagged other-coin return at lags 1-4 bars on next-bar return; abandon if |IC| < cost-implied threshold (a 30 bp round trip needs a multi-bar move).

### Weak / promising (single preprint, no cost, or horizon-cost mismatch)

**5. Quarter-hour opening order imbalance predicting 4-12 hour returns (Kim & Hansen 2026)**
- (a) Opening order imbalance at quarter-hour marks predicts returns 4-12 h ahead; one- and five-minute imbalances have negligible power. Lagged-flow component 5-6 bp stable across horizons; public-signal component rises from <1 bp at 4 h to 9.8 and 16.9 bp at 8 and 12 h. Walk-forward LASSO OOS R-squared averages 3.37% for the lag+indicator model [F: [arXiv 2607.09426](https://arxiv.org/abs/2607.09426)].
- (b) Six Binance USDT perpetuals (BTC, ETH, XRP, SOL, DOGE, ADA), 2021-01 to 2024-10, OOS from 2021-07.
- (c) Costs: authors state the forecast is an input to execution, not a net-of-cost strategy; mean sign-weighted target ~0.5 bp per boundary, about a tenth of one 5 bp taker fee. No net backtest.
- (d) Preprint; no replication yet.
- (e) Fit: partial. A 15m kline's taker-buy minus taker-sell volume IS the quarter-hour order imbalance, so the data is in hand. But the project already screened taker-buy features (S1b passed MS_VR/OF_TRG, no sizer win). New here: trailing multi-bar (4-12 h) aggregation, the quarter-hour clock-phase structure, and forward horizon 16-48 bars.
- (f) Screen: IC of 15m signed imbalance (and its 16-48 bar sum) vs forward 16-48 bar return, per pair and pooled; compare the realised gross move to 30 bp. Expect gross edge of order 10-20 bp, i.e. below cost unless combined with a trigger; treat as a feature test only.

**6. Crypto carry (futures basis) as a crash-risk gate**
- (a) High carry predicts future price crashes and co-moves with crash-insurance price; arbitrage capital is scarce and leveraged retail demand drives the basis (Schmeling, Schrimpf, Todorov, BIS WP 1087, rev. Oct 2025; listed as Management Science, DOI 10.1287/mnsc.2024.05069 - publication not confirmed) [S: [BIS](https://bis.org/publ/work1087.htm)].
- (c) A predictability result, not a traded net-of-cost strategy; regression specifics unverified.
- (e) Fit: moderate. The repo has funding (weak) and a vol-regime gate; basis/premium index (available) as a cross-sectional or market-wide risk-off signal differs from funding-level screens. Only plausible as an exit/size gate, not an entry signal. Funding-level was already rejected; check whether premium-index extremes add anything beyond the volatility regime.
- (f) Screen: next-1d/3d drawdown (min forward return) conditioned on premium-index top decile per pair, from 2021-12; compare to the same conditioning on trailing realised vol; keep only if it adds beyond vol.

**7. Factor momentum as a meta-timing layer (Fieberg, Liedtke, Metko, Zaremba, Quant. Finance 23(12), 2023)**
- (a) 34 anomalies, >3,900 coins 2014-2022: past winner factors continue to outperform; factor-level autocorrelation is mostly size and volatility anomalies [S: [IDEAS](https://ideas.repec.org/a/taf/quantf/v23y2023i12p1853-1869.html)]. (c) cost handling not confirmed.
- (e) Fit: low-moderate; needs several rank factors from items 1-2 and only 10 coins, so the factor returns are noisy. Screen only after items 1-2 produce at least two factors with stable IC: test whether last-month factor spread predicts next-month spread.

**8. LLM-constrained factor discovery (Huang et al., arXiv 2604.26747, Apr 2026)**
- (a) LLM agent proposes factors in a point-in-time DSL; fixed splits and costs enforced by an engine; trained on 2020-2022, pure OOS 2024-2026; 5 bp one-way cost; ridge-combined portfolio 44.55% annualised, Sharpe 1.55 [F]. Factor identities, universe and fee-sensitivity not visible in the abstract.
- (c)/(d) single preprint, one OOS window, long-short. Useful as a process template (frozen splits, deterministic cost engine, auditable negative results), which the project already largely has. Not a source of a ready signal.

### Weak: mixed or no durable evidence

| Item | Evidence | Why weak for us |
|---|---|---|
| Hour-of-day / time-of-day seasonality | One Bitcoin study (15M+ observations, seven exchanges) finds time-varying effects with no persistent pattern; a Quantpedia hourly Gemini study (2015-2022) reports 21:00-23:00 UTC strength, which is a backtested rule, not a validated forecast [S: [UWA](https://research-repository.uwa.edu.au/en/publications/bitcoin-time-of-day-day-of-week-and-month-of-year-effects-in-retu/), [Quantpedia](https://quantpedia.com/are-there-seasonal-intraday-or-overnight-anomalies-in-bitcoin/)]. A functional-data study ties the common intraday periodicity to NYSE/LSE/HKEX hours [S: [York](https://yorkspace.library.yorku.ca/items/ed6af3fc-f18f-4db6-9468-88b794a25910)]. | Project already rejected session and day-of-week features. Weekend and funding-settlement-time effects: no study found (search returned only definitions). |
| On-chain net flows | ETH exchange net inflows negatively predict ETH returns/volatility intraday; USDT inflows positively predict BTC/ETH returns; 2017-2023; no OOS or cost reported (Chi, Chu, Hao, arXiv 2411.06327) [F]. | Data not on Binance klines; covers BTC/ETH/USDT only, not your alts; no OOS. |
| MAX / idiosyncratic skew | Positive MAX-return link (Ozdamar et al. 2021, Financial Innovation) [S]; >1.5% weekly low-minus-high MAX spread in another paper [S]; contradicted by a thesis [S]. | Sign unstable, effect in small coins; subsumed by item 2's variance measures. |
| Open-interest changes | Only practitioner event studies found (e.g. K33 on a 12.46% one-day BTC OI drop; unverified). Data-quality warning: OI in BTC perpetuals is systematically misquoted by some large venues (Giagkiozis & Said, arXiv 2310.14973) [S]. | No rigorous OOS study found; vendor-quality OI plus a 2021-12 start give few folds. Treat OI/LS ratio as unproven. |

### Debunked / not robust after costs or OOS

| Item | Finding | Source |
|---|---|---|
| Cointegration/distance pairs trading on Binance | Gross edge at 5-min frequency turns negative net of costs in the copula paper; IEEE Access (26 Binance coins) strategies underperform benchmarks and are parameter-sensitive; theses conflict | [Financial Innovation copula tables](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00702-7/tables/7), [IEEE 9200323](https://ieeexplore.ieee.org/document/9200323) [S] |
| ChatGPT news sentiment for Bitcoin | In-sample fit improves; no statistically significant out-of-sample forecasting gain | Polish-team study via search summary [S] (authors/venue not captured) |
| Short-holding ML quantile long-short on BTC | Up to 39% monthly before costs, negative after costs due to short holds | 2021 GRU/LSTM/RF/GB study cited in [Cakici search result](https://ies.fsv.cuni.cz/sites/default/files/uploads/files/wp_2021_02_nechvatalova.pdf) [S] |
| Zero-shot TSFMs, generative augmentation, naive RL | Already covered in repo review; no new contrary evidence found | `01-literature-review.md` |
| Vol-managed portfolios | No crypto-specific peer-reviewed OOS test found; in equities, Cederburg et al. find gains not achievable OOS and Barroso-Detzel find costs erode them; DeMiguel et al. (J. Finance 2024) says a conditional multifactor version survives [S]. Project already runs a vol overlay. | no crypto evidence |

## Items searched that returned nothing usable

- VPIN for Binance returns: no peer-reviewed predictive test found (a Padova thesis only motivates it; a Substack gives a 0.7 "toxic" threshold, unverified).
- BTC/ETH-leads-alts intraday: only hourly practitioner notes (Robot Wealth, "probably not exploitable in a slower strategy") and a daily-data Granger chapter; Li et al. (arXiv 2205.00974) use DTW lead-lag to forecast Bitcoin, not a trading test [S].
- Risk parity / HRP / Kelly in crypto: nothing crypto-specific with OOS and cost found in this pass. 1/N performs well against mean-variance in crypto per a search summary [S], consistent with the project's equal/inverse-width-style sizers. Not pursued further.
- Weekend and funding-settlement effects: no study found.

## Suggested order of afternoon screens

1. Item 1 and item 2 rank-IC panels share one dataset (10 pairs x daily/weekly rebalance, features from 15m bars); run together, one two-half persistence split, multiple-testing adjusted (HLZ t>3).
2. Item 3 daily Donchian ensemble vs the live basket on the same folds.
3. Item 5 flow-sum IC at 16-48 bars (cheap, data in hand; expected below cost).
4. Item 6 only if 1-2 find a usable gate target.

## References

- Zaremba, Bilgin, Long, Mercik, Szczygielski (2021). Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets. Int. Rev. Financial Analysis 78, 101908. https://earsiv.medeniyet.edu.tr/items/23a96c9f-3952-4bdf-aec5-abafb3aaab70
- Cakici, Shahzad, Bedowska-Sojka, Zaremba (2024). Machine learning and the cross-section of cryptocurrency returns. Int. Rev. Financial Analysis. https://open.icm.edu.pl/handle/123456789/25543
- Dobrynskaya, V. Cryptocurrency momentum and reversal (HSE; SSRN). https://www.hse.ru/en/news/research/559533243.html
- Lee, Wang (2025). Variance decomposition and cryptocurrency return prediction. JFQA 60(4), 1859-1890. https://ideas.repec.org/a/cup/jfinqa/v60y2025i4p1859-1890_9.html
- Zhang, Zhao. Good volatility, bad volatility, and the cross section of cryptocurrency returns. https://papers.ssrn.com/abstract=3910202
- Zarattini, Pagani, Barbon (2025). Catching Crypto Trends. SFI RP 25-80. https://ideas.repec.org/p/chf/rpseri/rp2580.html
- Guo, Sang, Tu, Wang (2024). Cross-cryptocurrency return predictability. JEDC 163, 104863. https://doi.org/10.1016/j.jedc.2024.104863
- Kim, Hansen (2026). The Quarter-Hour Effect. arXiv:2607.09426. https://arxiv.org/abs/2607.09426
- Schmeling, Schrimpf, Todorov. Crypto carry. BIS WP 1087. https://bis.org/publ/work1087.htm
- Fieberg, Liedtke, Metko, Zaremba (2023). Cryptocurrency factor momentum. Quantitative Finance 23(12), 1853-1869. https://ideas.repec.org/a/taf/quantf/v23y2023i12p1853-1869.html
- Liu, Tsyvinski, Wu (2022). Common risk factors in cryptocurrency. J. Finance (market, size, momentum three-factor model) [S]. https://www.nber.org/papers/w25882.pdf
- Huang, Fan, Hu, Ye (2026). From Hypotheses to Factors: Constrained LLM Agents in Cryptocurrency Markets. arXiv:2604.26747. https://arxiv.org/abs/2604.26747
- Chi, Chu, Hao. Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets. arXiv:2411.06327. https://arxiv.org/abs/2411.06327
- Giagkiozis, Said (2023). Reconciling Open Interest with Traded Volume in Perpetual Swaps. arXiv:2310.14973. https://arxiv.org/html/2310.14973v2
- Systematic trend-following with adaptive portfolio construction in crypto (2026). arXiv:2602.11708. https://arxiv.org/pdf/2602.11708
- Mercik, Zaremba, Demir, Crypto factor zoo (working draft; liquidity variables dominate, only bid-ask spread stable across halves) [S]. https://open.icm.edu.pl/bitstreams/21636cf6-61df-4fb5-9ae4-4195d592cd26/download
