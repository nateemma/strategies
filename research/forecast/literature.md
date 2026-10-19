# Literature review: forecasting N-candle log returns of crypto (feature 038 follow-ups)

Date: 2026-10-19. Scope: classifier/forecaster study only; no trading discussion beyond cost-aware evaluation of forecasts.

Verification legend used throughout:
- **[V-web]** = checked against a web record (publisher/repository/IDEAS/arXiv) during this review.
- **[V-kb]** = bibliographic details match my standard knowledge of the reference but I did NOT re-fetch a record this session. Treat as likely-correct, not proven.
- **[UNVERIFIED]** = could not confirm.

## Summary

- The project's null result is the expected one. Return forecasting has very low out-of-sample R^2 even in equities with rich predictors (Welch & Goyal; Campbell & Thompson; Gu-Kelly-Xiu); ~1% R^2 at N=1 is in the range the literature considers economically meaningful only when the signal is cheap to trade, which a 30 bp round trip at 15-minute turnover is not.
- Trend/momentum beating random entry by 10-25 bp per trade at N >= 16 matches crypto TSMOM evidence (Moskowitz et al. for the concept; Shen-Urquhart-Wang and Borgards for intraday crypto). The edge scales with horizon, so the gross edge needed (>30 bp) argues for testing N beyond 96 candles or fewer, longer-hold trades, rather than more complex models at the same horizons.
- The only crypto intraday predictability that survives stated costs in published work is cross-sectional / cross-asset (Guo-Sang-Tu-Wang JEDC 2024 lagged other-coin returns up to 10 min; Fischer-Krauss-Deinert 120-min RF). Both use own-coin-excluded or relative targets and many coins; this project's single-asset own-history inputs are the weakest version. Cross-asset lagged returns (e.g. BTC lead into the 10 altcoins) is the cheapest new-information test.
- Order-flow / order-book imbalance is the best-documented source of genuinely new short-horizon information (Cont-Kukanov-Stoikov; Kolm-Turiel-Westray), but its horizon is seconds to minutes and gains decay quickly; at 15-min bars it needs aggregated imbalance (trade-signed volume, taker buy ratio) rather than book snapshots. Binance klines already carry taker-buy volume.
- Decomposition features (EMD/VMD/wavelet) have a documented leakage problem: decomposing the full series before the split yields spuriously good forecasts that vanish when decomposition is causal (Hasumi & Kajita 2018; "Bias effect on predicting market trends with EMD" 2017). The project's causal a-trous approach is the right remedy; the SWT failure is consistent with "no real gain once causal".
- FFT extrapolation is periodic by construction (the DFT implies periodic extension of the window); off-grid sinusoid fitting (Lomb-Scargle / nonlinear least squares) removes the on-grid restriction but not the fact that spectral peaks in noisy returns are mostly noise; expect no gain.
- Voting/averaging: equal-weight combination is hard to beat (forecast combination puzzle), but gains require members with partly independent errors; votes among correlated trend models add little beyond the best member. Check pairwise error correlation before building more members.
- With dozens of models x 96 horizons x halves tried, report multiple-testing-adjusted results (Harvey-Liu-Zhu t > 3; Bailey-Lopez de Prado deflated Sharpe; Hudson-Urquhart's use of data-snooping-robust tests in crypto). The two-half persistence rule already used is consistent with this.

---

## a. Intraday / high-frequency crypto return predictability

1. Guo, L., Sang, B., Tu, J., Wang, Y. (2024). "Cross-cryptocurrency return predictability." *Journal of Economic Dynamics and Control* 163, 104863. https://ideas.repec.org/a/eee/dyncon/v163y2024ics0165188924000551.html  [V-web]
   Minute-level Binance data: lagged returns of other coins predict a focal coin's returns for up to ~10 minutes; robust to adaptive LASSO and PCA. A working-paper version reports a 2.16% daily out-of-sample long-short return after costs (figure from the SMU working-paper abstract; not checked in the JEDC version, so do not rely on it).
   Implication: the strongest published intraday crypto result is cross-asset; test lagged returns of the other 9 pairs (and BTC) as features, but note the horizon (minutes) is shorter than 15-min bars.

2. Fischer, T., Krauss, C., Deinert, A. (2019). "Statistical Arbitrage in Cryptocurrency Markets." *Journal of Risk and Financial Management* 12(1), 31. https://ideas.repec.org/a/gam/jjrfmx/v12y2019i1p31-d205554.html  [V-web]
   Random forest on lagged returns of 40 coins predicting 120-minute outperformance vs cross-sectional median; 7.1 bp/day after 15 bp per half-turn, 18 Jun-17 Sep 2018 only. Authors caution about volume constraints.
   Implication: a relative (cross-sectional) target is easier than an absolute one; short sample, so weak evidence. Suggests a cross-sectional rank forecast variant.

3. Hudson, R., Urquhart, A. (2021). "Technical trading and cryptocurrencies." *Annals of Operations Research* 297, 191-220 (online 2019). https://doi.org/10.1007/s10479-019-03357-1  [V-web]
   ~15,000 technical rules; significant in-sample predictability, uses multiple-hypothesis corrections; out-of-sample predictability does not hold for Bitcoin though it persists for other coins; breakeven costs exceed typical crypto costs (daily-style rules, not intraday).
   Implication: supports the two-half persistence design and indicates altcoins are the more plausible venue for rule edges than BTC.

4. Wang, H. (2025). "Better Inputs Matter More Than Stacking Another Hidden Layer" (BTC/USDT order-book benchmark). arXiv:2506.05764. https://arxiv.org/abs/2506.05764  [V-web, search snippet only; accuracy numbers not read]
   Finds input preprocessing/feature engineering matters more than model depth on sub-second to seconds BTC order-book snapshots.
   Implication: consistent with the project's "LightGBM and deeper models do not rescue weak inputs" finding.

5. Recent on-chain-flow study: "Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets", arXiv:2411.06327. https://arxiv.org/pdf/2411.06327  [V-web, snippet only]
   Reports ETH net inflows negatively predict ETH returns and volatility intraday.
   Implication: non-price information is where predictability is claimed; not available on binance.us klines, so low priority.

## b. Time-series momentum / trend in crypto, intraday to daily

1. Shen, D., Urquhart, A., Wang, P. (2022). "Bitcoin intraday time series momentum." *Financial Review* 57(2), 319-344. https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/  [V-web]
   First half-hour (volume-clock) return predicts last half-hour return; attributed to liquidity provision rather than late-informed trading.
   Implication: intraday momentum is real but short-lived (sub-day windows); consistent with trend models working at N >= 16 gross but too small vs 30 bp.

2. Borgards, O. (2021). "Momentum in cryptocurrencies: interday and intraday." *North American Journal of Economics and Finance* 57. https://econpapers.repec.org/RePEc:eee:ecofin:v:57:y:2021:i:c:s1062940821000590  [V-web record located; exact title wording as given here is from memory and should be checked at the link: UNVERIFIED title]
   Tests momentum for twenty cryptocurrencies at interday and several intraday frequencies; reports larger and longer-lasting momentum than in other assets.
   Implication: motivates horizon sweep beyond 1 day for the same signals.

3. Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). "Time series momentum." *Journal of Financial Economics* 104(2), 228-250. [V-kb] https://doi.org/10.1016/j.jfineco.2011.11.003
   Past 1-12-month own-return sign predicts next-month return across 58 liquid instruments.
   Implication: the canonical TSMOM; horizon is months, so intraday transfer is by analogy only.

4. Liu, Y., Tsyvinski, A. (2021). "Risks and Returns of Cryptocurrency." *Review of Financial Studies* 34(6), 2689-2727. [V-kb] https://doi.org/10.1093/rfs/hhaa113
   Crypto returns show time-series momentum and attention-based predictors at daily-to-weekly horizons.
   Implication: supports longer horizons than 1 day for any own-return signal.

## c. Order flow / order book imbalance and short-horizon returns

1. Cont, R., Kukanov, A., Stoikov, S. (2014). "The price impact of order book events." *Journal of Financial Econometrics* 12(1), 47-88. [V-kb] https://doi.org/10.1093/jjfinec/nbt003
   Price changes over short intervals are mainly explained by linear order flow imbalance (OFI) at the best quotes, with depth-scaled impact.
   Implication: OFI is the canonical new-information input; at 15 minutes use aggregated signed volume as proxy.

2. Kolm, P.N., Turiel, J., Westray, N. (2023). "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book." *Mathematical Finance* 33(4), 1044-1081. https://doi.org/10.1111/mafi.12413  [V-web]
   Deep models on multi-level OFI for 115 NASDAQ stocks outperform raw-book inputs; per a secondary summary, accuracy peaks at short event horizons then declines (not confirmed in the paper itself).
   Implication: OFI inputs beat raw book; the horizon-decay point says do not expect it to reach N >= 16 candles.

3. "Exploring microstructural dynamics in cryptocurrency limit order books" (2025), arXiv:2506.05764 (same as a.4 above). https://arxiv.org/html/2506.05764v2  [V-web]
   Crypto LOB predictability comes from microstructure imbalances; simple models on good features rival deep nets.
   Implication: if order-book data are ever collected (the live recorder exists), a linear OFI baseline comes first.

4. Compact cross-asset OFI feature-set study, arXiv:2602.00776 (Jan 2026). https://arxiv.org/pdf/2602.00776  [V-web, snippet only; authors/results not read]
   Tests whether a small feature set including OFI predicts short-horizon returns across crypto assets.
   Implication: candidate to read in full before designing an OFI feature test.

## d. Wavelet / decomposition forecasting and look-ahead leakage

1. Hasumi, R., Kajita, Y. (2018). "Boundary problem and data leakage: A caveat for wavelet-based forecasting." JCER Research Staff Report / Discussion Paper 148. https://www.jcer.or.jp/?p=38548  [V-web]
   Improper wavelet processing (using non-causal boundary handling / whole-series transforms) leaks unobserved data and gives unrealistically accurate forecasts; with correct processing wavelet forecasts do not beat a naive prediction.
   Implication: directly explains why SWT/FFT feature gains, if seen in whole-series transforms, would be illusory; the project's windowed causal transforms are the right protocol, and the null result matches this paper.

2. "Bias effect on predicting market trends with EMD." *Expert Systems with Applications* (2017), DOI 10.1016/j.eswa.2017.03.053. https://www.datalearner.com/academic/journal-papers/0957-4174/volumes-and-issues/286/paper-detail/13712  [V-web record; authors not retrieved: UNVERIFIED authors]
   Earlier positive EEMD results on market indices came from pre-processing the whole series with EMD; with a bias-free protocol on 8 indices and 4 models, EMD/EEMD gave no meaningful improvement.
   Implication: same lesson for EMD/VMD; never decompose beyond the information set at forecast time.

3. Sliding-window EMD variants for leakage-free forecasting, Scientific Reports (Nov 2024) (SW-EMD, STMP-EMD, MTMP-EMD; water-quality application). https://doaj.org/article/4e5e24ec3bdc476e891900834f47eff5  [V-web snippet; authors/title not retrieved: UNVERIFIED]
   Re-decompose in a sliding window so each forecast uses only past data.
   Implication: gives a template for causal re-decomposition at every forecast origin (which the windowed protocol already does).

4. Moving-fronts + decomposition + deep learning, arXiv:2303.06394. https://arxiv.org/abs/2303.06394  [V-web snippet]
   Describes the "one-time decomposition then split" leak and notes new data points change earlier components (end-effect).
   Implication: the end-effect is the key point for a-trous: only one-sided (causal) filters keep past coefficients fixed.

5. Renaud, O., Starck, J.-L., Murtagh, F. (2005). "Wavelet-based combined signal filtering and prediction." *IEEE Trans. Systems, Man, and Cybernetics, Part B* 35(6), 1241-1251. https://doi.org/10.1109/TSMCB.2005.850182  [V-web] (see core references.)
   Multiresolution a-trous scheme: forecast each scale from a few past coefficients, in the spirit of a Kalman filter, with noise filtering built in.
   Implication: the method behind the running Haar a-trous test; the original shows gains on smooth/low-noise series, which crypto returns are not.

## e. Harmonic / sinusoidal extrapolation and FFT periodicity

- Why FFT extrapolation is periodic: the DFT of a length-L window represents it as a sum of L-periodic harmonics at the grid frequencies k/L, so any synthesis beyond the window repeats the window (the first forecast step is the window's first sample, not a continuation). This is textbook DSP (e.g. Oppenheim & Schafer, *Discrete-Time Signal Processing*); I did not retrieve a web source, so it is stated as standard theory [V-kb].
- Off-grid frequencies: Lomb-Scargle is the least-squares fit of a sinusoid (plus offset) at each trial frequency; it equals the Fourier periodogram only for regular sampling at Fourier frequencies. Source: Zechmeister/Kuerster-type formalism, "Unevenly-sampled signals: a general formalism of the Lomb-Scargle periodogram", arXiv:1006.2473. https://arxiv.org/pdf/1006.2473  [V-web]. Also VanderPlas, "Understanding the Lomb-Scargle periodogram," ApJS 236 (2018), arXiv:1703.09824 [V-kb, not re-fetched].
- Frequency resolution is limited by the window length (peak width ~ 1/baseline; Astropy Lomb-Scargle docs https://astropy.readthedocs.io/en/stable/timeseries/lombscargle.html [V-web]), so a 64-candle window cannot separate close periods and off-grid fits will be unstable.
- Implication for this project: off-grid fitting fixes the grid-mismatch bias, not the fundamental lack of stable periodicities in returns; a null here would be unsurprising. If tried, penalise or cap the number of components and judge out-of-window phase coherence across windows (does the fitted period persist?).
- Ehlers (2001, 2013) cycle indicators (see core references) are the practitioner version of the same idea; no peer-reviewed out-of-sample support found in this review.

## f. Forecast combination / voting with correlated members

1. Claeskens, G., Magnus, J.R., Vasnev, A.L., Wang, W. (2016). "The forecast combination puzzle: A simple theoretical explanation." *International Journal of Forecasting* 32(3), 754-762. https://research.vu.nl/en/publications/the-forecast-combination-puzzle-a-simple-theoretical-explanation-2/  [V-web]
   Estimated optimal weights are random, which adds bias and variance, so the simple average often beats estimated-weight combinations.
   Implication: use equal weights for N-of-M votes; do not fit vote weights on the sample.

2. Wang, X., Hyndman, R.J., Li, F., Kang, Y. (2023). "Forecast combinations: An over 50-year review." *International Journal of Forecasting* 39(4), 1518-1547; arXiv:2205.04216. https://arxiv.org/pdf/2205.04216  [V-web arXiv; IJF volume/pages from memory]
   Survey of combination methods; diversity of member errors is what drives gains.
   Implication: with members that are all trend/smoothing variants, errors are highly correlated, so expect the vote to equal its best member.

3. Smith, J., Wallis, K.F. (2009). "A simple explanation of the forecast combination puzzle." *Oxford Bulletin of Economics and Statistics* 71(3), 331-355. [V-kb; cited in the above sources as attributing the puzzle to weight-estimation error]
   Implication: same as f.1.

4. Bates & Granger (1969) and Timmermann (2006): see core references. Combination gain for two forecasts shrinks as error correlation rises; with correlation near 1 there is none.

## g. Economic evaluation of forecasts and multiple-testing

1. Harvey, C.R., Liu, Y., Zhu, H. (2016). "... and the Cross-Section of Expected Returns." *Review of Financial Studies* 29(1), 5-68 (NBER WP 20592). https://www.nber.org/papers/w20592  [V-web NBER; RFS volume from memory]
   With hundreds of tested factors, a new factor needs t > 3.0, not 2.0.
   Implication: with many models x horizons x halves tried here, a single t ~ 2 at one horizon is not evidence.

2. Harvey, C.R., Liu, Y. (2015). "Backtesting." *Journal of Portfolio Management* 42(1), 13-28. https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF  [V-web]; and Harvey & Liu (2014), "Evaluating trading strategies," *JPM* 40(5), 108-118. https://people.duke.edu/~charvey/Research/Published_Papers/P116_Evaluating_trading_strategies.pdf  [V-web]
   Nonlinear Sharpe haircut for multiple testing: marginal strategies are penalised heavily, high-Sharpe ones moderately.
   Implication: marginal 10-25 bp edges are the ones most exposed to a haircut.

3. Bailey, D.H., Borwein, J.M., Lopez de Prado, M., Zhu, Q.J. (2014). "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." *Notices of the AMS* 61(5), 458-471. https://scholarworks.wmich.edu/math_pubs/40/  [V-web]
   Trying many configurations makes strong backtests easy to find; with memory effects, expected out-of-sample is negative rather than zero. Report the number of trials.
   Implication: log the full trial count of feature 038 (all models x N) and apply a deflation.

4. Hudson & Urquhart (2021), see a.3: example of data-snooping-robust testing in crypto rules.

5. Makridakis et al. M6 (see core references): forecasts and investment decisions are only weakly linked; consistent outperformance was very hard.
   Implication: forecast accuracy metrics (R^2, MCC) should not be taken as proxies for cost-aware edge, matching the project's "predictability is not edge" stance.

6. Statistical comparison of forecasts: Diebold & Mariano (1995), *J. Business & Economic Statistics* 13(3), 253-263 [V-kb]; Clark & West (2007), *Journal of Econometrics* 138(1), 291-311 [V-kb]; White (2000) Reality Check, *Econometrica* 68(5), 1097-1126 [V-kb]; Hansen (2005) SPA, *J. Business & Economic Statistics* 23(4), 365-380 [V-kb]. A web search this session confirmed only the DM test's properties (size distortion in small samples, power loss with dependence); the others are not web-checked.
   Implication: use block bootstrap / Reality-Check style tests across the model set, and Clark-West for nested comparisons against a benchmark such as the zero-mean forecast (relevant to the N=1 R^2 ~1% finding).

---

## Verified core references

| # | Citation as supplied | Status | Corrected / full citation |
|---|---|---|---|
| 1 | Welch & Goyal (2008), RFS | [V-kb] correct | Welch, I., Goyal, A. (2008). "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction." *RFS* 21(4), 1455-1508. https://doi.org/10.1093/rfs/hhm014 |
| 2 | Campbell & Thompson (2008), RFS | [V-kb] correct | Campbell, J.Y., Thompson, S.B. (2008). "Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?" *RFS* 21(4), 1509-1531. https://doi.org/10.1093/rfs/hhm055 |
| 3 | Gu, Kelly & Xiu (2020), RFS | [V-kb] correct | Gu, S., Kelly, B., Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *RFS* 33(5), 2223-2273. https://doi.org/10.1093/rfs/hhaa009 |
| 4 | Makridakis et al., M6, IJF | [V-web] correct, with year/volume detail | Makridakis, S. et al. "The M6 forecasting competition: Bridging the gap between forecasting and investment decisions." *IJF* 41(4), 1315-1354 (2025; online Nov 2024); arXiv:2310.13357. https://arxiv.org/abs/2310.13357 . Cite as (2024/2025), not 2023. A separate, earlier M6 design/organisational paper may exist; not verified. |
| 5 | Moskowitz, Ooi & Pedersen (2012), JFE | [V-kb] correct | *JFE* 104(2), 228-250. https://doi.org/10.1016/j.jfineco.2011.11.003 |
| 6 | Liu & Tsyvinski (2021), RFS | [V-kb] correct | *RFS* 34(6), 2689-2727. https://doi.org/10.1093/rfs/hhaa113 |
| 7 | Cont, Kukanov & Stoikov (2014), JFEC | [V-kb] correct | *J. Financial Econometrics* 12(1), 47-88. https://doi.org/10.1093/jjfinec/nbt003 |
| 8 | Renaud, Starck & Murtagh (2005), IEEE SMC-B | [V-web] correct | *IEEE Trans. SMC-B (Cybernetics)* 35(6), 1241-1251, Dec 2005. DOI 10.1109/TSMCB.2005.850182. Author order confirmed against the published PDF; some repository entries list Murtagh first. |
| 9 | Percival & Walden (2000) | [V-kb] correct | *Wavelet Methods for Time Series Analysis*, Cambridge University Press. https://doi.org/10.1017/CBO9780511841040 |
| 10 | Gencay, Selcuk & Whitcher (2002) | [V-kb] correct | Gencay, R., Selcuk, F., Whitcher, B. *An Introduction to Wavelets and Other Filtering Methods in Finance and Economics*, Academic Press (Elsevier). |
| 11 | Ehlers (2001); Ehlers (2013) | [V-kb] correct | Ehlers, J.F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications.* Wiley. Ehlers, J.F. (2013). *Cycle Analytics for Traders.* Wiley. Practitioner books; not peer-reviewed. |
| 12 | Bates & Granger (1969) | [V-kb] correct | "The Combination of Forecasts." *Operational Research Quarterly* 20(4), 451-468. https://doi.org/10.2307/3008764 |
| 13 | Timmermann (2006) | [V-kb] correct | "Forecast Combinations," in Elliott, Granger, Timmermann (eds.), *Handbook of Economic Forecasting*, vol. 1, ch. 4, pp. 135-196, Elsevier. |
| 14 | Hyndman & Athanasopoulos | [V-kb] correct | *Forecasting: Principles and Practice*, 3rd ed. (2021), OTexts. https://otexts.com/fpp3/ (2nd ed. 2018 also exists; state the edition) |
| 15 | Zeng et al. (2023), AAAI | [V-kb] correct | Zeng, A., Chen, M., Zhang, L., Xu, Q. "Are Transformers Effective for Time Series Forecasting?" *AAAI 2023* (Proc. AAAI 37(9), 11121-11128). arXiv:2205.13504. Shows a one-layer linear model (DLinear) beating Transformer forecasters; relevant as a "linear/Ridge baseline first" argument. |
| 16 | Lopez de Prado (2018); Bailey & Lopez de Prado (2014) | [V-kb] correct | Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. Bailey, D.H., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management* 40(5), 94-107. https://doi.org/10.3905/jpm.2014.40.5.094 |

Corrections to the supplied list: (a) M6 is 2024/2025 (IJF 41(4)), not 2023; (b) Welch & Goyal's full title is "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction"; (c) the Lopez de Prado pairing is two separate works (a book and a JPM paper); (d) Renaud et al. author order is as given.

Relevance of the core wavelet and combination texts: Percival & Walden and Gencay et al. give the MODWT/a-trous definitions and the causality caveat (boundary coefficients depend on the filter width, so only the last L-1 samples are affected: this is the end-effect to handle); Bates & Granger and Timmermann give the combination theory behind f.

## Citations that could not be verified

- Borgards (2021) exact title (record located via search; title wording not confirmed). Check the econpapers link above.
- Authors of "Bias effect on predicting market trends with EMD" (ESWA 2017, DOI 10.1016/j.eswa.2017.03.053).
- Authors/title of the Scientific Reports (Nov 2024) sliding-window EMD paper.
- Guo-Sang-Tu-Wang's after-cost 2.16% daily figure (working-paper abstract only; JEDC full text not accessed).
- Findings of arXiv:2602.00776 (cross-asset OFI) and arXiv:2411.06327 (on-chain flows): located, abstracts only skimmed via search snippets.
- Kolm et al. horizon-decay claim (secondary summary only).
- Items tagged [V-kb] above were not re-fetched this session; page numbers and DOIs for them are from memory and should be spot-checked before external use.
- Oppenheim & Schafer reference for the DFT periodic-extension point; VanderPlas (2018) Lomb-Scargle paper; Smith & Wallis (2009); Wang-Hyndman-Li-Kang IJF volume/pages; Diebold-Mariano, Clark-West, White, Hansen page numbers.
