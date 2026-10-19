# Time Series Prediction (TSPredict)

Strategies that try to forecast where price will be a few candles ahead, and trade on the forecast.
For setup, data download, backtesting, plotting, hyperopt, dry runs and live trading, see the
[main README](../README.md); everything here is specific to this directory.

> **Status (2026-10-19): none of these approaches works after costs.** A full study (feature 038,
> summarised below) tested every forecasting method in this directory, and several more, against
> realistic costs. Nothing passed, which is what the published research predicts. Treat the strategies
> here as reference and research code, not as candidates for live trading.

## What is here

All strategies predict **gain**: the percentage change in close over the next `lookahead` candles. Gain
is used instead of price because it is comparable across pairs, so one model can serve every pair.

| Files | Approach |
|---|---|
| `TSPredict.py` | Base class: dataframe handling, model load/save/train, walk-forward predictions, entry/exit and custom stoploss |
| `TS_Gain_*.py` | Predict from the gain history alone. Baselines, and a check for lookahead bias |
| `TS_Coeff_*.py` | Add signal-decomposition coefficients (FFT, DWT, SWT, MODWT, WPT) to the indicators and train a regressor on them |
| `TS_Wavelet*.py`, `TS_Wavelet.py` | Decompose the gain with a transform, forecast each component, rebuild the series. These do not save models |
| `BuyRegions.py` | A simple strategy that marks buy/sell regions, used with hyperopt to tune the guard indicators the TSPredict family uses |
| `TS_ForecastPlot.py` | **Plot-only** view of the 038 trend forecasts (damped Holt on 32/64/128/256 candles, 12 h ahead) and the vote-C signal. Forecasts every 16 candles and caches them, so long ranges plot quickly |

The forecasters these strategies use live in `utils/Forecasters.py`. The trend vote also exists as
framework triggers, which run with plain `freqtrade backtesting`:

```
.venv/bin/freqtrade backtesting --strategy SimpleTrigger_HoltVoteUp \
    --strategy-path user_data/strategies/SimpleTrigger -c user_data/strategies/config/config_wf.json \
    --timerange 20250101-20250401
```

Use `SimpleTrigger_HoltVoteFade` for the long-only fade, and see `research/holt_trigger/report.md`.

To plot the forecasts:

```
zsh user_data/strategies/scripts/plot_strat.sh -c user_data/strategies/config/config_wf.json \
    -n 14 TSPredict TS_ForecastPlot SOL/USDT
```

### Models and custom indicators

- Strategies other than `TS_Wavelet*` load a saved model if one exists, and otherwise train one. Run them
  over a long period the first time.
- If you change the indicators or the regression algorithm, delete the strategy's saved model so it is
  regenerated.
- To add indicators, subclass a strategy and override `add_strategy_indicators()`.

### Library changes on 2026-10-19 (affect every result from before that date)

The 038 study found and fixed defects in `utils/Forecasters.py`. Known-answer tests now guard each one
(`utils/tests/test_forecaster_horizons.py`).

| | Defect | Now |
|---|---|---|
| D1 | Exponential, StateSpace, ETS, ARIMA and SimpleExp returned the forecast one candle **too far** ahead | Exact horizon |
| D2 | SimpleExpSmoothing crashed on statsmodels 0.15 | Fixed |
| D3 | ARIMA returned **zeros** when its fit failed (looks like a genuine "no move" forecast) | Raises `ForecastFailed` |
| D4 | Regressors added the **training** targets' trend to forecasts for a different window | Uses the forecast window's own trend |
| D5 | `FFT_EXTRAPOLATION` (predict type 4) fed FFT coefficients to a regressor trained on returns | Raises `ForecastUnavailable` |
| D6 | `TS_Wavelet.predict` / `rolling_predict` called a helper with the wrong arguments (dead code) | Raise `NotImplementedError` |
| D8 | ARIMA and Theta built their model once and **re-fitted the first window forever** | Fresh model per window |
| D9 | The base `retrend` adds past window positions' trend to forecasts instead of extending it | **Not fixed** (TSPredict defaults use it) |
| D10 | The base `smooth` is centred and zero-padded, so the window's last value collapses (4.663 → 3.496 on log close) | **Not fixed** (TSPredict defaults use it) |

D9 and D10 still affect any forecaster run with `set_detrend(True)` or `set_smooth(True)`, which are
TSPredict's defaults. `harmonic_forecaster` (new) does its own correct detrending and smoothing.

**Why FFT extrapolation cannot forecast:** the inverse FFT is periodic in the window length, so the
"forecast" h candles ahead is the window's own value h−1 candles after its start. Raising or lowering the
frequency cutoff only smooths what gets repeated. `harmonic_forecaster` (frequencies fitted off-grid by
least squares) extends sinusoids properly, but 15m crypto prices contain no stable cycles to extend.

## What the 038 study found

Full write-up: `research/forecast/closing.md`. Tables: `research/forecast/report.md`. Spec:
`specs/038-close-forecast-study/`.

**Question:** can any method forecast the return of close N candles ahead (N = 15 minutes to 4 days)
well enough to trade **long, after costs**? Costs were a 0.10% fee plus 5 bp slippage per side, a 30 bp
round trip.

**How it was judged:**
- Long-only trades on the strongest forecasts, entered at the next open, compared with random entry at
  the same cost.
- The result had to hold in both 2023–24 and 2025–26.
- The rule was committed before any model result, and a shuffled-target control guarded the learned
  models.
- The same rule covered every later addition, each registered before its own results.

**Methods tested (about 280 method × horizon combinations):**
- baselines;
- AR, Holt, SES, ETS, ARIMA and Theta;
- harmonic regression, with and without smoothing and detrending;
- Ridge and LightGBM on lagged returns, on the NNNC indicators, on SWT/FFT components, on causal
  à trous wavelets, and on cross-asset (BTC and market) returns;
- N-of-M forecast votes;
- the Holt vote as momentum and fade triggers, backtested in the full framework.

**Findings:**
1. **No method or horizon is tradeable.** There is no PASS anywhere. The few marginal cells (at most
   +10 bp net per trade) are what chance produces across this many tests, and the best one lost money in
   the most recent six weeks.
2. **Costs are not the obstacle; the forecasts are.** Typical moves exceed the 30 bp cost from 30
   minutes up (8× at 1 day). The forecasts just don't pick out the big ones.
3. **Predictable is not profitable.** The best fits are at 15 minutes (R² up to 3.8% with cross-asset
   inputs), where moves are smaller than the cost. Where moves are big enough, skill is close to zero.
4. **The only real effect is trend timing,** worth 10–35 bp per trade over random entry at 4–24 h. It's
   too small to pay its costs and changes sign with the market regime: it worked in 2023–24, and buying
   after rallies lost in late 2026. As a framework trigger it lost to the existing momentum and reversion
   triggers.
5. **More model does not help.** LightGBM was no better than Ridge, and Ridge no better than a damped
   Holt. Wavelet and FFT inputs added nothing once boundary leakage was removed. The limit is the
   information in the inputs, not the method.
6. **Cycles do not extend.** Harmonic models overshoot, and a vote across cutoffs is one forecast in
   disguise: its members correlate at 0.97.

**This is what the literature predicts.** Return forecasts reach out-of-sample R² below about 1% even
with rich inputs (Welch & Goyal; Campbell & Thompson; Gu, Kelly & Xiu). In the M6 competition forecast
accuracy and investment results were barely related. Crypto momentum is real but short-lived and only
pays at longer horizons. Wavelet and EMD forecasting gains reported in many papers come from decomposing
the whole series before the train/test split (Hasumi & Kajita). The intraday crypto edges that survive
costs in published work come from other information: other coins' returns at minute horizons, and order
flow.

**What could still change the answer:**
- **New information.** Order flow, which the live NNNC recorder (feature 036) is collecting.
- **Using the trend or relative-weakness state inside an existing book** (NNNC/NNMT) as a filter or
  sizer, where it does not have to pay its own costs.
- **Lower measured execution costs.**

## References

Verification: **[web]** = checked against a publisher, repository or arXiv record. **[kb]** = details
from standard knowledge, not re-fetched; spot-check before citing externally. The full annotated review,
with links and per-paper notes on what each implies for this work, is `research/forecast/literature.md`.

**Return predictability and evaluation**
- Welch, I., Goyal, A. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. *Review of Financial Studies* 21(4), 1455–1508. [kb]
- Campbell, J.Y., Thompson, S.B. (2008). Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average? *Review of Financial Studies* 21(4), 1509–1531. [kb]
- Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies* 33(5), 2223–2273. [kb]
- Makridakis, S. et al. (2025). The M6 forecasting competition: Bridging the gap between forecasting and investment decisions. *International Journal of Forecasting* 41(4), 1315–1354; arXiv:2310.13357. [web]
- Harvey, C.R., Liu, Y., Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies* 29(1), 5–68. [web: NBER WP 20592]
- Harvey, C.R., Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13–28. [web]
- Bailey, D.H., López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94–107. [kb]
- Bailey, D.H., Borwein, J.M., López de Prado, M., Zhu, Q.J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the AMS* 61(5), 458–471. [web]
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. [kb]

**Crypto: intraday predictability, momentum, order flow**
- Guo, L., Sang, B., Tu, J., Wang, Y. (2024). Cross-cryptocurrency return predictability. *Journal of Economic Dynamics and Control* 163, 104863. [web]
- Fischer, T., Krauss, C., Deinert, A. (2019). Statistical Arbitrage in Cryptocurrency Markets. *Journal of Risk and Financial Management* 12(1), 31. [web]
- Hudson, R., Urquhart, A. (2021). Technical trading and cryptocurrencies. *Annals of Operations Research* 297, 191–220. [web]
- Shen, D., Urquhart, A., Wang, P. (2022). Bitcoin intraday time series momentum. *Financial Review* 57(2), 319–344. [web]
- Borgards, O. (2021). Momentum in cryptocurrencies (interday and intraday). *North American Journal of Economics and Finance* 57. [web record; exact title unverified]
- Liu, Y., Tsyvinski, A. (2021). Risks and Returns of Cryptocurrency. *Review of Financial Studies* 34(6), 2689–2727. [kb]
- Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). Time series momentum. *Journal of Financial Economics* 104(2), 228–250. [kb]
- Cont, R., Kukanov, A., Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1), 47–88. [kb]
- Kolm, P.N., Turiel, J., Westray, N. (2023). Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book. *Mathematical Finance* 33(4), 1044–1081. [web]

**Wavelets, decomposition and spectral methods**
- Renaud, O., Starck, J.-L., Murtagh, F. (2005). Wavelet-based combined signal filtering and prediction. *IEEE Trans. Systems, Man, and Cybernetics, Part B* 35(6), 1241–1251. [web]
- Hasumi, R., Kajita, Y. (2018). Boundary problem and data leakage: A caveat for wavelet-based forecasting. JCER Discussion Paper 148. [web]
- Percival, D.B., Walden, A.T. (2000). *Wavelet Methods for Time Series Analysis.* Cambridge University Press. [kb]
- Gençay, R., Selçuk, F., Whitcher, B. (2002). *An Introduction to Wavelets and Other Filtering Methods in Finance and Economics.* Academic Press. [kb]
- Ehlers, J.F. (2001). *Rocket Science for Traders.* Wiley; (2013) *Cycle Analytics for Traders.* Wiley. Practitioner books. [kb]
- *Bias effect on predicting market trends with EMD.* Expert Systems with Applications (2017), DOI 10.1016/j.eswa.2017.03.053. [web record; authors unverified]

**Forecasting methods and combination**
- Hyndman, R.J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts, https://otexts.com/fpp3/. [kb]
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*; arXiv:2205.13504. [kb]
- Bates, J.M., Granger, C.W.J. (1969). The Combination of Forecasts. *Operational Research Quarterly* 20(4), 451–468. [kb]
- Timmermann, A. (2006). Forecast Combinations. In *Handbook of Economic Forecasting*, vol. 1, ch. 4, 135–196. Elsevier. [kb]
- Claeskens, G., Magnus, J.R., Vasnev, A.L., Wang, W. (2016). The forecast combination puzzle: A simple theoretical explanation. *International Journal of Forecasting* 32(3), 754–762. [web]
- Wang, X., Hyndman, R.J., Li, F., Kang, Y. (2023). Forecast combinations: An over 50-year review. *International Journal of Forecasting* 39(4), 1518–1547; arXiv:2205.04216. [web: arXiv]
