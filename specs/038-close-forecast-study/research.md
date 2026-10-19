# Research: 038 Close-Price Forecast Study

Decisions made while planning, each with its reason and the alternatives rejected. Facts were
measured on 2026-10-19 unless stated otherwise.

## R1 — Target and trade definition

**Decision**:
- **Forecast target**: the log return of close over N candles, `ln(close[t+N] / close[t])`, known at
  the close of candle t.
- **Trade definition for the cost check**: enter at `open[t+1]` and exit at `open[t+1+N]`, long only,
  so the trade starts after the signal candle closes.

**Rationale**:
- **Long only**: the venue is binance.us spot, with no shorting, and NNNC is long only. A short
  forecast cannot be traded, so short-side figures are reported as description only.
- **Entry at the next open**: an entry at `close[t]` would fill at a price that is already gone when
  the signal is known.
- **Log returns**: they add across candles, so a multi-step statistical forecast of log close turns
  directly into a return.

**Alternatives considered**:
- **Percentage gain, as TSPredict does**: it is the same information, and on these magnitudes it
  differs from the log return only at second order. The log form is kept for additivity.
- **Close level**: rejected (spec Assumptions).
- **Close-to-close fills**: rejected as optimistic (Principle V).

## R2 — Data, pairs, windows

**Decision**:
- **Data**: binance.us 15m OHLCV from `user_data/data/binanceus/<PAIR>-15m.feather`, gap-filled the
  same way as `research/indicator_screen/harvest.load_ohlcv(filled=True)`.
- **Pairs**: the 10 pairs in `config/config_wf.json`: ZEC, XRP, SOL, LINK, NEAR, AAVE, AVAX, LTC, BCH
  and DOT.
- **Folds**: the 15 folds of `evaluation/folds/plan_cs_aro.json`.
  - Training is anchored at 2021-01-01, and each test window is 90 days, f00 = 2023-01-01→04-01 to
    f14 = 2026-06-14→09-07.
  - The halves are EARLY (f00–f07) and LATE (f08–f14), the same as `evaluation/halves.py`.
- **Forward window**: 2026-09-07 → 2026-10-18 (the data now runs to 2026-10-19).
  - It is scored as description only, for every tier that does not need the 037 cache, which ends on
    2026-09-07.
  - At 6 weeks it is too short for a verdict.

**Rationale**: the same pairs and windows as every NNNC study, so the results can be compared.

**Measured**:
- **Coverage**: AAVE starts in 2021-07, ZEC in 2022-01, and XRP has a gap from 2021-01 to 2023-07.
- **Zero-volume share of 15m candles**: ZEC 53%, AAVE 52%, SOL 3.4%, XRP 6.5%.

**Alternatives considered**:
- **A distant era from binance.com**: `user_data/data/binance/` is empty, so there is no distant era.
  Halves are the persistence test, and verdicts are PROVISIONAL under quality gate 2, as in every
  study since 023.

## R3 — Zero-volume candles

**Decision**:
- A candle is **eligible for a trade** only if both its entry candle `t+1` and its exit candle
  `t+1+N` have volume > 0.
- All descriptive metrics (skill, IC, hit rate) are reported twice: on all candles, and on traded
  candles only.
- The verdict uses traded candles only.

**Rationale**:
- On a gap-filled zero-volume candle the close repeats the previous one, so the target is exactly 0.
  A model can "predict" those zeros from the volume history, which looks like skill but cannot be
  traded.
- 037 found that 32% of `data_ok` bars are zero volume.

**Alternatives considered**: dropping ZEC and AAVE would be a pair-specific choice (Principle X), so it
is rejected.

## R4 — Forecaster defects (Story 1) and how each is resolved

Every defect gets a known-answer test in `utils/tests/test_forecaster_horizons.py` before any result
is read.

| # | Defect | Resolution |
|---|--------|-----------|
| D1 | Exponential, StateSpace, ETS and ARIMA call `predict(0, len+steps)`, so the last element is `steps+1` ahead | Use `forecast(steps)` and append exactly `steps` values |
| D2 | SimpleExponential crashes on statsmodels 0.15 (`initialization_method must be a string`, measured) | Pass `initialization_method="estimated"` |
| D3 | ARIMA catches every exception and returns zeros | Re-raise a typed `ForecastFailed`; callers count failures (FR-007, Story 3 AS2) |
| D4 | The regressors' `retrend_results` adds the training window's trend to a prediction for a different window | Retrend with the trend of the window being predicted (the forecast input), never with the stored training detrender. Ignored when `detrend=False`, which the harness always uses (R6) |
| D5 | FFT_EXTRAPOLATION `predict_type=4` passes FFT coefficients to a regressor trained on returns | Mark unavailable: `forecast()` raises `ForecastUnavailable` with the reason. The FFT idea is tested soundly as Tier 3 inputs |
| D6 | `TS_Wavelet.predict()` and `rolling_predict()` call `predict_data(start, end)`, but it takes `X` | Both raise `NotImplementedError` naming `add_rolling_predictions` as the live path. They are dead code, never called (measured with grep) |
| D7 | The AR forecaster runs `ar_select_order` per window, so its lag order changes from window to window | Kept, as it is a model property, not a bug. Documented |
| D8 | Found while implementing: ARIMA and Theta build their model only when `self.model is None`, so every later call re-fits the FIRST window and the forecast never changes. Theta also used the index frequency `"5m"` and seasonal decomposition | A fresh model per window; Theta on a plain array with `period=1`. Test: `test_model_is_refitted_on_each_window` |

**Side effect**: TSPredict and TS_Coeff strategies that use these forecasters will see their forecasts
move by one candle. That is a correction. These strategies are not deployed, and none of them is
re-backtested in this feature (FR-014).

## R5 — Statistical tier (Story 3): series, window, stride, runtime

**Decision**:
- **Input series**: the trailing 64 candles of log close. These are level models: Holt (damped), simple
  exponential smoothing, ETS, state-space exponential smoothing, ARIMA(2,1,2) and Theta (`period=1`).
- **AR**: fit on the trailing 64 one-candle log returns, with the forecast summed over N steps.
- **All horizons from one fit**: forecast 96 steps once and read N ∈ {1,…,96} from the path.
- **Scoring stride**: score every 16th candle.
- **Window**: 64 is the window TSPredict was built with, and it is the only window tested
  (Principle VIII: no window search).

**Measured fit cost per 64-candle window** (forecasting 96 steps):

| Model | Cost per fit |
|---|---|
| AR (with lag selection) | 2.1 ms |
| Holt | 4.1 ms |
| SES | 1.0 ms |
| ExpSmoothing | 2.7 ms |
| ETS | 1.7 ms |
| StateSpace | 8.1 ms |
| ARIMA(2,1,2) | 24.4 ms |
| Theta | 3.0 ms |
| **All 8 models** | **≈ 47 ms per candle** |

**Gaps**: a window is not forecast when its last candle is gap-filled or more than half of its 64 candles
are. It is counted as `skipped_gap` (spec Edge Cases).

**Runtime**: about 1.29 M test candles across 10 pairs and 15 folds.
- At every candle that is about 17 h.
- At stride 16 it is about 1.1 h on one core, or about 20 min with 4 worker processes (SC-006 asks for
  under 1 h).

**Why stride 16**: it also makes successive samples non-overlapping for N ≤ 16.

**Parallelism**: the 4 worker processes run small CPU-only statsmodels fits, each using a few MB. The
one-trainer rule (memory `feedback_one_trainer_at_a_time`) targets GPU/MLX trainers, which faulted when
run in parallel. The learned tier still runs as one sequential job.

**Alternatives considered**:
- **Every candle**: too slow.
- **A random subsample**: harder to reproduce.

## R6 — Detrending and smoothing in the harness

**Decision**: the harness calls every forecaster with `set_detrend(False)` and smoothing off.

**Rationale**:
- The FFT detrender zeroes all but 6 frequency bins of a circular FFT, which wraps the window's end
  into its start.
- Detrending is not part of the question, and Tier 3 tests frequency content properly.
- The default stays unchanged for the strategies.

## R7 — Learned tier (Stories 4 and 5)

**Decision**: LightGBM regression, with the hyperparameters fixed and pre-registered:

```text
n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
subsample=0.8, subsample_freq=1, colsample_bytree=0.8, seed=1, n_jobs=6
```

**Training**:
- One model per (fold, N, feature set), trained on all pairs pooled.
- Rows: every set uses the same candles. Each row needs eligibility, real data at t, finite
  features, and the 037 cache's `meta:data_ok`, so the comparisons between sets are paired.
- Anchored training from 2021-01-01 up to the fold's test start, minus a purge of N candles (FR-004,
  Principle XI).
- Training rows are taken at stride 4 to bound the cost.

**Feature sets**:

| Set | Columns |
|---|---|
| **L** (lagged returns) | log returns over 1, 2, 4, 8, 16, 32, 64 and 96 candles; realised volatility over 16 and 96 candles; log volume ratio over 16 / 96 candles. All trailing |
| **LA** | L plus the 24 ARO model inputs from the 037 cache (the `aro:` aliases → `base:` columns: adx_scaled, aroonosc_scaled, atr_norm, bb_position, bb_width, cci_scaled, cg_ss, di_diff_scaled, ema_fast_norm, fast_diff, fastk_scaled, fisher_ss, gain_norm, guard_metric_neg/pos, macd_neg/pos, macdhist_norm, mfi_scaled, rsi_scaled, sar_ratio, spread_ma, vwap_neg/pos). The cache already passed 037's truncation test |
| **LW** (Story 5) | L plus decomposition inputs on the trailing 64 one-candle log returns: SWT `bior3.9` level-2 coefficients at the last position (approximation, D2, D1), plus FFT magnitudes of bins 1–8. They pass the truncation test (FR-012) before use |

**Other checks**:
- **Null control**: set L trained on targets shuffled within each fold's training span (FR-011).
- **Second seed**: any learned PASS is rerun with seed 2 and must PASS again (quality gate 5).

**Runtime**: 15 folds × 8 N × 4 sets (L, LA, LW, null) = 480 fits. A fit takes about 20–40 s on 0.5–1.9 M
rows × ≤ 50 columns, so the tier takes 3–5 h. It runs as one sequential job (one trainer at a time).

**Alternatives considered**:
- **The full 037 harvest (hundreds of columns)**: rejected. 037 found these columns redundant with the
  ARO inputs, and including them multiplies the cost.
- **MLX neural models**: out of scope (spec).
- **Per-pair models**: rejected under Principle X.

## R8 — Selection rule and the cost check (the verdict metric)

**Decision**:
- **Signal**: at candle t a method signals LONG when its forecast is positive (`f > 0`) and at or above
  the causal 90th-percentile threshold. The positive condition matters because trades are long only:
  without it, a constant forecast would trade every candle, and a model that only ever forecasts falls
  would buy its least-bad candles. The threshold is the 0.9 quantile of the same pair, method and N's
  forecasts over the previous 30 days of scored candles (2,880 at stride 1, 180 at stride 16). There is
  no signal until 30 days of history exist.
- **Trade simulation, per pair**: one position at a time. A signal at t opens a trade at `open[t+1]`
  that closes at `open[t+1+N]`. Signals during an open trade are ignored, and the candle must be
  eligible (R3).
- **Net return**: gross log return − round-trip cost.
  - The cost is 2 × (0.10% fee + s), with s ∈ {0, 2.5, 5, 10} bps per side.
  - Fees: binance.us charges 0.10/0.10 (memory `reference_exchange_venues`). `config_wf.json` has no
    `fee` key, so the exchange default applies.
- **Benchmark (random entry)**: the unconditional mean N-candle return of all eligible candles in the
  same pair and half, minus the same cost.
- **Uncertainty**: a block bootstrap over calendar weeks (2,000 resamples) of the pooled trades in each
  half. The blocks are longer than any N (96 candles = 1 day), so overlap cannot inflate significance.

**Rationale**:
- A causal threshold avoids choosing the decile with the test window's own forecasts.
- Comparing with random entry separates timing skill from market drift, so a model that only learned
  "crypto went up" does not pass.

## R9 — Pass rule (pre-registered; written to `research/forecast/preregistration.json`)

**Slippage level**: 5 bps per side, so the round-trip cost is 30 bps.

**Per (method, N), pooled over pairs, traded candles only, in each half**:

| Verdict | Condition |
|---|---|
| **PASS (PROVISIONAL)** | All of: (a) at least 200 trades in each half; (b) mean net return per trade > 0 in each half; (c) mean gross return per trade − the random-entry mean gross > 0 in each half; (d) the bootstrap 95% lower bound of the mean net return > 0 in the LATE half (Principle I: the deployment regime); (e) for learned methods, the null control FAILs at the same N and the seed-2 rerun PASSes |
| **FAIL** | Mean net ≤ 0 in either half, or condition (c) fails in either half |
| **INCONCLUSIVE** | Anything else (e.g. too few trades, or the lower bound spans 0) |

**Exclusions**:
- A method with more than 5% failed windows at an N is excluded from the ranking. Its verdict is
  INCONCLUSIVE with the failure share.
- **Baselines**:
  - "No change" never signals, so it has 0 trades and is INCONCLUSIVE by quality gate 6. Its role is
    the skill check (SC-002).
  - The historical mean is scored through the same rule.

**Viable horizon**:
- **Most viable N**: the N of the PASS cell with the highest late-half net return per pair-year, which
  rewards both margin and trade frequency.
- **Break-even horizon** (FR-008): the smallest N at which any method has mean net > 0 in both halves at
  5 bps.

**Multiple testing**:
- About 10 methods × 8 N = 80 cells. The shuffled-target control goes through the same rule at every N.
- The report states how many null cells PASS, which gives the false-pass rate seen in this test.
- If any null cell PASSes, every PASS at that tier is downgraded to INCONCLUSIVE.

## R10 — Descriptive metrics (FR-007)

These are reported, but never decide a verdict (Principle III):

| Metric | Definition |
|---|---|
| Skill | `R²_oos = 1 − Σ(r − f)² / Σ r²` (against the zero forecast) |
| IC | Spearman rank correlation of f and r, per fold, then the median |
| Hit rate | sign agreement where f ≠ 0 |
| Short side | the mirror of the cost check on short signals |

All are given per pair, per half and pooled.

## R11 — Code location and reuse

- **New package**: `research/forecast/`. It sits beside the other studies. It uses freqtrade only for
  its history loader (gap-filled candles, R2); there are no strategies or backtests.
- **Reused**:
  - `utils/Forecasters.py` (after the fixes);
  - `utils/Wavelets.py` / pywt;
  - `research/indicator_screen/harvest.read_cache` and `load_ohlcv`;
  - the fold windows from `plan_cs_aro.json`;
  - `research/control_sizer/judge._git` for the commit-time guard.
- **Tests**: `utils/tests/` for the forecaster fixes, `research/forecast/tests/` for the harness.
- **Outputs**:
  - large prediction tables go to `user_data/data/forecast_038/` (git-ignored);
  - summaries and verdicts go to `research/forecast/*.json` and `report.md` (committed).

## R12 — Liveness (Principle IV)

- The harness asserts that forecasts at different N differ for every model except "no change" and the
  historical mean, whose horizon dependence is linear by construction.
- The null control must differ from L.
- Any pair of arms that comes out byte-identical is reported as a defect.
