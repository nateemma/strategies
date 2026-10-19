# 038 Close-Price Forecast Study: Closing Note (2026-10-19)

**Verdict: no method forecasts close well enough to trade at any horizon.** That covers 13 methods
(2 baselines, 7 distinct statistical models, 3 LightGBM sets plus a null) at 8 horizons from 15
minutes to 1 day, under the rule committed before any model result (`preregistration.json`, 20360e9c).
The score is 104 FAIL and 8 INCONCLUSIVE (the zero forecast, which never trades). There is no
most-viable N and no break-even N at 5 bp slippage, and the null control passes nowhere. The instrument
passed every check: known-answer tests, zero skill for the zero forecast, null behaving as noise, full
coverage, liveness and truncation.

## What the numbers say

1. **The cost line is not the binding limit; the edge is.** Median moves exceed the 30 bp round trip
   from N=2 upward (8× at N=96). The forecasts simply don't select moves large enough.
2. **Trend-following shows real timing skill, but it is smaller than the cost.**
   - Holt, AR, Theta and ARIMA beat random entry by about 10–25 bp per trade at N ≥ 16, in both halves.
     For example, AR at N=32 is +18/+23 bp, and Holt at N=16 is +12/+18 bp.
   - Their gross return per trade is about 20–45 bp, which is below 30 bp in the late half every time.
   - At fees only (20 bp, zero slippage) the best cells net +2 to +3 bp late. That is marginal, and no
     real fill gets zero slippage.
3. **Predictability is not edge (Principle III).**
   - The learned models' best out-of-sample R² is at N=1: +0.7% (L) and +1.2% (LA, with the ARO
     inputs).
   - Their edge over random entry at N=1 is negative (−1 to −2 bp), because 1-candle moves (25 bp
     median) are below cost.
   - At longer N, R² turns negative while edge over random entry turns positive.
4. **More model does not help.** LightGBM on lagged returns plus the 24 ARO inputs, or plus
   wavelet/FFT components, does no better than a damped Holt on 64 log closes. The SWT and FFT inputs
   change the paired results by a few bp either way. The information ceiling is in the inputs, not the
   method (global lesson: an unlearnable target is a data problem).
5. **Late is harder than early.** Several cells net +10 to +27 bp in the early half (2023–24), then
   −7 to −44 bp in the late half (2025–26), the same early/late drop seen in the NNNC studies.

## What was fixed along the way (kept)

The forecaster library (`utils/Forecasters.py`) had seven defects that invalidated any earlier
TSPredict comparison:
- an off-by-one horizon in 5 wrappers;
- a SimpleExpSmoothing crash;
- ARIMA returning zeros on failure;
- regressors restoring the training targets' trend;
- a meaningless FFT_EXTRAPOLATION mode;
- ARIMA and Theta re-fitting their first window forever;
- dead TS_Wavelet methods.

Known-answer tests now guard all of these (`utils/tests/test_forecaster_horizons.py`). Note that any
TSPredict backtest from before 2026-10-19 ran on the defective library.

## Where this points (options, not decisions)

- **The trend signal as an entry filter or sizer for NNNC, not a standalone strategy.**
  - Holt and AR timing adds 10–25 bp over random entry at 4–24 h. On a book that already pays its
    costs, that could be worth more than it is alone.
  - This is the follow-up the spec names (FR-014).
  - First check its correlation with what the momentum basket and NNNC's width sort already capture
    (Principle IX).
- **New inputs, if standalone forecasting stays the goal.** Funding, order flow and cross-asset data are
  the inputs that could move the ceiling (035 tested price-derived inputs only).
- **Costs.** The trend cells clear costs at fees alone. A maker-fee or lower-fee venue, or the 036 cost
  model's real slippage measurements (live gate pending), would change the arithmetic before the signal
  does.

## Post-hoc follow-ups (2026-10-19, each registered and committed before its results)

All were scored under the same rule, and all are labelled post-hoc: 15 more methods, 120 more cells.
The conclusion doesn't change. No PASS anywhere, and the null control passes nowhere.

| Addition | Amendment | Result |
|---|---|---|
| Harmonic regression (off-grid sinusoids), plain / smoothed / detrended / both | `preregistration_harmonic.json` | All FAIL. The plain sinusoids are worse than random entry in the late half (R² −1.5 to −6: the extended cycles overshoot). Only detrending helps, and the gain is the extended straight line, i.e. Holt's trend, not the cycles. Smoothing adds nothing. |
| Ridge (`ridge_L`), the linear benchmark | `preregistration_ridge.json` | All FAIL, with edge over random entry of about 0 (−3 to +5 bp). It does no worse than LightGBM, so the learned models found nothing nonlinear. |
| N-of-M votes A/B/C | `preregistration_vote.json` | A (harmonic K = 1–5): members correlate 0.97 (N=16), so it acts as one forecast; FAIL. B (6 different methods): members correlate 0.07, so 4-of-6 rarely fires; FAIL. **C (Holt on 32/64/128/256 candles), N=48: net +10.1 / +7.5 bp, the only cell in the study positive in both halves**, edge +20 / +34 bp, 1,847 / 1,409 trades. But the late lower bound is −19 bp, so it is INCONCLUSIVE, and in the forward window (2026-09-07 → 10-18, 138 trades) it nets −42 bp. |
| Causal à trous (Haar) multiscale forecast: `ridge_AT` (the published method), `lgb_AT`, `lgb_LAT` | `preregistration_atrous.json` | All FAIL. `ridge_AT` has no edge (−5 to +1 bp). LAT − L is a few bp either way, the same as the periodic-boundary SWT inputs. Wavelets add nothing to price-only forecasting here once leakage is excluded. That is consistent with the literature (`literature.md`). |

**Library defects found while building these (recorded, base helpers unchanged):**
- D9: base `retrend` adds the trend of past window positions to forecasts instead of extending it. On a perfect line, the linear forecaster gives 4.6537 for a true 4.667.
- D10: base `smooth` is a centred, zero-padded box filter. The window's last value collapses (4.663 → 3.496 on log close).
- Every Forecasters model with `set_detrend(True)` or `set_smooth(True)` is affected, which means TSPredict's defaults. The harmonic forecaster implements correct versions itself.

**Multiple testing:** about 28 methods × 8 horizons, roughly 220 cells. A single cell positive in both halves with a lower bound below zero is about what chance produces. The deflated-Sharpe / t > 3 standard in `literature.md` would not accept it.

**What the follow-ups add to "where this points":**
- **The trend signal improves with a longer look-back.** Holt on 256 candles at N=32 nets −4 / −2 bp, with edge +12 / +27. This matches the literature: crypto momentum lives at longer horizons. A test at N ≥ 96 with look-backs of days is the natural next horizon question.
- **The literature's most actionable lead is cross-asset inputs** (other pairs' and BTC's lagged returns): new information from data already held.

## Cross-asset inputs and 2–4 day horizons (`preregistration_long_xasset.json`, post-hoc)

7 more methods, 59 cells: still no PASS, and the null control passes nowhere. Break-even N stays 48.

- **Cross-asset information is real, but it is short-horizon information.**
  - Adding BTC and other-pair market returns raises the out-of-sample R² at N=1 five-fold: +3.8% (LightGBM LX) against +0.7% (L).
  - The edge over random entry at N=1 turns positive (+2.5 / +1.6 bp, from −0.5 / −1.3).
  - The effect fades by N ≥ 16, and 2.5 bp is nowhere near a 30 bp round trip.
  - This matches the literature: cross-coin lead-lag lives at minutes. It is new information, but at a horizon our costs can't pay for.
  - Ridge on the same inputs: R² +2.1% at N=1, edge ≈ 0.
- **At 2–4 days, the early half is market drift, not skill.**
  - Early-half nets are large (+40 to +98 bp), but the shuffled-target control is also positive early (+5 to +35 bp) with negative edge over random entry. 2023–24 rewarded any long held for days.
  - Every method's late-half net at 2–4 days is negative except one: LightGBM L at N=288 (3 days), +74.9 / +1.6 bp net, edge +11 / +45 bp. That cell is INCONCLUSIVE (lower bound −76 bp; about 1,300–1,500 trades).
- **Longer Holt windows help only up to about 3 days of look-back.** Holt on 288 candles at N=48 nets +3.6 / +0.5 bp, edge +14 / +30 (INCONCLUSIVE). That's the same horizon and sign as vote C. On 7 or 14 days of look-back the edge disappears or turns negative.
- **Overall:** about 280 method × horizon cells, three marginal INCONCLUSIVE cells (vote C N=48, Holt-288 N=48, LightGBM L N=288), no PASS. Price-only and cross-asset forecasting of close does not clear costs on these pairs. The remaining live lead is the 12 h trend timing as a filter inside NNNC/NNMT (memory: project_todo_trend_filter_nnnc).

`TSPredict/TS_ForecastPlot.py` plots the Holt-window forecasts and the vote C signal for inspection (plot-only; see its docstring).
