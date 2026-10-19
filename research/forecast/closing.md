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
