# 037 addendum: indicators vs a gbb-neutral triple-barrier label

**SCREEN (descriptive), added after the pre-registration lock (user request, 2026-10-17); rules are the locked ones, this addendum's choices are in `preregistration_tb.json`.** Nothing here is a P&L claim.

Label `label:tbatr_buy`: triple barrier on every bar, {'horizon': 48, 'pt': 1e-09, 'sl': 1e-09, 'atr_scale': 4.0} (symmetric ATR barriers; timeouts and ties are NaN, so the label is the direction of resolution). Comparators: the gbb trigger (`base:gbb_candidate`) and the width book's entry rule (`cmp:width_book`).

## Comparator lifts (median per half)

| population | comparator | E0 | early | late |
|---|---|---|---|---|
| all | `base:gbb_candidate` | 1.017 | 1.088 | 1.007 |
| all | `cmp:width_book` | 0.955 | 1.091 | 1.105 |
| candidate | `base:gbb_candidate` | – | – | – |
| candidate | `cmp:width_book` | 0.939 | 1.020 | 1.073 |

## Single columns (all bars): top 20 by median strength, indicator-tagged only

| column | median AUC | early | late | E0 | lift (signals) | E0 agrees |
|---|---|---|---|---|---|---|
| `TEMACross:tema_long/close` | 0.535 | 0.534 | 0.535 | 0.518 | – | True |
| `Ehlers_DC:dc_upper/close` | 0.529 | 0.533 | 0.523 | 0.518 | – | True |
| `EMACross:ema_short/close` | 0.529 | 0.529 | 0.525 | 0.516 | – | True |
| `EhlersBase:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_BB:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_DC:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_DYMI:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_Fisher:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_Ichimoku:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_Indicator:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_KC:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_LW:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_ROC:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `Ehlers_ind_DYMI:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `HTDCP:ss/close` | 0.529 | 0.529 | 0.528 | 0.517 | – | True |
| `EhlersBase:dsp/close` | 0.471 | 0.471 | 0.472 | 0.483 | – | True |
| `Ehlers_BB:dsp/close` | 0.471 | 0.471 | 0.472 | 0.483 | – | True |
| `Ehlers_DC:dsp/close` | 0.471 | 0.471 | 0.472 | 0.483 | – | True |
| `Ehlers_DYMI:dsp/close` | 0.471 | 0.471 | 0.472 | 0.483 | – | True |
| `Ehlers_Fisher:dsp/close` | 0.471 | 0.471 | 0.472 | 0.483 | – | True |

## Signals (all bars): top 15 by median lift, beside the comparators

| signal | median lift (15 test windows) | fire rate | E0 agrees |
|---|---|---|---|
| `raw_sell:TRIX` | 1.32 | 0.003 | True |
| `raw_sell:KST` | 1.29 | 0.002 | True |
| `raw_sell:EWO` | 1.27 | 0.002 | True |
| `raw_sell:COPP` | 1.26 | 0.004 | True |
| `raw_buy:WOBV` | 1.19 | 0.000 | False |
| `raw_buy:CMO` | 1.13 | 0.037 | True |
| `raw_buy:ZScore` | 1.12 | 0.079 | True |
| `raw_sell:DCBreakout` | 1.12 | 0.079 | True |
| `raw_buy:CCI` | 1.11 | 0.044 | True |
| `raw_sell:WOBV` | 1.11 | 0.000 | False |
| `raw_buy:UO` | 1.11 | 0.055 | True |
| `cmp:width_book` | 1.11 | 0.005 | False |
| `raw_sell:ROC` | 1.10 | 0.006 | True |
| `raw_buy:LWRevert` | 1.10 | 0.106 | True |
| `raw_sell:Ehlers_Ichimoku:trend` | 1.10 | 0.072 | True |
| `base:gbb_candidate` | 1.05 | 0.139 | True |

## Signal pairs (both must fire)

- all bars: 25 pass the rule, 20 survivors, 4 confirmed (E0 agrees).
  - `raw_buy:KST` AND `raw_buy:ZScore`: early 1.46, late 1.21, E0 1.09
  - `raw_buy:CMF` AND `raw_buy:CMO`: early 1.40, late 1.24, E0 1.22
  - `raw_buy:CCI` AND `raw_buy:KST`: early 1.39, late 1.20, E0 1.18
  - `raw_buy:TRIX` AND `raw_buy:ZScore`: early 1.28, late 1.20, E0 1.09
- candidate bars: 17 pass the rule, 17 survivors, 3 confirmed (E0 agrees).
  - `raw_buy:HTSINE` AND `raw_buy:ZScore`: early 1.28, late 1.23, E0 1.03
  - `raw_buy:SMACross` AND `raw_buy:WTO`: early 1.21, late 1.25, E0 1.15
  - `raw_buy:EMACross` AND `raw_buy:VFI`: early 1.20, late 1.29, E0 1.19

## Beyond ARO: AUC within ARO p_buy quintiles (test windows; descriptive)

- ARO p_buy AUC on label:buy [candidate]: early 0.603, late 0.599
- ARO p_buy AUC on label:tbatr_buy [all]: early 0.518, late 0.497
- ARO p_buy AUC on label:tbatr_buy [candidate]: early 0.541, late 0.539

**label:tbatr_buy [all]:** 57 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `TEMACross:tema_long/close` | 0.526 | 0.535 | 0.534 | 0.535 |
| `base:fast_diff` | 0.526 | 0.532 | 0.525 | 0.532 |
| `raw8:ret1` | 0.474 | 0.468 | 0.473 | 0.469 |
| `base:ss_close/close` | 0.526 | 0.527 | 0.527 | 0.528 |
| `EhlersBase:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_BB:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_DC:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_DYMI:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_Fisher:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_Ichimoku:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_Indicator:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |
| `Ehlers_KC:ss/close` | 0.525 | 0.530 | 0.529 | 0.528 |

**label:tbatr_buy [candidate]:** 148 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `EhlersBase:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_BB:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_DC:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_DYMI:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_Fisher:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_Ichimoku:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_Indicator:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_KC:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_LW:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_ROC:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `Ehlers_ind_DYMI:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |
| `HTDCP:ss/close` | 0.537 | 0.548 | 0.549 | 0.549 |

**label:buy [candidate]:** 74 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `base:atr_pct_roll` | 0.553 | 0.551 | 0.598 | 0.588 |
| `base:atr_pct` | 0.565 | 0.549 | 0.604 | 0.592 |
| `Ehlers_KC:kc_upper/close` | 0.560 | 0.541 | 0.599 | 0.595 |
| `LWRevert:sell_level/close` | 0.553 | 0.541 | 0.583 | 0.576 |
| `Ehlers_LW:lw_cycle_sell/close` | 0.552 | 0.539 | 0.581 | 0.577 |
| `ni:ni_ms_vr` | 0.540 | 0.539 | 0.564 | 0.554 |
| `LWBreakout:buy_level/close` | 0.542 | 0.536 | 0.569 | 0.569 |
| `Ehlers_LW:lw_trend_buy/close` | 0.541 | 0.533 | 0.568 | 0.565 |
| `raw8:log_hc` | 0.533 | 0.533 | 0.549 | 0.550 |
| `KCBreakout:kc_upper/close` | 0.551 | 0.532 | 0.590 | 0.592 |
| `KCRevert:kc_upper/close` | 0.551 | 0.532 | 0.590 | 0.592 |
| `Ehlers_LW:lw_atr/close` | 0.550 | 0.530 | 0.582 | 0.572 |
