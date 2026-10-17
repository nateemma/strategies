# 037 addendum: indicators vs a gbb-neutral triple-barrier label

**SCREEN (descriptive), added after the pre-registration lock (user request, 2026-10-17); rules are the locked ones, this addendum's choices are in `preregistration_tb.json`.** Nothing here is a P&L claim.

Label `label:tbatr_buy`: triple barrier on every bar, {'horizon': 48, 'pt': 1e-09, 'sl': 1e-09, 'atr_scale': 4.0} (symmetric ATR barriers; timeouts and ties are NaN, so the label is the direction of resolution). Comparators: the gbb trigger (`base:gbb_candidate`) and the width book's entry rule (`cmp:width_book`).

## Comparator lifts (median per half)

| population | comparator | E0 | early | late |
|---|---|---|---|---|
| all | `base:gbb_candidate` | 1.025 | 1.107 | 1.070 |
| all | `cmp:width_book` | 0.956 | 1.164 | 1.153 |
| candidate | `base:gbb_candidate` | – | – | – |
| candidate | `cmp:width_book` | 0.933 | 1.051 | 1.030 |

## Single columns (all bars): top 20 by median strength, indicator-tagged only

| column | median AUC | early | late | E0 | lift (signals) | E0 agrees |
|---|---|---|---|---|---|---|
| `WILLR:wobv` | 0.442 | 0.444 | 0.438 | 0.469 | – | True |
| `base:fisher_wr` | 0.443 | 0.444 | 0.438 | 0.471 | – | True |
| `base:fastk_scaled` | 0.446 | 0.447 | 0.436 | 0.467 | – | True |
| `CMO:cmo` | 0.446 | 0.447 | 0.442 | 0.474 | – | True |
| `Ehlers_DC:dc_upper/close` | 0.553 | 0.553 | 0.553 | 0.525 | – | True |
| `DCBreakout:dc_upper/close` | 0.552 | 0.552 | 0.552 | 0.525 | – | True |
| `DCRevert:dc_upper/close` | 0.552 | 0.552 | 0.552 | 0.525 | – | True |
| `Ehlers_DC:dc_mid/close` | 0.552 | 0.551 | 0.552 | 0.523 | – | True |
| `base:fastd_scaled` | 0.450 | 0.453 | 0.443 | 0.471 | – | True |
| `Ehlers_Ichimoku:tenkan_sen/close` | 0.550 | 0.549 | 0.556 | 0.526 | – | True |
| `Ichimoku:tenkan_sen/close` | 0.550 | 0.549 | 0.556 | 0.526 | – | True |
| `base:dymi` | 0.451 | 0.451 | 0.443 | 0.477 | – | True |
| `base:dymi_scaled` | 0.451 | 0.451 | 0.443 | 0.477 | – | True |
| `Ehlers_Ichimoku:kijun_sen/close` | 0.549 | 0.549 | 0.549 | 0.521 | – | True |
| `Ichimoku:kijun_sen/close` | 0.549 | 0.549 | 0.549 | 0.521 | – | True |
| `base:rsi` | 0.451 | 0.452 | 0.444 | 0.478 | – | True |
| `base:rsi_scaled` | 0.451 | 0.452 | 0.444 | 0.478 | – | True |
| `ZScore:rolling_mean/close` | 0.549 | 0.550 | 0.545 | 0.521 | – | True |
| `base:bb_middleband/close` | 0.549 | 0.550 | 0.545 | 0.521 | – | True |
| `base:bb_position` | 0.451 | 0.453 | 0.451 | 0.474 | – | True |

## Signals (all bars): top 15 by median lift, beside the comparators

| signal | median lift (15 test windows) | fire rate | E0 agrees |
|---|---|---|---|
| `raw_sell:EWO` | 1.30 | 0.001 | True |
| `raw_sell:KST` | 1.27 | 0.001 | True |
| `raw_sell:TRIX` | 1.25 | 0.003 | True |
| `raw_buy:CMO` | 1.22 | 0.060 | True |
| `raw_sell:DCBreakout` | 1.19 | 0.126 | True |
| `raw_buy:WOBV` | 1.19 | 0.000 | False |
| `raw_sell:COPP` | 1.18 | 0.003 | True |
| `raw_buy:TSI` | 1.17 | 0.092 | True |
| `cmp:width_book` | 1.16 | 0.004 | False |
| `raw_buy:UO` | 1.16 | 0.137 | True |
| `raw_buy:DCRevert` | 1.16 | 0.125 | True |
| `raw_buy:COPP` | 1.15 | 0.020 | False |
| `raw_buy:WILLR` | 1.14 | 0.253 | True |
| `raw_buy:ZScore` | 1.14 | 0.056 | True |
| `raw_buy:CCI` | 1.13 | 0.030 | True |
| `base:gbb_candidate` | 1.10 | 0.133 | True |

## Signal pairs (both must fire)

- all bars: 57 pass the rule, 20 survivors, 14 confirmed (E0 agrees).
  - `raw_buy:CMO` AND `raw_buy:DCBreakout`: early 1.71, late 1.46, E0 1.16
  - `raw_buy:DCBreakout` AND `raw_buy:TSI`: early 1.55, late 1.32, E0 1.18
  - `raw_buy:CMO` AND `raw_buy:PVO`: early 1.49, late 1.37, E0 1.21
  - `raw_buy:KST` AND `raw_buy:ZScore`: early 1.46, late 1.23, E0 1.10
  - `raw_buy:CMO` AND `raw_buy:TEMACross`: early 1.42, late 1.38, E0 1.23
  - `raw_buy:CCI` AND `raw_buy:KST`: early 1.38, late 1.23, E0 1.19
  - `raw_buy:CMO` AND `raw_buy:MACDhist`: early 1.36, late 1.32, E0 1.32
  - `raw_buy:CMO` AND `raw_buy:Ehlers_ROC:trend`: early 1.35, late 1.29, E0 1.24
  - `raw_buy:CMO` AND `raw_buy:Ehlers_Fisher:trend`: early 1.35, late 1.24, E0 1.33
  - `raw_buy:CCI` AND `raw_buy:TRIX`: early 1.34, late 1.22, E0 1.15
- candidate bars: 19 pass the rule, 19 survivors, 6 confirmed (E0 agrees).
  - `raw_buy:PVO` AND `raw_buy:TSI`: early 1.32, late 1.21, E0 1.13
  - `raw_buy:PVO` AND `raw_buy:VBM`: early 1.31, late 1.25, E0 1.11
  - `raw_buy:CMO` AND `raw_buy:STC`: early 1.26, late 1.23, E0 1.08
  - `raw_buy:PVO` AND `raw_buy:WILLR`: early 1.24, late 1.26, E0 1.07
  - `raw_buy:EWO` AND `raw_buy:PVO`: early 1.22, late 1.22, E0 1.06
  - `raw_buy:PVO` AND `raw_buy:TRIX`: early 1.21, late 1.22, E0 1.07

## Beyond ARO: AUC within ARO p_buy quintiles (test windows; descriptive)

- ARO p_buy AUC on label:buy [candidate]: early 0.607, late 0.597
- ARO p_buy AUC on label:tbatr_buy [all]: early 0.530, late 0.514
- ARO p_buy AUC on label:tbatr_buy [candidate]: early 0.554, late 0.540

**label:tbatr_buy [all]:** 221 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `WILLR:wobv` | 0.454 | 0.444 | 0.444 | 0.438 |
| `base:fisher_wr` | 0.455 | 0.443 | 0.444 | 0.438 |
| `base:fastk_scaled` | 0.456 | 0.443 | 0.447 | 0.436 |
| `Ehlers_DC:dc_upper/close` | 0.544 | 0.546 | 0.553 | 0.553 |
| `CMO:cmo` | 0.458 | 0.448 | 0.447 | 0.442 |
| `DCBreakout:dc_upper/close` | 0.542 | 0.543 | 0.552 | 0.552 |
| `DCRevert:dc_upper/close` | 0.542 | 0.543 | 0.552 | 0.552 |
| `Ehlers_Ichimoku:tenkan_sen/close` | 0.538 | 0.551 | 0.549 | 0.556 |
| `Ichimoku:tenkan_sen/close` | 0.538 | 0.551 | 0.549 | 0.556 |
| `base:fastd_scaled` | 0.462 | 0.450 | 0.453 | 0.443 |
| `Ehlers_DC:dc_mid/close` | 0.537 | 0.547 | 0.551 | 0.552 |
| `EMACross:ema_short/close` | 0.537 | 0.544 | 0.547 | 0.549 |

**label:tbatr_buy [candidate]:** 202 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `base:fastk_scaled` | 0.444 | 0.441 | 0.426 | 0.428 |
| `WILLR:wobv` | 0.445 | 0.444 | 0.440 | 0.433 |
| `Ehlers_DC:dc_lower/close` | 0.553 | 0.551 | 0.560 | 0.560 |
| `base:fisher_wr` | 0.449 | 0.442 | 0.437 | 0.437 |
| `EMACross:ema_short/close` | 0.550 | 0.552 | 0.565 | 0.551 |
| `DCBreakout:dc_lower/close` | 0.549 | 0.550 | 0.556 | 0.555 |
| `DCRevert:dc_lower/close` | 0.549 | 0.550 | 0.556 | 0.555 |
| `base:fastd_scaled` | 0.451 | 0.447 | 0.445 | 0.437 |
| `Ehlers_Ichimoku:tenkan_sen/close` | 0.548 | 0.563 | 0.570 | 0.560 |
| `Ichimoku:tenkan_sen/close` | 0.548 | 0.563 | 0.570 | 0.560 |
| `CMO:cmo` | 0.453 | 0.443 | 0.450 | 0.436 |
| `SAR:sar/close` | 0.546 | 0.563 | 0.561 | 0.573 |

**label:buy [candidate]:** 134 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `Ehlers_KC:kc_upper/close` | 0.555 | 0.544 | 0.597 | 0.582 |
| `KCBreakout:kc_upper/close` | 0.550 | 0.539 | 0.590 | 0.581 |
| `KCRevert:kc_upper/close` | 0.550 | 0.539 | 0.590 | 0.581 |
| `SAR:sar/close` | 0.538 | 0.547 | 0.574 | 0.583 |
| `PSAR:psar/close` | 0.536 | 0.545 | 0.572 | 0.584 |
| `base:atr_pct_roll` | 0.547 | 0.536 | 0.595 | 0.576 |
| `Ehlers_KC:kc_mid/close` | 0.543 | 0.535 | 0.575 | 0.559 |
| `base:ema_fast/close` | 0.543 | 0.535 | 0.575 | 0.559 |
| `base:fastk_scaled` | 0.465 | 0.462 | 0.441 | 0.441 |
| `Ehlers_Ichimoku:tenkan_sen/close` | 0.534 | 0.539 | 0.575 | 0.570 |
| `Ichimoku:tenkan_sen/close` | 0.534 | 0.539 | 0.575 | 0.570 |
| `ZScore:rolling_mean/close` | 0.541 | 0.534 | 0.570 | 0.552 |
