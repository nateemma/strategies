# 037 addendum: indicators vs a gbb-neutral triple-barrier label

**SCREEN (descriptive), added after the pre-registration lock (user request, 2026-10-17); rules are the locked ones, this addendum's choices are in `preregistration_tb.json`.** Nothing here is a P&L claim.

Label `label:tb_buy`: triple barrier on every bar, {'horizon': 48, 'pt': 0.02, 'sl': 0.05, 'min_gain': 0.02}. Comparators: the gbb trigger (`base:gbb_candidate`) and the width book's entry rule (`cmp:width_book`).

## Comparator lifts (median per half)

| population | comparator | E0 | early | late |
|---|---|---|---|---|
| all | `base:gbb_candidate` | 1.197 | 1.384 | 1.302 |
| all | `cmp:width_book` | 1.361 | 1.616 | 1.619 |
| candidate | `base:gbb_candidate` | – | – | – |
| candidate | `cmp:width_book` | 1.137 | 1.166 | 1.225 |

## Single columns (all bars): top 20 by median strength, indicator-tagged only

| column | median AUC | early | late | E0 | lift (signals) | E0 agrees |
|---|---|---|---|---|---|---|
| `base:atr_pct` | 0.625 | 0.658 | 0.606 | 0.631 | – | True |
| `Ehlers_LW:lw_atr/close` | 0.620 | 0.652 | 0.593 | 0.622 | – | True |
| `LWBreakout:level_pct_diff` | 0.619 | 0.652 | 0.594 | 0.624 | – | True |
| `LWBreakout:atr/close` | 0.619 | 0.652 | 0.594 | 0.624 | – | True |
| `LWRevert:atr/close` | 0.619 | 0.652 | 0.594 | 0.624 | – | True |
| `ZScore:rolling_std/close` | 0.619 | 0.640 | 0.609 | 0.602 | – | True |
| `base:bb_width` | 0.619 | 0.639 | 0.608 | 0.602 | – | True |
| `VBM:vbm` | 0.618 | 0.650 | 0.595 | 0.624 | – | True |
| `base:atr_pct_roll` | 0.616 | 0.647 | 0.602 | 0.629 | – | True |
| `Ehlers_DC:dc_upper/close` | 0.591 | 0.596 | 0.582 | 0.579 | – | True |
| `DCBreakout:dc_upper/close` | 0.590 | 0.595 | 0.580 | 0.579 | – | True |
| `DCRevert:dc_upper/close` | 0.590 | 0.595 | 0.580 | 0.579 | – | True |
| `Ehlers_LW:lw_cycle_sell/close` | 0.588 | 0.608 | 0.580 | 0.582 | – | True |
| `LWRevert:sell_level/close` | 0.588 | 0.608 | 0.579 | 0.583 | – | True |
| `base:bb_upperband/close` | 0.588 | 0.591 | 0.574 | 0.571 | – | True |
| `BBBreakout:bb_upperband/close` | 0.587 | 0.590 | 0.571 | 0.571 | – | True |
| `BBRevert:bb_upperband/close` | 0.587 | 0.590 | 0.571 | 0.571 | – | True |
| `Ehlers_KC:kc_upper/close` | 0.577 | 0.586 | 0.549 | 0.573 | – | True |
| `raw8:range` | 0.571 | 0.575 | 0.547 | 0.582 | – | True |
| `KCBreakout:kc_upper/close` | 0.566 | 0.570 | 0.534 | 0.563 | – | True |

## Signals (all bars): top 15 by median lift, beside the comparators

| signal | median lift (15 test windows) | fire rate | E0 agrees |
|---|---|---|---|
| `raw_sell:KST` | 1.71 | 0.003 | True |
| `raw_sell:TRIX` | 1.68 | 0.006 | True |
| `cmp:width_book` | 1.62 | 0.010 | True |
| `raw_sell:COPP` | 1.61 | 0.006 | True |
| `raw_sell:EWO` | 1.59 | 0.003 | True |
| `raw_buy:COPP` | 1.50 | 0.030 | True |
| `raw_buy:KST` | 1.50 | 0.038 | True |
| `raw_buy:WOBV` | 1.49 | 0.000 | True |
| `raw_buy:ROC` | 1.44 | 0.015 | True |
| `raw_buy:TRIX` | 1.40 | 0.069 | True |
| `raw_sell:WOBV` | 1.40 | 0.000 | False |
| `base:gbb_candidate` | 1.35 | 0.159 | True |
| `raw_buy:PPO` | 1.33 | 0.080 | True |
| `raw_sell:PPO` | 1.30 | 0.086 | True |
| `raw_buy:EWO` | 1.30 | 0.136 | True |

## Signal pairs (both must fire)

- all bars: 69 pass the rule, 20 survivors, 19 confirmed (E0 agrees).
  - `raw_buy:KST` AND `raw_buy:ZScore`: early 1.94, late 1.88, E0 1.44
  - `raw_buy:COPP` AND `raw_buy:MI`: early 1.89, late 1.54, E0 1.42
  - `raw_buy:CCI` AND `raw_buy:KST`: early 1.85, late 1.73, E0 1.41
  - `raw_buy:KST` AND `raw_buy:LWRevert`: early 1.82, late 1.57, E0 1.38
  - `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST`: early 1.80, late 1.58, E0 1.35
  - `raw_buy:CCI` AND `raw_buy:COPP`: early 1.80, late 1.89, E0 1.46
  - `raw_buy:CCI` AND `raw_buy:TRIX`: early 1.79, late 1.69, E0 1.35
  - `raw_buy:KST` AND `raw_buy:WTO`: early 1.78, late 1.54, E0 1.32
  - `raw_buy:ROC` AND `raw_buy:STC`: early 1.77, late 1.49, E0 1.32
  - `raw_buy:MI` AND `raw_buy:ROC`: early 1.77, late 1.50, E0 1.37
- candidate bars: 53 pass the rule, 20 survivors, 13 confirmed (E0 agrees).
  - `raw_buy:WTO` AND `raw_buy:ZScore`: early 1.45, late 1.28, E0 1.11
  - `raw_buy:CCI` AND `raw_buy:COPP`: early 1.44, late 1.48, E0 1.22
  - `raw_buy:KST` AND `raw_buy:ZScore`: early 1.42, late 1.47, E0 1.23
  - `raw_buy:CCI` AND `raw_buy:WTO`: early 1.42, late 1.25, E0 1.11
  - `raw_buy:COPP` AND `raw_buy:WTO`: early 1.41, late 1.29, E0 1.18
  - `raw_buy:CCI` AND `raw_buy:KST`: early 1.41, late 1.33, E0 1.23
  - `raw_buy:HTSINE` AND `raw_buy:ZScore`: early 1.38, late 1.36, E0 1.07
  - `raw_buy:KST` AND `raw_buy:LWRevert`: early 1.36, late 1.24, E0 1.17
  - `raw_buy:SMACross` AND `raw_buy:ZScore`: early 1.35, late 1.37, E0 1.11
  - `raw_buy:STC` AND `raw_buy:ZScore`: early 1.34, late 1.21, E0 1.07

## Beyond ARO: AUC within ARO p_buy quintiles (test windows; descriptive)

- ARO p_buy AUC on label:buy [candidate]: early 0.603, late 0.599
- ARO p_buy AUC on label:tb_buy [all]: early 0.598, late 0.577
- ARO p_buy AUC on label:tb_buy [candidate]: early 0.593, late 0.591

**label:tb_buy [all]:** 42 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `base:atr_pct` | 0.611 | 0.582 | 0.658 | 0.606 |
| `base:atr_pct_roll` | 0.603 | 0.576 | 0.647 | 0.602 |
| `LWBreakout:level_pct_diff` | 0.600 | 0.574 | 0.652 | 0.594 |
| `LWBreakout:atr/close` | 0.600 | 0.574 | 0.652 | 0.594 |
| `LWRevert:atr/close` | 0.600 | 0.574 | 0.652 | 0.594 |
| `VBM:vbm` | 0.600 | 0.574 | 0.650 | 0.595 |
| `Ehlers_LW:lw_atr/close` | 0.599 | 0.573 | 0.652 | 0.593 |
| `ZScore:rolling_std/close` | 0.578 | 0.563 | 0.640 | 0.609 |
| `base:bb_width` | 0.577 | 0.563 | 0.639 | 0.608 |
| `LWRevert:sell_level/close` | 0.569 | 0.553 | 0.608 | 0.579 |
| `Ehlers_LW:lw_cycle_sell/close` | 0.568 | 0.552 | 0.608 | 0.580 |
| `Ehlers_KC:kc_lower/close` | 0.453 | 0.457 | 0.453 | 0.461 |

**label:tb_buy [candidate]:** 18 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `base:atr_pct` | 0.578 | 0.567 | 0.612 | 0.599 |
| `base:atr_pct_roll` | 0.566 | 0.570 | 0.607 | 0.595 |
| `LWBreakout:atr/close` | 0.562 | 0.545 | 0.597 | 0.583 |
| `LWBreakout:level_pct_diff` | 0.562 | 0.545 | 0.597 | 0.583 |
| `LWRevert:atr/close` | 0.562 | 0.545 | 0.597 | 0.583 |
| `ni:ni_ms_vr` | 0.547 | 0.544 | 0.567 | 0.556 |
| `VBM:vbm` | 0.562 | 0.543 | 0.595 | 0.582 |
| `Ehlers_LW:lw_atr/close` | 0.559 | 0.542 | 0.591 | 0.579 |
| `LWRevert:sell_level/close` | 0.553 | 0.541 | 0.580 | 0.576 |
| `Ehlers_LW:lw_cycle_sell/close` | 0.552 | 0.539 | 0.578 | 0.577 |
| `LWBreakout:buy_level/close` | 0.539 | 0.537 | 0.563 | 0.567 |
| `Ehlers_LW:lw_trend_buy/close` | 0.539 | 0.536 | 0.562 | 0.563 |

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
