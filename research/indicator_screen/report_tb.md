# 037 addendum: indicators vs a gbb-neutral triple-barrier label

**SCREEN (descriptive), added after the pre-registration lock (user request, 2026-10-17); rules are the locked ones, this addendum's choices are in `preregistration_tb.json`.** Nothing here is a P&L claim.

Label `label:tb_buy`: triple barrier on every bar, {'horizon': 48, 'pt': 0.02, 'sl': 0.05, 'min_gain': 0.02}. Comparators: the gbb trigger (`base:gbb_candidate`) and the width book's entry rule (`cmp:width_book`).

## Comparator lifts (median per half)

| population | comparator | E0 | early | late |
|---|---|---|---|---|
| all | `base:gbb_candidate` | 1.244 | 1.422 | 1.300 |
| all | `cmp:width_book` | 1.412 | 1.688 | 1.673 |
| candidate | `base:gbb_candidate` | – | – | – |
| candidate | `cmp:width_book` | 1.135 | 1.155 | 1.196 |

## Single columns (all bars): top 20 by median strength, indicator-tagged only

| column | median AUC | early | late | E0 | lift (signals) | E0 agrees |
|---|---|---|---|---|---|---|
| `base:atr_pct` | 0.620 | 0.652 | 0.590 | 0.637 | – | True |
| `base:atr_pct_roll` | 0.614 | 0.645 | 0.585 | 0.636 | – | True |
| `VBM:vbm` | 0.607 | 0.642 | 0.577 | 0.632 | – | True |
| `LWBreakout:atr/close` | 0.606 | 0.642 | 0.578 | 0.631 | – | True |
| `LWRevert:atr/close` | 0.606 | 0.642 | 0.578 | 0.631 | – | True |
| `LWBreakout:level_pct_diff` | 0.606 | 0.642 | 0.578 | 0.631 | – | True |
| `Ehlers_LW:lw_atr/close` | 0.606 | 0.641 | 0.577 | 0.628 | – | True |
| `ZScore:rolling_std/close` | 0.602 | 0.638 | 0.596 | 0.612 | – | True |
| `base:bb_width` | 0.601 | 0.637 | 0.595 | 0.612 | – | True |
| `DCBreakout:dc_upper/close` | 0.592 | 0.605 | 0.580 | 0.591 | – | True |
| `DCRevert:dc_upper/close` | 0.592 | 0.605 | 0.580 | 0.591 | – | True |
| `Ehlers_DC:dc_upper/close` | 0.591 | 0.605 | 0.580 | 0.591 | – | True |
| `BBBreakout:bb_upperband/close` | 0.589 | 0.601 | 0.572 | 0.582 | – | True |
| `BBRevert:bb_upperband/close` | 0.589 | 0.601 | 0.572 | 0.582 | – | True |
| `base:bb_upperband/close` | 0.588 | 0.600 | 0.574 | 0.582 | – | True |
| `Ehlers_KC:kc_upper/close` | 0.584 | 0.597 | 0.550 | 0.584 | – | True |
| `LWRevert:sell_level/close` | 0.584 | 0.606 | 0.567 | 0.595 | – | True |
| `Ehlers_LW:lw_cycle_sell/close` | 0.583 | 0.605 | 0.567 | 0.593 | – | True |
| `KCBreakout:kc_upper/close` | 0.575 | 0.582 | 0.543 | 0.572 | – | True |
| `KCRevert:kc_upper/close` | 0.575 | 0.582 | 0.543 | 0.572 | – | True |

## Signals (all bars): top 15 by median lift, beside the comparators

| signal | median lift (15 test windows) | fire rate | E0 agrees |
|---|---|---|---|
| `raw_sell:KST` | 1.68 | 0.002 | True |
| `cmp:width_book` | 1.68 | 0.009 | True |
| `raw_sell:TRIX` | 1.66 | 0.004 | True |
| `raw_sell:EWO` | 1.59 | 0.002 | True |
| `raw_sell:COPP` | 1.59 | 0.005 | True |
| `raw_buy:COPP` | 1.58 | 0.027 | True |
| `raw_buy:ROC` | 1.53 | 0.010 | True |
| `raw_buy:KST` | 1.53 | 0.035 | True |
| `raw_buy:WOBV` | 1.51 | 0.000 | True |
| `raw_sell:WOBV` | 1.41 | 0.000 | False |
| `raw_buy:TRIX` | 1.41 | 0.061 | True |
| `base:gbb_candidate` | 1.40 | 0.154 | True |
| `raw_buy:EWO` | 1.37 | 0.124 | True |
| `raw_buy:PPO` | 1.37 | 0.072 | True |
| `raw_sell:ROC` | 1.31 | 0.004 | True |

## Signal pairs (both must fire)

- all bars: 83 pass the rule, 20 survivors, 19 confirmed (E0 agrees).
  - `raw_buy:KST` AND `raw_buy:VBM`: early 2.03, late 1.52, E0 1.46
  - `raw_buy:COPP` AND `raw_buy:VBM`: early 1.96, late 1.68, E0 1.48
  - `raw_buy:KST` AND `raw_buy:ZScore`: early 1.96, late 1.85, E0 1.52
  - `raw_buy:COPP` AND `raw_buy:WTO`: early 1.95, late 1.71, E0 1.49
  - `raw_buy:COPP` AND `raw_buy:MI`: early 1.93, late 1.69, E0 1.49
  - `raw_buy:CCI` AND `raw_buy:KST`: early 1.93, late 1.89, E0 1.47
  - `raw_buy:TRIX` AND `raw_buy:ZScore`: early 1.87, late 1.70, E0 1.43
  - `raw_buy:ROC` AND `raw_buy:STC`: early 1.86, late 1.57, E0 1.38
  - `raw_buy:MI` AND `raw_buy:ROC`: early 1.86, late 1.52, E0 1.45
  - `raw_buy:CCI` AND `raw_buy:TRIX`: early 1.85, late 1.64, E0 1.42
- candidate bars: 67 pass the rule, 20 survivors, 13 confirmed (E0 agrees).
  - `raw_buy:CCI` AND `raw_buy:COPP`: early 1.45, late 1.48, E0 1.24
  - `raw_buy:CCI` AND `raw_buy:WTO`: early 1.44, late 1.28, E0 1.12
  - `raw_buy:CCI` AND `raw_buy:KST`: early 1.43, late 1.35, E0 1.24
  - `raw_buy:WTO` AND `raw_buy:ZScore`: early 1.43, late 1.31, E0 1.12
  - `raw_buy:COPP` AND `raw_buy:WTO`: early 1.43, late 1.29, E0 1.19
  - `raw_buy:KST` AND `raw_buy:ZScore`: early 1.42, late 1.40, E0 1.25
  - `raw_buy:HTSINE` AND `raw_buy:ZScore`: early 1.40, late 1.34, E0 1.08
  - `raw_buy:KST` AND `raw_buy:WTO`: early 1.36, late 1.23, E0 1.15
  - `raw_buy:FVE` AND `raw_buy:ZScore`: early 1.35, late 1.32, E0 1.12
  - `raw_buy:SMACross` AND `raw_buy:ZScore`: early 1.35, late 1.28, E0 1.13

## Beyond ARO: AUC within ARO p_buy quintiles (test windows; descriptive)

- ARO p_buy AUC on label:buy [candidate]: early 0.607, late 0.597
- ARO p_buy AUC on label:tb_buy [all]: early 0.597, late 0.574
- ARO p_buy AUC on label:tb_buy [candidate]: early 0.599, late 0.587

**label:tb_buy [all]:** 54 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `base:atr_pct` | 0.607 | 0.553 | 0.652 | 0.590 |
| `base:atr_pct_roll` | 0.604 | 0.553 | 0.645 | 0.585 |
| `DCBreakout:dc_upper/close` | 0.555 | 0.547 | 0.605 | 0.580 |
| `DCRevert:dc_upper/close` | 0.555 | 0.547 | 0.605 | 0.580 |
| `VBM:vbm` | 0.593 | 0.545 | 0.642 | 0.577 |
| `Ehlers_DC:dc_upper/close` | 0.556 | 0.543 | 0.605 | 0.580 |
| `LWBreakout:atr/close` | 0.592 | 0.543 | 0.642 | 0.578 |
| `LWRevert:atr/close` | 0.592 | 0.543 | 0.642 | 0.578 |
| `LWBreakout:level_pct_diff` | 0.592 | 0.543 | 0.642 | 0.578 |
| `ZScore:rolling_std/close` | 0.574 | 0.542 | 0.638 | 0.596 |
| `base:bb_width` | 0.573 | 0.541 | 0.637 | 0.595 |
| `Ehlers_LW:lw_atr/close` | 0.589 | 0.540 | 0.641 | 0.577 |

**label:tb_buy [candidate]:** 22 indicator columns keep strength >= 0.02 within ARO quintiles in both halves.

| column | resid AUC early | late | plain AUC early | late |
|---|---|---|---|---|
| `base:atr_pct_roll` | 0.558 | 0.551 | 0.603 | 0.583 |
| `base:atr_pct` | 0.560 | 0.542 | 0.602 | 0.584 |
| `Ehlers_KC:kc_upper/close` | 0.542 | 0.531 | 0.587 | 0.573 |
| `ni:ni_ms_vr` | 0.541 | 0.529 | 0.565 | 0.549 |
| `KCBreakout:kc_upper/close` | 0.537 | 0.527 | 0.578 | 0.569 |
| `KCRevert:kc_upper/close` | 0.537 | 0.527 | 0.578 | 0.569 |
| `Ehlers_Ichimoku:tenkan_sen/close` | 0.527 | 0.528 | 0.559 | 0.564 |
| `Ichimoku:tenkan_sen/close` | 0.527 | 0.528 | 0.559 | 0.564 |
| `SAR:sar/close` | 0.526 | 0.530 | 0.559 | 0.576 |
| `base:fisher_wr` | 0.474 | 0.474 | 0.463 | 0.459 |
| `base:sar_ratio` | 0.524 | 0.525 | 0.545 | 0.558 |
| `PSAR:psar/close` | 0.524 | 0.528 | 0.557 | 0.577 |

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
