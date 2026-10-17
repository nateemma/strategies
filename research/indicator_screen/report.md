# 037 Indicator screen: report

**SCREEN (descriptive).** Association with a label or a regime decides nothing about acceptance (Principle III). Every candidate below needs a backtest in a later feature before any P&L claim (FR-025).

## Conduct

- Pre-registration lock: `4ccb4ad0`.
- Windows scored (labels): E0, f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14.
- Populations: all bars, and gbb candidate bars; the 10 pairs pooled per window; rows with `meta:data_ok` only (XRP's gap-filled 2021-01 → 2023-07 bars are not data).
- Persistence: early (2023–24) and late (2025–26) are adjacent, so E0 (2021-04 → 2022-12) must agree for a single column or pair to count. Models and A2 cannot be scored on E0: "distant era not tested"; a later backtest must cover 2022.

- Instrument amendment: 2026-10-16, after the first gate run FAILED under the original rule (both columns in the top 5 by median strength): guard_metric ranked 13 with a stable AUC of 0.21-0.27 in every gate window. Cause: the label uses guard_metric < 0.3 (RMI 65) as a loose overbought veto that removes ~35% of bars, so it cannot rank within the remaining 65%; bb_width (> 0.035 on ~30% of bars) and its volatility look-alikes rank above it. The check is meant to show the screen recovers the label's known inputs with the right sign, so it now asks for the expected direction and strength >= 0.15 in EVERY gate window. Also fixed: base:gbb_candidate (the label's own entry rule) was scored as a column and ranked #1; it now only defines the candidate population. Amendment approved by the user.

## Harvest

- In-scope classes: 67; columns: 797.
- Excluded classes: `AllIndicators (harvest_wrapper)`, `BreakoutBbOpen (trigger_arm)`, `BreakoutGbbOpen (trigger_arm)`, `BreakoutOpen (trigger_arm)`, `BreakoutVolOpen (trigger_arm)`, `GbbGuardGridOpen (trigger_arm)`, `GbbGuardedOpen (trigger_arm)`, `GbbOpen (trigger_arm)`, `GbbRobustOpen (trigger_arm)`, `GeometryOpen (trigger_arm)`, `ZLEMACross (signal_error)`
- Lookahead (truncation test) FAIL, excluded: `Ehlers_Ichimoku:chikou_span`, `STOCHRSI:stochrsi`, `raw_buy:STOCHRSI`, `raw_sell:STOCHRSI`.
- Column tags (added after the lock at the user's request; no score changed): calendar 6, indicator 491, non_stationary 2, price_level 93, price_scaled 51. Tagged columns are listed apart from candidates.
- Extra columns added 2026-10-16, after the pre-registration lock (user request); rules unchanged: 173 (`raw8:*` raw-price inputs, `ni:*` 035 inputs, `<price column>/close`); scored by the locked rules and refit into every model.
- Constant on some pair (never fires / never varies): `Ehlers_DC:trend_entry_trigger`, `raw_buy:Combo`, `raw_buy:DYMI`, `raw_buy:Ehlers_DC:trend`, `raw_buy:MACDsignal`, `raw_buy:PZO`, `raw_sell:DYMI`, `raw_sell:Ehlers_DC:cycle`, `raw_sell:MACDsignal`, `raw_sell:TSI`.

## Instrument check (SC-002)

- Gate (e0+early): **PASS** (expected direction and strength >= 0.15 in every window).
  - `base:bb_width`: pass=True, rank 6, AUC by window E0 0.859, f00 0.903, f01 0.949, f02 0.948, f03 0.906, f04 0.884, f05 0.911, f06 0.896, f07 0.861
  - `base:guard_metric`: pass=True, rank 90, AUC by window E0 0.224, f00 0.212, f01 0.259, f02 0.268, f03 0.223, f04 0.240, f05 0.252, f06 0.223, f07 0.243
- Confirmation over the 15 test windows: **PASS**.

### label:buy, all bars: top 15 by median strength

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `base:bb_upperband/close` | indicator | 0.927 | 0.928 | 0.925 | 0.901 | 1.00 | 201 | -0.61 | 0.60 |
| `BBBreakout:bb_upperband/close` | indicator | 0.925 | 0.929 | 0.923 | 0.901 | 1.00 | 201 | -0.64 | 0.55 |
| `BBRevert:bb_upperband/close` | indicator | 0.925 | 0.929 | 0.923 | 0.901 | 1.00 | 201 | -0.64 | 0.55 |
| `Ehlers_DC:dc_upper/close` | indicator | 0.907 | 0.911 | 0.903 | 0.896 | 1.00 | 201 | -0.62 | 0.50 |
| `ZScore:rolling_std/close` | indicator | 0.906 | 0.908 | 0.904 | 0.863 | 1.00 | 159 | 0.01 | 1.00 |
| `base:bb_width` | indicator | 0.903 | 0.905 | 0.901 | 0.859 | 1.00 | 159 | 0.02 | 1.00 |
| `DCBreakout:dc_upper/close` | indicator | 0.902 | 0.906 | 0.895 | 0.892 | 1.00 | 201 | -0.63 | 0.47 |
| `DCRevert:dc_upper/close` | indicator | 0.902 | 0.906 | 0.895 | 0.892 | 1.00 | 201 | -0.63 | 0.47 |
| `KCBreakout:kc_upper/close` | indicator | 0.887 | 0.890 | 0.882 | 0.879 | 1.00 | 201 | -0.76 | 0.26 |
| `KCRevert:kc_upper/close` | indicator | 0.887 | 0.890 | 0.882 | 0.879 | 1.00 | 201 | -0.76 | 0.26 |
| `Ehlers_KC:kc_upper/close` | indicator | 0.878 | 0.884 | 0.871 | 0.871 | 1.00 | 201 | -0.69 | 0.34 |
| `BBBreakout:bb_middleband/close` | indicator | 0.824 | 0.820 | 0.824 | 0.809 | 1.00 | 203 | -0.87 | -0.02 |
| `BBRevert:bb_middleband/close` | indicator | 0.824 | 0.820 | 0.824 | 0.809 | 1.00 | 203 | -0.87 | -0.02 |
| `KCRevert:kc_middle/close` | indicator | 0.823 | 0.818 | 0.824 | 0.810 | 1.00 | 203 | -0.87 | -0.02 |
| `ZScore:rolling_mean/close` | indicator | 0.822 | 0.813 | 0.824 | 0.803 | 1.00 | 203 | -0.85 | -0.02 |

### label:buy, candidate bars: top 15 by median strength

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `base:atr_pct_roll` | indicator | 0.594 | 0.595 | 0.576 | 0.583 | 1.00 | 83 | 0.28 | 0.29 |
| `Ehlers_KC:kc_upper/close` | indicator | 0.593 | 0.597 | 0.582 | 0.569 | 1.00 | 201 | -0.48 | 0.46 |
| `base:atr_pct` | indicator | 0.593 | 0.595 | 0.576 | 0.580 | 1.00 | 83 | 0.22 | 0.39 |
| `KCBreakout:kc_upper/close` | indicator | 0.587 | 0.590 | 0.581 | 0.565 | 1.00 | 201 | -0.56 | 0.50 |
| `KCRevert:kc_upper/close` | indicator | 0.587 | 0.590 | 0.581 | 0.565 | 1.00 | 201 | -0.56 | 0.50 |
| `Ehlers_DC:dc_upper/close` | indicator | 0.584 | 0.585 | 0.577 | 0.559 | 1.00 | 201 | -0.50 | 0.59 |
| `DCBreakout:dc_upper/close` | indicator | 0.580 | 0.583 | 0.579 | 0.557 | 1.00 | 201 | -0.48 | 0.57 |
| `DCRevert:dc_upper/close` | indicator | 0.580 | 0.583 | 0.579 | 0.557 | 1.00 | 201 | -0.48 | 0.57 |
| `EMACross:ema_short/close` | indicator | 0.578 | 0.579 | 0.566 | 0.547 | 1.00 | 116 | -0.35 | 0.09 |
| `base:bb_upperband/close` | indicator | 0.575 | 0.581 | 0.565 | 0.549 | 1.00 | 201 | -0.62 | 0.67 |
| `SAR:sar/close` | indicator | 0.575 | 0.574 | 0.583 | 0.551 | 1.00 | 136 | -0.39 | 0.16 |
| `LWBreakout:atr/close` | indicator | 0.574 | 0.580 | 0.560 | 0.568 | 1.00 | 83 | 0.25 | 0.44 |
| `LWRevert:atr/close` | indicator | 0.574 | 0.580 | 0.560 | 0.568 | 1.00 | 83 | 0.25 | 0.44 |
| `LWBreakout:level_pct_diff` | indicator | 0.574 | 0.580 | 0.560 | 0.568 | 1.00 | 83 | 0.25 | 0.44 |
| `Ehlers_Ichimoku:tenkan_sen/close` | indicator | 0.574 | 0.575 | 0.570 | 0.547 | 1.00 | 115 | -0.40 | 0.12 |

### label:sell, all bars: top 15 by median strength

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `base:bb_lowerband/close` | indicator | 0.047 | 0.045 | 0.050 | 0.065 | 1.00 | 192 | -0.61 | -0.63 |
| `BBBreakout:bb_lowerband/close` | indicator | 0.048 | 0.046 | 0.053 | 0.067 | 1.00 | 192 | -0.64 | -0.58 |
| `BBRevert:bb_lowerband/close` | indicator | 0.048 | 0.046 | 0.053 | 0.067 | 1.00 | 192 | -0.64 | -0.58 |
| `Ehlers_DC:dc_lower/close` | indicator | 0.061 | 0.061 | 0.061 | 0.077 | 1.00 | 193 | -0.59 | -0.55 |
| `KCBreakout:kc_lower/close` | indicator | 0.063 | 0.055 | 0.065 | 0.077 | 1.00 | 203 | -0.77 | -0.30 |
| `KCRevert:kc_lower/close` | indicator | 0.063 | 0.055 | 0.065 | 0.077 | 1.00 | 203 | -0.77 | -0.30 |
| `KCRevert:kc_middle/close` | indicator | 0.065 | 0.064 | 0.068 | 0.076 | 1.00 | 203 | -0.87 | -0.02 |
| `BBBreakout:bb_middleband/close` | indicator | 0.066 | 0.065 | 0.068 | 0.077 | 1.00 | 203 | -0.87 | -0.02 |
| `BBRevert:bb_middleband/close` | indicator | 0.066 | 0.065 | 0.068 | 0.077 | 1.00 | 203 | -0.87 | -0.02 |
| `DCBreakout:dc_lower/close` | indicator | 0.066 | 0.066 | 0.069 | 0.083 | 1.00 | 193 | -0.61 | -0.52 |
| `DCRevert:dc_lower/close` | indicator | 0.066 | 0.066 | 0.069 | 0.083 | 1.00 | 193 | -0.61 | -0.52 |
| `EhlersBase:dpo/close` | indicator | 0.929 | 0.932 | 0.916 | 0.912 | 1.00 | 203 | 0.84 | 0.02 |
| `Ehlers_BB:dpo/close` | indicator | 0.929 | 0.932 | 0.916 | 0.912 | 1.00 | 203 | 0.84 | 0.02 |
| `Ehlers_DC:dpo/close` | indicator | 0.929 | 0.932 | 0.916 | 0.912 | 1.00 | 203 | 0.84 | 0.02 |
| `Ehlers_DYMI:dpo/close` | indicator | 0.929 | 0.932 | 0.916 | 0.912 | 1.00 | 203 | 0.84 | 0.02 |

### label:sell, candidate bars: top 15 by median strength

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|

### Residual beyond bb_width + guard_metric (label:buy, candidate bars)

| column | strat AUC early | late | E0 | AUC | width-sort AUC | persistence |
|---|---|---|---|---|---|---|
| `KCBreakout:kc_upper/close` | 0.592 | 0.563 | 0.567 | 0.587 | 0.552 | E0 agrees |
| `KCRevert:kc_upper/close` | 0.592 | 0.563 | 0.567 | 0.587 | 0.552 | E0 agrees |
| `Ehlers_KC:kc_upper/close` | 0.589 | 0.572 | 0.567 | 0.593 | 0.552 | E0 agrees |
| `base:atr_pct_roll` | 0.582 | 0.571 | 0.578 | 0.594 | 0.552 | E0 agrees |
| `Ehlers_KC:kc_mid/close` | 0.581 | 0.563 | 0.546 | 0.573 | 0.552 | E0 agrees |
| `base:ema_fast/close` | 0.581 | 0.563 | 0.546 | 0.573 | 0.552 | E0 agrees |
| `base:atr_pct` | 0.581 | 0.567 | 0.574 | 0.593 | 0.552 | E0 agrees |
| `EMACross:ema_short/close` | 0.579 | 0.564 | 0.546 | 0.578 | 0.552 | E0 agrees |
| `ZScore:rolling_mean/close` | 0.577 | 0.567 | 0.548 | 0.569 | 0.552 | E0 agrees |
| `base:bb_middleband/close` | 0.577 | 0.567 | 0.548 | 0.569 | 0.552 | E0 agrees |
| `base:bb_upperband/close` | 0.577 | 0.568 | 0.547 | 0.575 | 0.552 | E0 agrees |
| `Ehlers_Ichimoku:tenkan_sen/close` | 0.576 | 0.567 | 0.547 | 0.574 | 0.552 | E0 agrees |
| `Ichimoku:tenkan_sen/close` | 0.576 | 0.567 | 0.547 | 0.574 | 0.552 | E0 agrees |
| `SMACross:sma_short/close` | 0.575 | 0.562 | 0.544 | 0.574 | 0.552 | E0 agrees |
| `ZScore:zscore` | 0.426 | 0.445 | 0.455 | 0.444 | 0.552 | E0 agrees |

## Signal pairs (both must fire)

### A1:288, all bars: 2775 pairs, 25 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:CCI` AND `raw_buy:KST` | 1.45 | 0.99 / 1.14 | 1.65 | 1.18 | yes |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.39 | 0.99 / 1.14 | 1.59 | 1.22 | yes |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.39 | 0.99 / 1.14 | 1.59 | 1.22 | yes |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:ROC` | 1.37 | 1.01 / 1.23 | 1.44 | 1.02 | early+late only |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.36 | 1.08 / 1.14 | 1.13 | 1.25 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.36 | 1.14 / 0.98 | 1.62 | 1.25 | yes |
| `raw_buy:BOP` AND `raw_buy:KST` | 1.35 | 1.06 / 1.14 | 1.12 | 1.25 | no |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.35 | 0.99 / 1.14 | 1.65 | 1.22 | yes |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.34 | 0.99 / 1.20 | 1.83 | 1.17 | yes |
| `raw_buy:COPP` AND `raw_buy:WTO` | 1.32 | 1.20 / 1.02 | 1.52 | 1.30 | yes |
| `raw_buy:KST` AND `raw_buy:VZO` | 1.30 | 1.14 / 0.98 | 1.15 | 1.32 | no |
| `raw_buy:HTSINE` AND `raw_buy:PPO` | 1.29 | 1.00 / 1.08 | 1.39 | 1.09 | early+late only |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.29 | 0.99 / 1.09 | 1.63 | 1.19 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.28 | 1.14 / 1.02 | 1.52 | 1.20 | yes |
| `raw_buy:CCI` AND `raw_buy:PPO` | 1.28 | 0.99 / 1.08 | 1.71 | 1.14 | yes |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 1.28 | 1.00 / 1.14 | 1.45 | 1.20 | yes |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.27 | 1.14 / 0.98 | 1.49 | 1.20 | yes |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:PPO` | 1.26 | 1.03 / 1.08 | 1.32 | 1.11 | yes |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:PPO` | 1.25 | 1.02 / 1.08 | 1.36 | 0.96 | early+late only |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:TRIX` | 1.25 | 1.03 / 1.09 | 1.40 | 1.14 | early+late only |

### A1:288, candidate bars: 2775 pairs, 14 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.32 | 1.09 / 1.02 | 1.38 | 1.17 | yes |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.30 | 1.00 / 1.09 | 1.37 | 1.14 | yes |
| `raw_buy:BOP` AND `raw_buy:KST` | 1.24 | 1.04 / 1.09 | 1.06 | 1.17 | no |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.24 | 1.04 / 1.09 | 1.07 | 1.17 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.23 | 1.09 / 1.08 | 1.35 | 1.13 | yes |
| `raw_buy:ROC` AND `raw_buy:TEMACross` | 1.23 | 1.07 / 1.07 | 1.45 | 1.22 | yes |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:ROC` | 1.23 | 1.00 / 1.07 | 1.17 | 0.95 | no |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:KST` | 1.22 | 1.05 / 1.09 | 1.42 | 0.93 | early+late only |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:KST` | 1.22 | 1.04 / 1.09 | 1.37 | 1.09 | early+late only |
| `raw_buy:EMV` AND `raw_buy:VBM` | 1.22 | 1.01 / 1.04 | 0.85 | 1.09 | no |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.21 | 1.06 / 1.04 | 1.38 | 1.06 | early+late only |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.21 | 1.04 / 1.02 | 1.33 | 1.14 | yes |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.20 | 1.01 / 1.09 | 1.30 | 1.14 | yes |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.20 | 1.01 / 1.09 | 1.30 | 1.14 | yes |

### A1:48, all bars: 2775 pairs, 48 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.48 | 1.01 / 1.11 | 1.14 | 1.23 | no |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:ROC` | 1.41 | 1.01 / 1.10 | 1.34 | 1.05 | early+late only |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:KST` | 1.41 | 1.02 / 1.11 | 1.37 | 1.25 | yes |
| `raw_buy:KCBreakout` AND `raw_buy:TRIX` | 1.36 | 1.00 / 1.10 | 1.10 | 1.22 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.36 | 1.11 / 1.02 | 1.19 | 1.39 | no |
| `raw_buy:COPP` AND `raw_buy:STC` | 1.36 | 1.07 / 1.01 | 1.11 | 1.30 | no |
| `raw_buy:Aroon` AND `raw_buy:KST` | 1.35 | 0.99 / 1.11 | 1.01 | 1.17 | no |
| `raw_buy:HTSINE` AND `raw_buy:PPO` | 1.35 | 1.01 / 1.06 | 1.14 | 1.12 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.34 | 1.11 / 0.99 | 1.09 | 1.37 | no |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.33 | 1.10 / 1.01 | 1.28 | 1.29 | yes |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:PPO` | 1.32 | 1.02 / 1.06 | 1.22 | 1.11 | early+late only |
| `raw_buy:HTTREND` AND `raw_buy:VBM` | 1.32 | 1.00 / 1.01 | 0.89 | 1.13 | no |
| `raw_buy:HTSINE` AND `raw_buy:TRIX` | 1.31 | 1.01 / 1.10 | 1.19 | 1.22 | no |
| `raw_buy:Ehlers_ind_DYMI:trend` AND `raw_buy:ROC` | 1.31 | 0.99 / 1.10 | 1.18 | 1.22 | no |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.30 | 0.98 / 1.11 | 1.23 | 1.39 | yes |
| `raw_buy:FVE` AND `raw_buy:ROC` | 1.30 | 1.00 / 1.10 | 1.13 | 1.18 | no |
| `raw_buy:BOP` AND `raw_buy:VBM` | 1.30 | 0.95 / 1.01 | 1.04 | 1.02 | no |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.29 | 0.98 / 1.10 | 1.06 | 1.37 | no |
| `raw_buy:COPP` AND `raw_buy:Ehlers_ind_DYMI:trend` | 1.29 | 1.07 / 0.99 | 1.11 | 1.26 | no |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.28 | 1.10 / 1.02 | 0.94 | 1.36 | no |

### A1:48, candidate bars: 2775 pairs, 34 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.46 | 1.02 / 1.11 | 1.19 | 1.13 | no |
| `raw_buy:HTSINE` AND `raw_buy:TRIX` | 1.36 | 1.02 / 1.03 | 1.12 | 1.14 | no |
| `raw_buy:HTTREND` AND `raw_buy:MI` | 1.32 | 1.00 / 1.08 | 0.84 | 1.06 | no |
| `raw_buy:FVE` AND `raw_buy:KST` | 1.31 | 1.05 / 1.11 | 1.07 | 1.19 | no |
| `raw_buy:VFI` AND `raw_buy:WTO` | 1.30 | 1.08 / 1.05 | 1.09 | 1.04 | no |
| `raw_buy:Aroon` AND `raw_buy:KST` | 1.29 | 1.04 / 1.11 | 1.22 | 1.10 | early+late only |
| `raw_buy:EWO` AND `raw_buy:EhlersBase:cycle` | 1.28 | 1.01 / 1.05 | 1.07 | 1.14 | no |
| `raw_buy:COPP` AND `raw_buy:STC` | 1.28 | 1.05 / 1.04 | 1.22 | 1.18 | yes |
| `raw_buy:MI` AND `raw_buy:VFI` | 1.27 | 1.08 / 1.08 | 1.13 | 1.04 | no |
| `raw_buy:TEMACross` AND `raw_buy:WTO` | 1.27 | 1.08 / 1.05 | 1.15 | 1.11 | no |
| `raw_buy:DCRevert` AND `raw_buy:VFI` | 1.27 | 1.01 / 1.08 | 1.10 | 0.99 | no |
| `raw_buy:COPP` AND `raw_buy:VBM` | 1.27 | 1.05 / 1.02 | 1.06 | 1.22 | no |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:ROC` | 1.26 | 1.00 / 1.06 | 1.26 | 0.96 | early+late only |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.26 | 1.03 / 1.05 | 0.99 | 1.28 | no |
| `raw_buy:COPP` AND `raw_buy:FVE` | 1.26 | 1.05 / 1.05 | 1.02 | 1.14 | no |
| `raw_buy:COPP` AND `raw_buy:ZScore` | 1.25 | 1.05 / 1.03 | 1.25 | 1.09 | yes |
| `raw_buy:CMO` AND `raw_buy:FVE` | 1.25 | 0.91 / 1.05 | 0.94 | 0.82 | no |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.25 | 1.06 / 1.04 | 1.23 | 1.17 | yes |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:TRIX` | 1.25 | 1.05 / 1.03 | 1.17 | 1.13 | no |
| `raw_buy:KST` AND `raw_buy:STC` | 1.24 | 1.11 / 1.04 | 1.24 | 1.19 | yes |

### A1:672, all bars: 2775 pairs, 21 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:VZO` | 1.37 | 1.18 / 0.99 | 0.96 | 1.32 | no |
| `raw_buy:COPP` AND `raw_buy:WFRACTAL` | 1.35 | 1.15 / 0.97 | 1.33 | 1.16 | yes |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.35 | 1.00 / 1.18 | 1.31 | 1.31 | yes |
| `raw_buy:KST` AND `raw_buy:UO` | 1.34 | 1.18 / 0.97 | 1.19 | 1.31 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.34 | 1.18 / 1.00 | 1.18 | 1.26 | no |
| `raw_buy:DCRevert` AND `raw_buy:KST` | 1.32 | 0.97 / 1.18 | 1.09 | 1.26 | no |
| `raw_buy:KST` AND `raw_buy:VFI` | 1.31 | 1.18 / 1.07 | 0.96 | 1.06 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.31 | 1.18 / 1.01 | 1.21 | 1.16 | yes |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.30 | 1.09 / 1.18 | 0.96 | 1.14 | no |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.30 | 1.18 / 1.00 | 0.91 | 1.16 | no |
| `raw_buy:ROC` AND `raw_buy:VZO` | 1.29 | 1.12 / 0.99 | 0.89 | 1.19 | no |
| `raw_buy:COPP` AND `raw_buy:VFI` | 1.29 | 1.15 / 1.07 | 1.09 | 1.05 | no |
| `raw_buy:ROC` AND `raw_buy:VFI` | 1.27 | 1.12 / 1.07 | 1.01 | 1.04 | no |
| `raw_buy:BOP` AND `raw_buy:TEMACross` | 1.27 | 1.06 / 1.00 | 1.35 | 1.20 | yes |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:PPO` | 1.26 | 1.00 / 1.11 | 1.19 | 0.93 | no |
| `raw_buy:PPO` AND `raw_buy:SMACross` | 1.26 | 1.11 / 1.00 | 0.92 | 0.95 | no |
| `raw_buy:ROC` AND `raw_buy:TEMACross` | 1.24 | 1.12 / 1.00 | 0.76 | 1.33 | no |
| `raw_buy:CMO` AND `raw_buy:EMV` | 1.23 | 0.98 / 1.06 | 0.87 | 0.87 | no |
| `raw_buy:HTTREND` AND `raw_buy:VFI` | 1.22 | 1.09 / 1.07 | 1.22 | 1.01 | early+late only |
| `raw_buy:EWO` AND `raw_buy:HTSINE` | 1.22 | 1.10 / 1.00 | 1.27 | 1.08 | yes |

### A1:672, candidate bars: 2775 pairs, 6 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:CCI` AND `raw_buy:KST` | 1.31 | 1.03 / 1.11 | 1.21 | 1.26 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.26 | 1.11 / 1.07 | 1.23 | 1.10 | yes |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.25 | 1.03 / 1.05 | 1.27 | 1.17 | yes |
| `raw_buy:KST` AND `raw_buy:VZO` | 1.25 | 1.11 / 1.04 | 0.96 | 1.25 | no |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.23 | 1.00 / 1.11 | 1.00 | 1.11 | no |
| `raw_buy:BOP` AND `raw_buy:KST` | 1.22 | 0.98 / 1.11 | 0.98 | 1.10 | no |

### A2:288, all bars: 2775 pairs, 49 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:VFI` | 1.65 | 1.15 / 1.05 | 1.65 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:VFI` | 1.58 | 1.15 / 1.05 | 1.68 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:VFI` | 1.54 | 1.14 / 1.05 | 1.48 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.52 | 1.15 / 0.96 | 1.77 | – | early+late only |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.51 | 0.98 / 1.15 | 1.68 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:VFI` | 1.48 | 1.22 / 1.05 | 1.62 | – | early+late only |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.45 | 1.00 / 1.15 | 1.75 | – | early+late only |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.45 | 1.00 / 1.15 | 1.75 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:MI` | 1.44 | 1.15 / 1.00 | 1.38 | – | no |
| `raw_buy:BBRevert` AND `raw_buy:TRIX` | 1.42 | 1.00 / 1.22 | 1.52 | – | early+late only |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:TRIX` | 1.42 | 1.00 / 1.22 | 1.52 | – | early+late only |
| `raw_buy:PPO` AND `raw_buy:VFI` | 1.41 | 1.13 / 1.05 | 1.47 | – | early+late only |
| `raw_buy:CCI` AND `raw_buy:PPO` | 1.40 | 0.96 / 1.13 | 1.62 | – | early+late only |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 1.39 | 1.01 / 1.15 | 1.74 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.38 | 1.22 / 1.01 | 1.72 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:TRIX` | 1.37 | 1.14 / 1.22 | 1.41 | – | early+late only |
| `raw_buy:EWO` AND `raw_buy:HTDCP:cycle` | 1.36 | 1.15 / 1.03 | 1.73 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:Ehlers_LW:cycle` | 1.36 | 1.15 / 1.01 | 1.52 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:ROC` | 1.36 | 1.15 / 1.14 | 1.48 | – | no |
| `raw_buy:Ehlers_KC:cycle` AND `raw_buy:KST` | 1.35 | 1.00 / 1.15 | 1.62 | – | early+late only |

### A2:288, candidate bars: 2775 pairs, 23 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:VFI` | 1.31 | 1.02 / 1.10 | 1.47 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.31 | 1.02 / 1.02 | 1.45 | – | early+late only |
| `raw_buy:HTSINE` AND `raw_buy:VFI` | 1.30 | 1.04 / 1.10 | 1.32 | – | no |
| `raw_buy:FVE` AND `raw_buy:VFI` | 1.27 | 1.04 / 1.10 | 1.35 | – | early+late only |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.27 | 1.00 / 1.02 | 1.42 | – | early+late only |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.27 | 1.00 / 1.02 | 1.42 | – | early+late only |
| `raw_buy:Ehlers_DC:cycle` AND `raw_buy:TRIX` | 1.26 | 1.04 / 1.02 | 1.25 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.26 | 1.02 / 1.02 | 1.37 | – | early+late only |
| `raw_buy:HTSINE` AND `raw_buy:TRIX` | 1.26 | 1.04 / 1.02 | 1.63 | – | early+late only |
| `raw_buy:PSAR` AND `raw_buy:VFI` | 1.25 | 1.04 / 1.10 | 1.28 | – | no |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.25 | 1.02 / 1.06 | 1.43 | – | early+late only |
| `raw_buy:RVOL` AND `raw_buy:VFI` | 1.24 | 1.04 / 1.10 | 1.24 | – | no |
| `raw_buy:KST` AND `raw_buy:VZO` | 1.24 | 1.02 / 0.92 | 1.41 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:VFI` | 1.24 | 1.02 / 1.10 | 1.45 | – | early+late only |
| `raw_buy:PSAR` AND `raw_buy:WILLR` | 1.24 | 1.04 / 0.99 | 1.12 | – | no |
| `raw_buy:CCI` AND `raw_buy:PPO` | 1.23 | 1.06 / 0.98 | 1.34 | – | early+late only |
| `raw_buy:TEMACross` AND `raw_buy:VFI` | 1.23 | 1.03 / 1.10 | 1.49 | – | early+late only |
| `raw_buy:PPO` AND `raw_buy:VFI` | 1.23 | 0.98 / 1.10 | 1.32 | – | no |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.22 | 1.02 / 1.03 | 1.42 | – | early+late only |
| `raw_buy:CCI` AND `raw_buy:STC` | 1.21 | 1.06 / 1.03 | 1.13 | – | no |

### A2:48, all bars: 2775 pairs, 78 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:STC` | 1.65 | 1.13 / 1.03 | 1.75 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:VFI` | 1.63 | 1.10 / 0.97 | 1.91 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.62 | 1.10 / 1.03 | 1.56 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:VFI` | 1.55 | 1.14 / 0.97 | 1.41 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:STC` | 1.51 | 1.10 / 1.03 | 1.81 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:TEMACross` | 1.50 | 1.13 / 1.01 | 1.63 | – | early+late only |
| `raw_buy:MI` AND `raw_buy:ROC` | 1.50 | 0.93 / 1.10 | 1.45 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:VFI` | 1.47 | 1.10 / 0.97 | 1.59 | – | early+late only |
| `raw_buy:FVE` AND `raw_buy:TRIX` | 1.47 | 1.04 / 1.14 | 1.45 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:MI` | 1.44 | 1.10 / 0.93 | 1.57 | – | early+late only |
| `raw_buy:STC` AND `raw_buy:TRIX` | 1.43 | 1.03 / 1.14 | 1.60 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:VFI` | 1.42 | 1.13 / 0.97 | 1.70 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.42 | 1.14 / 0.89 | 1.51 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:TEMACross` | 1.41 | 1.10 / 1.01 | 1.70 | – | early+late only |
| `raw_buy:EWO` AND `raw_buy:WTO` | 1.41 | 1.11 / 1.04 | 1.35 | – | early+late only |
| `raw_buy:EMV` AND `raw_buy:EWO` | 1.40 | 1.13 / 1.11 | 1.22 | – | early+late only |
| `raw_buy:LWRevert` AND `raw_buy:TRIX` | 1.39 | 1.01 / 1.14 | 1.39 | – | early+late only |
| `raw_buy:EMV` AND `raw_buy:WILLR` | 1.39 | 1.13 / 0.86 | 1.00 | – | no |
| `raw_buy:COPP` AND `raw_buy:Ehlers_LW:cycle` | 1.37 | 1.13 / 0.98 | 1.32 | – | early+late only |
| `raw_buy:LWRevert` AND `raw_buy:PPO` | 1.36 | 1.01 / 1.12 | 1.18 | – | no |

### A2:48, candidate bars: 2775 pairs, 25 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:EMV` AND `raw_buy:MI` | 1.41 | 1.09 / 1.09 | 1.29 | – | early+late only |
| `raw_buy:FVE` AND `raw_buy:VFI` | 1.36 | 1.11 / 1.06 | 1.26 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:TEMACross` | 1.35 | 0.91 / 1.05 | 1.37 | – | early+late only |
| `raw_buy:MI` AND `raw_buy:ROC` | 1.33 | 1.09 / 0.91 | 1.18 | – | no |
| `raw_buy:MI` AND `raw_buy:PSAR` | 1.31 | 1.09 / 1.10 | 1.06 | – | no |
| `raw_buy:STC` AND `raw_buy:VFI` | 1.30 | 1.07 / 1.06 | 1.15 | – | no |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.30 | 0.91 / 1.07 | 1.24 | – | early+late only |
| `raw_buy:EWO` AND `raw_buy:TEMACross` | 1.28 | 0.97 / 1.05 | 1.19 | – | no |
| `raw_buy:KST` AND `raw_buy:VFI` | 1.28 | 0.89 / 1.06 | 1.40 | – | early+late only |
| `raw_buy:EMV` AND `raw_buy:VFI` | 1.27 | 1.09 / 1.06 | 1.29 | – | early+late only |
| `raw_buy:PSAR` AND `raw_buy:VFI` | 1.27 | 1.10 / 1.06 | 1.08 | – | no |
| `raw_buy:STC` AND `raw_buy:TRIX` | 1.25 | 1.07 / 0.94 | 1.28 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:STC` | 1.25 | 0.89 / 1.07 | 1.44 | – | early+late only |
| `raw_buy:LWBreakout` AND `raw_buy:MI` | 1.25 | 1.01 / 1.09 | 1.13 | – | no |
| `raw_buy:TRIX` AND `raw_buy:VFI` | 1.24 | 0.94 / 1.06 | 1.17 | – | no |
| `raw_buy:COPP` AND `raw_buy:STC` | 1.24 | 0.91 / 1.07 | 1.38 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:MI` | 1.22 | 0.91 / 1.09 | 1.18 | – | no |
| `raw_buy:EMV` AND `raw_buy:STC` | 1.22 | 1.09 / 1.07 | 1.23 | – | early+late only |
| `raw_buy:SAR` AND `raw_buy:TEMACross` | 1.22 | 0.97 / 1.05 | 1.13 | – | no |
| `raw_buy:MI` AND `raw_buy:STC` | 1.22 | 1.09 / 1.07 | 1.21 | – | early+late only |

### A2:672, all bars: 2775 pairs, 121 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VFI` | 1.82 | 1.26 / 1.09 | 1.57 | – | early+late only |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:KST` | 1.81 | 1.00 / 1.48 | 1.37 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:VFI` | 1.80 | 1.26 / 1.09 | 1.51 | – | early+late only |
| `raw_buy:Ehlers_ind_DYMI:trend` AND `raw_buy:KST` | 1.79 | 1.01 / 1.48 | 1.34 | – | no |
| `raw_buy:Ehlers_DC:cycle` AND `raw_buy:KST` | 1.79 | 1.01 / 1.48 | 1.38 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:STC` | 1.78 | 1.48 / 1.02 | 1.55 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.78 | 1.48 / 1.01 | 1.51 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:PSAR` | 1.76 | 1.48 / 1.02 | 1.46 | – | early+late only |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.74 | 1.05 / 1.48 | 1.65 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:VFI` | 1.73 | 1.48 / 1.09 | 1.57 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:WILLR` | 1.72 | 1.48 / 0.98 | 1.24 | – | no |
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.72 | 0.98 / 1.48 | 1.59 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.71 | 1.26 / 1.02 | 1.49 | – | early+late only |
| `raw_buy:FVE` AND `raw_buy:KST` | 1.71 | 1.00 / 1.48 | 1.53 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:MACDhist` | 1.69 | 1.26 / 1.01 | 1.55 | – | early+late only |
| `raw_buy:Ehlers_ROC:trend` AND `raw_buy:KST` | 1.69 | 1.00 / 1.48 | 1.28 | – | no |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:TRIX` | 1.68 | 1.05 / 1.40 | 1.35 | – | early+late only |
| `raw_buy:BBRevert` AND `raw_buy:TRIX` | 1.68 | 1.05 / 1.40 | 1.35 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:TEMACross` | 1.67 | 1.48 / 1.00 | 1.32 | – | no |
| `raw_buy:COPP` AND `raw_buy:Ehlers_DC:cycle` | 1.67 | 1.26 / 1.01 | 1.31 | – | no |

### A2:672, candidate bars: 2775 pairs, 48 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.55 | 1.01 / 1.27 | 1.68 | – | early+late only |
| `raw_buy:HTSINE` AND `raw_buy:TRIX` | 1.49 | 1.01 / 1.18 | 1.54 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:VFI` | 1.44 | 1.20 / 1.12 | 1.36 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:VFI` | 1.44 | 1.18 / 1.12 | 1.32 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:VFI` | 1.44 | 1.27 / 1.12 | 1.31 | – | early+late only |
| `raw_buy:KST` AND `raw_buy:ROC` | 1.40 | 1.27 / 1.20 | 1.18 | – | no |
| `raw_buy:CCI` AND `raw_buy:VFI` | 1.38 | 1.11 / 1.12 | 1.56 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:TEMACross` | 1.38 | 1.18 / 1.04 | 1.53 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:MACDhist` | 1.38 | 1.18 / 1.03 | 1.39 | – | early+late only |
| `raw_buy:VFI` AND `raw_buy:WTO` | 1.37 | 1.12 / 1.09 | 1.26 | – | no |
| `raw_buy:ROC` AND `raw_buy:VZO` | 1.36 | 1.20 / 0.99 | 1.00 | – | no |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.36 | 1.18 / 1.07 | 1.26 | – | early+late only |
| `raw_buy:COPP` AND `raw_buy:STC` | 1.35 | 1.18 / 1.02 | 1.57 | – | early+late only |
| `raw_buy:TRIX` AND `raw_buy:VFI` | 1.35 | 1.18 / 1.12 | 1.27 | – | no |
| `raw_buy:PSAR` AND `raw_buy:ROC` | 1.35 | 1.03 / 1.20 | 1.71 | – | early+late only |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.35 | 1.20 / 1.02 | 1.30 | – | early+late only |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:TRIX` | 1.33 | 1.02 / 1.18 | 1.13 | – | no |
| `raw_buy:BBRevert` AND `raw_buy:TRIX` | 1.33 | 1.02 / 1.18 | 1.13 | – | no |
| `raw_buy:COPP` AND `raw_buy:PSAR` | 1.33 | 1.18 / 1.03 | 1.30 | – | early+late only |
| `raw_buy:Ehlers_LW:trend` AND `raw_buy:ROC` | 1.32 | 1.03 / 1.20 | 1.11 | – | no |

### B-btc:288, all bars: 2775 pairs, 22 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:ROC` AND `raw_buy:TEMACross` | 1.34 | 1.07 / 1.00 | 1.58 | 1.07 | yes |
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.29 | 1.01 / 1.08 | 1.39 | 1.08 | yes |
| `raw_buy:COPP` AND `raw_buy:FVE` | 1.27 | 1.09 / 1.01 | 1.31 | 0.92 | early+late only |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.25 | 1.01 / 1.08 | 1.54 | 1.10 | yes |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.25 | 1.01 / 1.09 | 1.54 | 1.09 | yes |
| `raw_buy:COPP` AND `raw_buy:Ehlers_Fisher:trend` | 1.25 | 1.09 / 1.01 | 1.15 | 0.98 | no |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:PPO` | 1.25 | 1.01 / 1.01 | 1.13 | 1.01 | no |
| `raw_buy:PSAR` AND `raw_buy:ROC` | 1.24 | 1.01 / 1.07 | 1.34 | 0.96 | early+late only |
| `raw_buy:FVE` AND `raw_buy:ROC` | 1.24 | 1.01 / 1.07 | 1.31 | 0.90 | early+late only |
| `raw_buy:COPP` AND `raw_buy:TEMACross` | 1.23 | 1.09 / 1.00 | 1.39 | 1.05 | yes |
| `raw_buy:EhlersBase:trend` AND `raw_buy:TRIX` | 1.22 | 1.01 / 1.00 | 1.36 | 0.96 | early+late only |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:KST` | 1.22 | 1.01 / 1.08 | 1.23 | 1.08 | no |
| `raw_buy:COPP` AND `raw_buy:Ehlers_ind_DYMI:trend` | 1.22 | 1.09 / 1.00 | 1.22 | 0.98 | no |
| `raw_buy:MI` AND `raw_buy:ROC` | 1.22 | 1.01 / 1.07 | 1.27 | 1.02 | yes |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:PPO` | 1.21 | 1.01 / 1.01 | 1.15 | 0.96 | no |
| `raw_buy:Ehlers_ROC:trend` AND `raw_buy:KST` | 1.21 | 1.01 / 1.08 | 1.27 | 0.99 | early+late only |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:KST` | 1.21 | 1.00 / 1.08 | 1.33 | 0.95 | early+late only |
| `raw_buy:Ehlers_ind_DYMI:trend` AND `raw_buy:KST` | 1.21 | 1.00 / 1.08 | 1.26 | 0.97 | early+late only |
| `raw_buy:KST` AND `raw_buy:WFRACTAL` | 1.21 | 1.08 / 1.01 | 1.22 | 1.00 | no |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:KST` | 1.21 | 1.01 / 1.08 | 1.29 | 0.96 | early+late only |

### B-btc:288, candidate bars: 2775 pairs, 14 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:ROC` AND `raw_buy:TEMACross` | 1.31 | 1.06 / 1.01 | 1.40 | 1.08 | yes |
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.30 | 1.02 / 1.08 | 1.30 | 1.05 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.27 | 1.08 / 1.01 | 1.41 | 1.06 | yes |
| `raw_buy:HTSINE` AND `raw_buy:TRIX` | 1.26 | 1.02 / 1.00 | 1.21 | 1.01 | early+late only |
| `raw_buy:COPP` AND `raw_buy:TEMACross` | 1.26 | 1.08 / 1.01 | 1.26 | 1.07 | yes |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.24 | 0.99 / 1.00 | 1.35 | 1.03 | yes |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.22 | 0.99 / 1.08 | 1.41 | 1.10 | yes |
| `raw_buy:Ehlers_ind_DYMI:trend` AND `raw_buy:KST` | 1.22 | 1.02 / 1.08 | 1.26 | 0.98 | early+late only |
| `raw_buy:Ehlers_Indicator:trend` AND `raw_buy:KST` | 1.21 | 1.00 / 1.08 | 1.25 | 0.95 | early+late only |
| `raw_buy:Ehlers_ROC:trend` AND `raw_buy:KST` | 1.21 | 1.00 / 1.08 | 1.24 | 0.99 | early+late only |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:KST` | 1.21 | 1.05 / 1.08 | 1.13 | 1.00 | no |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:PPO` | 1.20 | 1.00 / 1.00 | 1.13 | 0.98 | no |
| `raw_buy:PSAR` AND `raw_buy:ROC` | 1.20 | 0.99 / 1.06 | 1.33 | 0.97 | early+late only |
| `raw_buy:COPP` AND `raw_buy:FVE` | 1.20 | 1.08 / 1.05 | 1.25 | 0.93 | early+late only |

### B-btc:48, all bars: 2775 pairs, 15 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:DCBreakout` AND `raw_buy:VORTEX` | 1.38 | 1.03 / 0.99 | 0.90 | 0.93 | no |
| `raw_buy:KST` AND `raw_buy:VBM` | 1.36 | 1.04 / 1.00 | 0.85 | 1.01 | no |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.27 | 1.03 / 1.04 | 1.29 | 1.13 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.27 | 1.04 / 1.01 | 1.12 | 1.02 | no |
| `raw_buy:Ehlers_ind_DYMI:cycle` AND `raw_buy:SMACross` | 1.27 | 1.14 / 1.00 | 1.01 | 0.97 | no |
| `raw_buy:EMACross` AND `raw_buy:Ehlers_ind_DYMI:cycle` | 1.26 | 0.99 / 1.14 | 1.04 | 1.02 | no |
| `raw_buy:Ehlers_ind_DYMI:cycle` AND `raw_buy:WFRACTAL` | 1.26 | 1.14 / 1.02 | 0.98 | 0.95 | no |
| `raw_buy:CCI` AND `raw_buy:Ehlers_KC:cycle` | 1.26 | 1.03 / 1.03 | – | 1.09 | no |
| `raw_buy:PPO` AND `raw_buy:SMACross` | 1.25 | 1.02 / 1.00 | 0.77 | 1.17 | no |
| `raw_buy:EMACross` AND `raw_buy:PVO` | 1.24 | 0.99 / 1.05 | 1.01 | 1.08 | no |
| `raw_buy:DCBreakout` AND `raw_buy:VZO` | 1.23 | 1.03 / 1.01 | 1.01 | 1.11 | no |
| `raw_buy:KST` AND `raw_buy:WFRACTAL` | 1.22 | 1.04 / 1.02 | 1.16 | 0.97 | no |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.21 | 1.03 / 1.04 | 1.37 | 1.08 | yes |
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.21 | 1.00 / 1.04 | 1.07 | 0.93 | no |
| `raw_buy:CMO` AND `raw_buy:SMACross` | 1.20 | 0.98 / 1.00 | 1.00 | 0.92 | no |

### B-btc:48, candidate bars: 2775 pairs, 6 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:VBM` | 1.29 | 1.03 / 0.96 | 0.92 | 0.96 | no |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:UO` | 1.28 | 1.02 / 1.00 | 1.02 | 1.07 | no |
| `raw_buy:HTSINE` AND `raw_buy:KST` | 1.28 | 1.04 / 1.03 | 1.23 | 0.95 | early+late only |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.23 | 1.03 / 0.99 | 1.17 | 1.01 | no |
| `raw_buy:HTSINE` AND `raw_buy:PVO` | 1.22 | 1.04 / 1.03 | 1.06 | 1.01 | no |
| `raw_buy:HTSINE` AND `raw_buy:TRIX` | 1.22 | 1.04 / 1.01 | 1.11 | 0.96 | no |

### B-btc:672, all bars: 2775 pairs, 12 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.35 | 1.13 / 1.00 | 0.92 | 1.13 | no |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 1.34 | 1.13 / 1.00 | 0.95 | 1.08 | no |
| `raw_buy:MI` AND `raw_buy:ROC` | 1.32 | 1.00 / 1.15 | 1.04 | 1.07 | no |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 1.30 | 1.00 / 1.13 | 0.92 | 1.09 | no |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.29 | 1.01 / 1.13 | 1.02 | 1.09 | no |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.29 | 1.01 / 1.13 | 1.02 | 1.09 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.27 | 1.13 / 1.00 | 0.96 | 1.06 | no |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.26 | 1.02 / 1.13 | 0.74 | 1.12 | no |
| `raw_buy:BOP` AND `raw_buy:KST` | 1.25 | 1.02 / 1.13 | 0.79 | 1.13 | no |
| `raw_buy:KST` AND `raw_buy:WFRACTAL` | 1.25 | 1.13 / 1.01 | 0.85 | 1.10 | no |
| `raw_buy:Ehlers_DYMI:trend` AND `raw_buy:KST` | 1.24 | 0.99 / 1.13 | 1.03 | 1.03 | no |
| `raw_buy:LWRevert` AND `raw_buy:TRIX` | 1.23 | 1.00 / 1.05 | 0.93 | 1.07 | no |

### B-btc:672, candidate bars: 2775 pairs, 3 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:VFI` AND `raw_buy:WTO` | 1.24 | 1.10 / 1.01 | 1.06 | 1.00 | no |
| `raw_buy:VBM` AND `raw_buy:VFI` | 1.23 | 1.02 / 1.10 | 0.83 | 1.16 | no |
| `raw_buy:LWBreakout` AND `raw_buy:TEMACross` | 1.22 | 1.01 / 1.02 | 1.00 | 1.02 | no |

### B-pair:288, all bars: 2775 pairs, 44 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.45 | 1.06 / 1.12 | 1.60 | 1.09 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.41 | 1.16 / 1.02 | 1.33 | 1.06 | early+late only |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.39 | 1.06 / 1.16 | 1.55 | 1.08 | yes |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.38 | 1.01 / 1.16 | 1.45 | 1.07 | yes |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:PPO` | 1.37 | 0.99 / 1.07 | 1.30 | 1.03 | early+late only |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.36 | 1.16 / 1.05 | 1.53 | 1.12 | yes |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.36 | 1.06 / 1.08 | 1.44 | 1.06 | early+late only |
| `raw_buy:KST` AND `raw_buy:WFRACTAL` | 1.36 | 1.16 / 1.01 | 1.37 | 1.06 | early+late only |
| `raw_buy:CCI` AND `raw_buy:PPO` | 1.32 | 1.06 / 1.07 | 1.44 | 1.09 | yes |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.32 | 1.02 / 1.16 | 1.38 | 1.09 | yes |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.32 | 1.02 / 1.16 | 1.38 | 1.09 | yes |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 1.31 | 1.12 / 1.02 | 1.30 | 1.11 | yes |
| `raw_buy:COPP` AND `raw_buy:ZScore` | 1.31 | 1.12 / 1.05 | 1.51 | 1.11 | yes |
| `raw_buy:COPP` AND `raw_buy:STC` | 1.30 | 1.12 / 1.00 | 1.31 | 1.08 | early+late only |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.29 | 1.08 / 1.02 | 1.16 | 1.09 | no |
| `raw_buy:COPP` AND `raw_buy:DCRevert` | 1.29 | 1.12 / 1.03 | 1.19 | 1.16 | no |
| `raw_buy:ROC` AND `raw_buy:UO` | 1.28 | 1.10 / 1.01 | 1.26 | 1.21 | yes |
| `raw_buy:COPP` AND `raw_buy:Ehlers_ind_DYMI:trend` | 1.28 | 1.12 / 0.99 | 1.27 | 1.08 | early+late only |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.26 | 1.08 / 1.05 | 1.42 | 1.12 | yes |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.25 | 1.10 / 1.00 | 1.39 | 1.04 | early+late only |

### B-pair:288, candidate bars: 2775 pairs, 15 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:WTO` | 1.36 | 1.08 / 1.06 | 1.19 | 1.05 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.32 | 1.08 / 1.04 | 1.31 | 1.10 | yes |
| `raw_buy:CMO` AND `raw_buy:FVE` | 1.30 | 1.04 / 1.00 | 1.12 | 1.05 | no |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.30 | 1.08 / 1.06 | 1.34 | 1.05 | yes |
| `raw_buy:CCI` AND `raw_buy:WTO` | 1.29 | 1.08 / 1.06 | 1.09 | 1.03 | no |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.27 | 1.03 / 1.08 | 1.26 | 1.04 | yes |
| `raw_buy:VFI` AND `raw_buy:WTO` | 1.22 | 1.07 / 1.06 | 1.23 | 1.02 | yes |
| `raw_buy:FVE` AND `raw_buy:ZScore` | 1.22 | 1.00 / 1.04 | 1.11 | 1.09 | no |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.21 | 1.03 / 1.06 | 1.15 | 1.06 | no |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.21 | 0.98 / 1.08 | 1.16 | 1.15 | no |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.21 | 1.03 / 1.04 | 1.25 | 1.08 | yes |
| `raw_buy:KST` AND `raw_buy:UO` | 1.21 | 1.08 / 1.05 | 1.26 | 1.28 | yes |
| `raw_buy:FVE` AND `raw_buy:KST` | 1.20 | 1.00 / 1.08 | 1.18 | 1.02 | no |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.20 | 1.00 / 1.08 | 1.17 | 1.04 | no |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.20 | 1.00 / 1.08 | 1.17 | 1.04 | no |

### B-pair:48, all bars: 2775 pairs, 44 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.51 | 1.21 / 1.16 | 1.28 | 1.18 | yes |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.50 | 1.15 / 1.23 | 1.59 | 1.22 | yes |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.44 | 1.15 / 1.21 | 1.35 | 1.27 | yes |
| `raw_buy:CMO` AND `raw_buy:PVO` | 1.44 | 1.17 / 1.01 | 1.24 | 1.34 | yes |
| `raw_buy:COPP` AND `raw_buy:WTO` | 1.41 | 1.23 / 1.02 | 1.25 | 1.08 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.38 | 1.21 / 1.02 | 1.18 | 1.17 | no |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.37 | 1.25 / 0.98 | 1.10 | 1.04 | no |
| `raw_buy:CCI` AND `raw_buy:TRIX` | 1.37 | 1.15 / 1.19 | 1.25 | 1.22 | yes |
| `raw_buy:Ehlers_Fisher:trend` AND `raw_buy:PPO` | 1.37 | 0.95 / 1.16 | 1.02 | 1.14 | no |
| `raw_buy:KST` AND `raw_buy:VFI` | 1.36 | 1.21 / 1.07 | 1.05 | 1.09 | no |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 1.36 | 1.23 / 1.08 | 1.22 | 1.17 | yes |
| `raw_buy:DCBreakout` AND `raw_buy:VORTEX` | 1.36 | 0.89 / 1.07 | 1.02 | 1.51 | no |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.36 | 1.19 / 1.16 | 1.25 | 1.17 | yes |
| `raw_buy:COPP` AND `raw_buy:UO` | 1.35 | 1.23 / 1.10 | 1.31 | 1.17 | yes |
| `raw_buy:CCI` AND `raw_buy:PPO` | 1.35 | 1.15 / 1.16 | 1.30 | 1.17 | yes |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.35 | 1.21 / 1.08 | 1.20 | 1.19 | yes |
| `raw_buy:Ehlers_Indicator:cycle` AND `raw_buy:VORTEX` | 1.33 | 1.03 / 1.07 | 1.18 | 1.10 | no |
| `raw_buy:Ehlers_ind_DYMI:cycle` AND `raw_buy:KCRevert` | 1.33 | 1.10 / 1.11 | 1.17 | 0.89 | no |
| `raw_buy:CMO` AND `raw_buy:TEMACross` | 1.33 | 1.17 / 0.97 | 1.38 | 1.38 | yes |
| `raw_buy:PPO` AND `raw_buy:ZScore` | 1.32 | 1.16 / 1.16 | 1.30 | 1.09 | yes |

### B-pair:48, candidate bars: 2775 pairs, 23 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.36 | 1.18 / 1.11 | 1.46 | 1.16 | yes |
| `raw_buy:CMO` AND `raw_buy:FVE` | 1.36 | 1.11 / 0.96 | 1.12 | 0.95 | no |
| `raw_buy:CCI` AND `raw_buy:WTO` | 1.36 | 1.18 / 1.10 | 1.14 | 1.20 | no |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.33 | 1.18 / 1.09 | 1.23 | 1.23 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.32 | 1.09 / 1.13 | 1.18 | 1.13 | no |
| `raw_buy:HTSINE` AND `raw_buy:ZScore` | 1.30 | 0.97 / 1.13 | 1.19 | 1.08 | no |
| `raw_buy:KST` AND `raw_buy:VBM` | 1.28 | 1.09 / 0.91 | 1.04 | 1.05 | no |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.28 | 1.07 / 1.09 | 1.07 | 1.07 | no |
| `raw_buy:WTO` AND `raw_buy:ZScore` | 1.28 | 1.10 / 1.13 | 1.14 | 1.12 | no |
| `raw_buy:PSAR` AND `raw_buy:UO` | 1.27 | 0.93 / 1.11 | 0.94 | 1.12 | no |
| `raw_buy:FVE` AND `raw_buy:ZScore` | 1.25 | 0.96 / 1.13 | 1.20 | 1.06 | yes |
| `raw_buy:EWO` AND `raw_buy:PVO` | 1.25 | 1.07 / 1.04 | 1.09 | 1.15 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.25 | 1.09 / 1.10 | 1.12 | 1.12 | no |
| `raw_buy:Ehlers_DYMI:trend` AND `raw_buy:ROC` | 1.25 | 0.96 / 1.12 | 0.92 | 1.01 | no |
| `raw_buy:ROC` AND `raw_buy:STC` | 1.24 | 1.12 / 0.98 | 1.03 | 0.99 | no |
| `raw_buy:FVE` AND `raw_buy:ROC` | 1.24 | 0.96 / 1.12 | 1.13 | 0.99 | no |
| `raw_buy:COPP` AND `raw_buy:WTO` | 1.24 | 1.11 / 1.10 | 1.16 | 1.03 | no |
| `raw_buy:ROC` AND `raw_buy:VFI` | 1.23 | 1.12 / 1.12 | 1.09 | 0.96 | no |
| `raw_buy:FVE` AND `raw_buy:TSI` | 1.22 | 0.96 / 1.08 | 1.09 | 1.02 | no |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.22 | 1.04 / 1.09 | 1.10 | 1.08 | no |

### B-pair:672, all bars: 2775 pairs, 43 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VBM` | 1.45 | 1.15 / 0.98 | 1.15 | 1.33 | no |
| `raw_buy:BOP` AND `raw_buy:KST` | 1.44 | 1.04 / 1.15 | 1.16 | 1.13 | no |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.43 | 1.07 / 1.15 | 1.09 | 1.13 | no |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.37 | 1.15 / 1.02 | 1.12 | 1.12 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.36 | 1.15 / 1.03 | 1.13 | 1.16 | no |
| `raw_buy:KST` AND `raw_buy:VBM` | 1.34 | 1.15 / 0.98 | 1.21 | 1.32 | yes |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.34 | 1.03 / 1.15 | 1.31 | 1.11 | yes |
| `raw_buy:KST` AND `raw_buy:UO` | 1.34 | 1.15 / 1.01 | 1.16 | 1.39 | no |
| `raw_buy:KST` AND `raw_buy:MI` | 1.33 | 1.15 / 0.99 | 1.08 | 1.16 | no |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.33 | 1.02 / 1.15 | 0.96 | 1.09 | no |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.33 | 1.02 / 1.15 | 0.96 | 1.09 | no |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 1.32 | 1.15 / 1.02 | 1.06 | 1.13 | no |
| `raw_buy:DCRevert` AND `raw_buy:KST` | 1.32 | 1.02 / 1.15 | 1.12 | 1.20 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.31 | 1.15 / 1.01 | 1.07 | 1.04 | no |
| `raw_buy:KST` AND `raw_buy:WILLR` | 1.31 | 1.15 / 1.02 | 1.05 | 1.13 | no |
| `raw_buy:CCI` AND `raw_buy:PPO` | 1.30 | 1.03 / 1.12 | 1.29 | 1.10 | yes |
| `raw_buy:COPP` AND `raw_buy:MI` | 1.30 | 1.15 / 0.99 | 1.14 | 1.19 | no |
| `raw_buy:COPP` AND `raw_buy:DCRevert` | 1.30 | 1.15 / 1.02 | 1.13 | 1.23 | no |
| `raw_buy:TRIX` AND `raw_buy:UO` | 1.30 | 1.13 / 1.01 | 1.06 | 1.36 | no |
| `raw_buy:MI` AND `raw_buy:ROC` | 1.29 | 0.99 / 1.14 | 1.11 | 1.10 | no |

### B-pair:672, candidate bars: 2775 pairs, 15 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VBM` | 1.31 | 1.07 / 1.01 | 1.05 | 1.28 | no |
| `raw_buy:COPP` AND `raw_buy:UO` | 1.25 | 1.07 / 1.06 | 1.13 | 1.34 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.25 | 1.08 / 1.05 | 1.13 | 1.14 | no |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.25 | 1.01 / 1.08 | 1.04 | 1.06 | no |
| `raw_buy:CCI` AND `raw_buy:WTO` | 1.24 | 1.08 / 1.08 | 1.07 | 1.14 | no |
| `raw_buy:KST` AND `raw_buy:UO` | 1.24 | 1.08 / 1.06 | 1.05 | 1.36 | no |
| `raw_buy:VFI` AND `raw_buy:WTO` | 1.24 | 1.10 / 1.08 | 1.13 | 0.94 | no |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.23 | 1.08 / 1.03 | 1.05 | 1.09 | no |
| `raw_buy:KST` AND `raw_buy:MI` | 1.23 | 1.08 / 1.01 | 1.09 | 1.12 | no |
| `raw_buy:VFI` AND `raw_buy:WFRACTAL` | 1.22 | 1.10 / 1.04 | 1.04 | 1.01 | no |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.22 | 1.05 / 1.05 | 1.10 | 1.10 | no |
| `raw_buy:DCRevert` AND `raw_buy:KST` | 1.22 | 1.04 / 1.08 | 1.02 | 1.16 | no |
| `raw_buy:CMO` AND `raw_buy:FVE` | 1.22 | 1.04 / 0.99 | 1.00 | 0.97 | no |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 1.21 | 1.07 / 1.03 | 1.04 | 1.09 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.20 | 1.08 / 1.08 | 1.09 | 1.02 | no |

### C-btc:288, all bars: 2775 pairs, 81 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 2.47 | 1.61 / 1.09 | 1.45 | 1.74 | yes |
| `raw_buy:COPP` AND `raw_buy:ZScore` | 2.46 | 1.61 / 1.06 | 1.41 | 1.73 | yes |
| `raw_buy:BOP` AND `raw_buy:COPP` | 2.45 | 1.44 / 1.61 | 1.32 | 1.64 | no |
| `raw_buy:COPP` AND `raw_buy:HTTREND` | 2.43 | 1.61 / 1.52 | 1.30 | 1.64 | no |
| `raw_buy:PPO` AND `raw_buy:ZScore` | 2.40 | 1.46 / 1.06 | 1.37 | 1.64 | yes |
| `raw_buy:HTTREND` AND `raw_buy:PPO` | 2.27 | 1.52 / 1.46 | 1.28 | 1.57 | no |
| `raw_buy:BOP` AND `raw_buy:PPO` | 2.26 | 1.44 / 1.46 | 1.30 | 1.58 | yes |
| `raw_buy:COPP` AND `raw_buy:Ehlers_LW:cycle` | 2.21 | 1.61 / 1.12 | 1.43 | 1.70 | yes |
| `raw_buy:CCI` AND `raw_buy:KST` | 2.06 | 0.93 / 1.63 | 1.36 | 1.75 | no |
| `raw_buy:COPP` AND `raw_buy:MACDsignal` | 2.02 | 1.61 / 1.32 | 1.32 | 1.74 | no |
| `raw_buy:CCI` AND `raw_buy:PPO` | 2.01 | 0.93 / 1.46 | 2.13 | 1.70 | yes |
| `raw_buy:HTTREND` AND `raw_buy:ZScore` | 1.98 | 1.52 / 1.06 | 1.37 | 1.47 | yes |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.96 | 0.93 / 1.61 | 1.53 | 1.80 | yes |
| `raw_buy:HTTREND` AND `raw_buy:LWRevert` | 1.92 | 1.52 / 1.09 | 1.33 | 1.43 | yes |
| `raw_buy:COPP` AND `raw_buy:Ehlers_Fisher:trend` | 1.89 | 1.61 / 1.04 | 1.43 | 1.79 | yes |
| `raw_buy:KST` AND `raw_buy:WFRACTAL` | 1.84 | 1.63 / 0.86 | 1.25 | 1.72 | no |
| `raw_buy:HTTREND` AND `raw_buy:TSI` | 1.83 | 1.52 / 0.88 | 1.33 | 1.46 | yes |
| `raw_buy:PPO` AND `raw_buy:SMACross` | 1.82 | 1.46 / 1.03 | 1.06 | 1.63 | no |
| `raw_buy:HTTREND` AND `raw_buy:VFI` | 1.81 | 1.52 / 1.17 | 1.79 | 1.54 | yes |
| `raw_buy:BOP` AND `raw_buy:TSI` | 1.80 | 1.44 / 0.88 | 1.46 | 1.46 | yes |

### C-btc:288, candidate bars: 2775 pairs, 29 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:WTO` | 1.47 | 1.19 / 1.08 | 1.16 | 1.21 | no |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.41 | 0.98 / 1.19 | 1.17 | 1.19 | no |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.41 | 0.98 / 1.19 | 1.17 | 1.19 | no |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.41 | 1.13 / 1.19 | 1.47 | 1.23 | yes |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 1.38 | 1.16 / 1.14 | 1.18 | 1.18 | no |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 1.37 | 1.17 / 1.14 | 1.17 | 1.20 | no |
| `raw_buy:COPP` AND `raw_buy:ZScore` | 1.36 | 1.17 / 1.14 | 1.16 | 1.20 | no |
| `raw_buy:BOP` AND `raw_buy:COPP` | 1.36 | 1.15 / 1.17 | 1.11 | 1.14 | no |
| `raw_buy:EMV` AND `raw_buy:EhlersBase:trend` | 1.36 | 1.16 / 1.01 | 1.14 | 1.13 | no |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 1.35 | 1.10 / 1.19 | 1.20 | 1.20 | yes |
| `raw_buy:COPP` AND `raw_buy:HTTREND` | 1.35 | 1.17 / 1.17 | 1.11 | 1.13 | no |
| `raw_buy:EMV` AND `raw_buy:LWRevert` | 1.34 | 1.16 / 1.14 | 1.30 | 1.13 | yes |
| `raw_buy:PPO` AND `raw_buy:ZScore` | 1.34 | 1.03 / 1.14 | 1.31 | 1.14 | yes |
| `raw_buy:COPP` AND `raw_buy:Ehlers_LW:cycle` | 1.33 | 1.17 / 1.10 | 1.15 | 1.18 | no |
| `raw_buy:Ehlers_DYMI:trend` AND `raw_buy:KST` | 1.33 | 1.01 / 1.19 | 1.15 | 1.14 | no |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.31 | 1.16 / 1.08 | 1.15 | 1.18 | no |
| `raw_buy:BOP` AND `raw_buy:TRIX` | 1.31 | 1.15 / 1.16 | 1.11 | 1.12 | no |
| `raw_buy:LWRevert` AND `raw_buy:TRIX` | 1.30 | 1.14 / 1.16 | 1.21 | 1.17 | yes |
| `raw_buy:HTTREND` AND `raw_buy:TRIX` | 1.30 | 1.17 / 1.16 | 1.10 | 1.11 | no |
| `raw_buy:CMO` AND `raw_buy:EMV` | 1.30 | 1.01 / 1.16 | 1.23 | 1.06 | early+late only |

### C-btc:48, all bars: 2775 pairs, 132 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:ZScore` | 2.26 | 1.77 / 1.26 | 1.72 | 1.79 | yes |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 2.12 | 1.12 / 1.77 | 2.24 | 1.73 | yes |
| `raw_buy:COPP` AND `raw_buy:ZScore` | 2.12 | 1.85 / 1.26 | 1.65 | 1.78 | yes |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 2.10 | 1.16 / 1.77 | 2.04 | 1.71 | yes |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 2.10 | 1.16 / 1.77 | 2.04 | 1.71 | yes |
| `raw_buy:PPO` AND `raw_buy:ZScore` | 2.09 | 1.54 / 1.26 | 2.03 | 1.69 | yes |
| `raw_buy:COPP` AND `raw_buy:LWRevert` | 2.08 | 1.85 / 1.22 | 2.07 | 1.77 | yes |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 2.05 | 1.57 / 1.26 | 2.22 | 1.69 | yes |
| `raw_buy:BOP` AND `raw_buy:COPP` | 2.05 | 1.31 / 1.85 | 2.68 | 1.75 | yes |
| `raw_buy:COPP` AND `raw_buy:HTTREND` | 2.05 | 1.85 / 1.57 | 2.67 | 1.74 | yes |
| `raw_buy:ROC` AND `raw_buy:STC` | 2.01 | 1.79 / 1.00 | 1.92 | 1.79 | yes |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 2.00 | 1.18 / 1.77 | 1.63 | 1.72 | yes |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.99 | 1.77 / 1.22 | 1.65 | 1.76 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.96 | 1.77 / 1.02 | 1.87 | 1.73 | yes |
| `raw_buy:EMV` AND `raw_buy:UO` | 1.91 | 1.22 / 0.80 | 1.13 | 1.59 | no |
| `raw_buy:BOP` AND `raw_buy:PPO` | 1.89 | 1.31 / 1.54 | 1.58 | 1.68 | yes |
| `raw_buy:HTTREND` AND `raw_buy:PPO` | 1.89 | 1.57 / 1.54 | 1.56 | 1.68 | yes |
| `raw_buy:BOP` AND `raw_buy:TRIX` | 1.88 | 1.31 / 1.57 | 2.20 | 1.69 | yes |
| `raw_buy:HTTREND` AND `raw_buy:TRIX` | 1.88 | 1.57 / 1.57 | 2.13 | 1.69 | yes |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:TRIX` | 1.87 | 1.12 / 1.57 | 1.86 | 1.63 | yes |

### C-btc:48, candidate bars: 2775 pairs, 37 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 1.58 | 1.09 / 1.16 | 1.35 | 1.19 | yes |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:KST` | 1.58 | 1.10 / 1.16 | 1.33 | 1.17 | yes |
| `raw_buy:BBRevert` AND `raw_buy:KST` | 1.58 | 1.10 / 1.16 | 1.33 | 1.17 | yes |
| `raw_buy:CCI` AND `raw_buy:LWRevert` | 1.38 | 1.24 / 1.18 | 1.47 | 1.09 | yes |
| `raw_buy:COPP` AND `raw_buy:HTTREND` | 1.35 | 1.22 / 1.23 | 1.51 | 1.18 | yes |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 1.35 | 1.23 / 1.16 | 1.24 | 1.18 | yes |
| `raw_buy:BBBreakout:tm0` AND `raw_buy:TRIX` | 1.33 | 1.10 / 1.12 | 1.23 | 1.12 | yes |
| `raw_buy:BBRevert` AND `raw_buy:TRIX` | 1.33 | 1.10 / 1.12 | 1.23 | 1.12 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.32 | 1.16 / 1.07 | 1.28 | 1.20 | yes |
| `raw_buy:LWRevert` AND `raw_buy:VZO` | 1.32 | 1.18 / 1.04 | 1.07 | 1.04 | no |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.30 | 1.16 / 1.18 | 1.22 | 1.20 | yes |
| `raw_buy:CMO` AND `raw_buy:EMV` | 1.27 | 1.02 / 1.11 | 1.08 | 1.12 | no |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:TRIX` | 1.27 | 1.09 / 1.12 | 1.23 | 1.14 | yes |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:Ehlers_LW:trend` | 1.27 | 1.09 / 1.07 | 1.12 | 1.13 | no |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:PPO` | 1.26 | 1.09 / 1.05 | 1.14 | 1.10 | no |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:Ehlers_LW:trend` | 1.26 | 1.12 / 1.07 | 1.23 | 1.08 | yes |
| `raw_buy:VBM` AND `raw_buy:VFI` | 1.25 | 1.08 / 1.10 | 1.08 | 1.11 | no |
| `raw_buy:TRIX` AND `raw_buy:WTO` | 1.25 | 1.12 / 1.07 | 1.16 | 1.17 | no |
| `raw_buy:EMV` AND `raw_buy:Ehlers_BB:cycle` | 1.24 | 1.11 / 1.09 | 1.22 | 1.12 | yes |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:VZO` | 1.24 | 1.12 / 1.04 | 0.99 | 0.98 | no |

### C-btc:672, all bars: 2775 pairs, 0 pass the rule

No survivors.

### C-btc:672, candidate bars: 2775 pairs, 0 pass the rule

No survivors.

### C-pair:288, all bars: 2775 pairs, 140 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VBM` | 2.37 | 1.90 / 0.94 | 1.96 | 1.98 | yes |
| `raw_buy:KST` AND `raw_buy:VBM` | 2.34 | 1.74 / 0.94 | 2.29 | 1.86 | yes |
| `raw_buy:FVE` AND `raw_buy:ROC` | 2.32 | 0.97 / 1.74 | 1.73 | 1.86 | yes |
| `raw_buy:EMACross` AND `raw_buy:PPO` | 2.31 | 1.02 / 1.63 | 2.05 | 1.85 | yes |
| `raw_buy:COPP` AND `raw_buy:WTO` | 2.28 | 1.90 / 1.02 | 1.60 | 1.87 | no |
| `raw_buy:COPP` AND `raw_buy:MI` | 2.27 | 1.90 / 1.00 | 2.27 | 1.85 | yes |
| `raw_buy:DCRevert` AND `raw_buy:KST` | 2.25 | 1.07 / 1.74 | 2.09 | 1.56 | early+late only |
| `raw_buy:COPP` AND `raw_buy:FVE` | 2.24 | 1.90 / 0.97 | 2.10 | 1.86 | yes |
| `raw_buy:MI` AND `raw_buy:ROC` | 2.23 | 1.00 / 1.74 | 1.91 | 1.87 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 2.21 | 1.74 / 0.97 | 1.60 | 1.86 | no |
| `raw_buy:KST` AND `raw_buy:MI` | 2.19 | 1.74 / 1.00 | 2.21 | 1.76 | yes |
| `raw_buy:PPO` AND `raw_buy:SMACross` | 2.18 | 1.63 / 1.00 | 1.78 | 1.89 | yes |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:PPO` | 2.16 | 0.99 / 1.63 | 2.23 | 1.86 | yes |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 2.15 | 1.74 / 0.97 | 1.87 | 1.83 | yes |
| `raw_buy:COPP` AND `raw_buy:UO` | 2.13 | 1.90 / 1.08 | 2.18 | 1.45 | early+late only |
| `raw_buy:COPP` AND `raw_buy:DCRevert` | 2.13 | 1.90 / 1.07 | 2.06 | 1.60 | early+late only |
| `raw_buy:COPP` AND `raw_buy:STC` | 2.13 | 1.90 / 1.00 | 1.85 | 1.89 | yes |
| `raw_buy:KST` AND `raw_buy:UO` | 2.13 | 1.74 / 1.08 | 1.86 | 1.36 | early+late only |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 2.12 | 0.95 / 1.74 | 1.63 | 1.83 | no |
| `raw_buy:ROC` AND `raw_buy:STC` | 2.11 | 1.74 / 1.00 | 1.66 | 1.87 | yes |

### C-pair:288, candidate bars: 2775 pairs, 27 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:VBM` | 1.29 | 1.11 / 1.17 | 1.29 | 1.22 | yes |
| `raw_buy:Ichimoku` AND `raw_buy:MI` | 1.29 | 1.14 / 1.16 | 1.21 | 1.19 | yes |
| `raw_buy:EMACross` AND `raw_buy:WTO` | 1.29 | 1.14 / 1.14 | 1.22 | 1.17 | yes |
| `raw_buy:EhlersBase:cycle` AND `raw_buy:MI` | 1.28 | 1.13 / 1.16 | 1.23 | 1.17 | yes |
| `raw_buy:SMACross` AND `raw_buy:WTO` | 1.28 | 1.15 / 1.14 | 1.19 | 1.16 | no |
| `raw_buy:EMACross` AND `raw_buy:UO` | 1.28 | 1.14 / 1.05 | 1.15 | 1.01 | no |
| `raw_buy:AO` AND `raw_buy:Ichimoku` | 1.28 | 0.98 / 1.14 | 1.04 | 1.15 | no |
| `raw_buy:Aroon` AND `raw_buy:Ichimoku` | 1.28 | 1.08 / 1.14 | 1.05 | 1.12 | no |
| `raw_buy:EMACross` AND `raw_buy:MFI` | 1.27 | 1.14 / 0.86 | 1.12 | 1.07 | no |
| `raw_buy:EMACross` AND `raw_buy:PPO` | 1.26 | 1.14 / 0.91 | 1.22 | 1.18 | yes |
| `raw_buy:COPP` AND `raw_buy:WTO` | 1.26 | 1.00 / 1.14 | 1.10 | 1.18 | no |
| `raw_buy:HTSINE` AND `raw_buy:MFI` | 1.26 | 1.14 / 0.86 | 1.10 | 0.89 | no |
| `raw_buy:EMACross` AND `raw_buy:TEMACross` | 1.26 | 1.14 / 0.98 | 1.02 | 1.07 | no |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.25 | 1.00 / 1.00 | 1.10 | 1.19 | no |
| `raw_buy:FVE` AND `raw_buy:UO` | 1.25 | 1.11 / 1.05 | 1.07 | 0.77 | no |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:UO` | 1.25 | 1.11 / 1.05 | 1.16 | 0.90 | no |
| `raw_buy:STC` AND `raw_buy:UO` | 1.25 | 1.08 / 1.05 | 1.11 | 0.86 | no |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:MACDhist` | 1.24 | 1.11 / 1.00 | 1.09 | 1.13 | no |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:MFI` | 1.24 | 1.11 / 0.86 | 1.09 | 0.98 | no |
| `raw_buy:AO` AND `raw_buy:KCBreakout` | 1.23 | 0.98 / 1.11 | 1.06 | 1.02 | no |

### C-pair:48, all bars: 2775 pairs, 111 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VBM` | 2.41 | 2.05 / 0.94 | 1.99 | 1.97 | yes |
| `raw_buy:KST` AND `raw_buy:VBM` | 2.39 | 1.98 / 0.94 | 1.96 | 1.89 | yes |
| `raw_buy:COPP` AND `raw_buy:WTO` | 2.34 | 2.05 / 1.03 | 1.84 | 1.91 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 2.33 | 1.98 / 1.07 | 2.06 | 1.88 | yes |
| `raw_buy:FVE` AND `raw_buy:ROC` | 2.30 | 0.97 / 1.97 | 1.81 | 1.91 | yes |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 2.30 | 1.98 / 1.07 | 2.00 | 1.88 | yes |
| `raw_buy:COPP` AND `raw_buy:MI` | 2.28 | 2.05 / 0.97 | 2.01 | 1.87 | yes |
| `raw_buy:BOP` AND `raw_buy:KST` | 2.26 | 1.21 / 1.98 | 1.68 | 1.87 | no |
| `raw_buy:HTTREND` AND `raw_buy:KST` | 2.25 | 1.41 / 1.98 | 1.68 | 1.87 | no |
| `raw_buy:TRIX` AND `raw_buy:VBM` | 2.24 | 1.87 / 0.94 | 1.88 | 1.81 | yes |
| `raw_buy:Ehlers_BB:cycle` AND `raw_buy:KST` | 2.24 | 1.00 / 1.98 | 1.73 | 1.87 | yes |
| `raw_buy:PPO` AND `raw_buy:SMACross` | 2.23 | 1.81 / 1.00 | 1.75 | 1.91 | yes |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:PPO` | 2.23 | 1.03 / 1.81 | 1.74 | 1.86 | yes |
| `raw_buy:ROC` AND `raw_buy:STC` | 2.22 | 1.97 / 1.00 | 1.93 | 1.92 | yes |
| `raw_buy:Ehlers_LW:cycle` AND `raw_buy:KST` | 2.21 | 1.00 / 1.98 | 1.82 | 1.87 | yes |
| `raw_buy:MI` AND `raw_buy:ROC` | 2.21 | 0.97 / 1.97 | 2.04 | 1.90 | yes |
| `raw_buy:EMACross` AND `raw_buy:PPO` | 2.20 | 1.01 / 1.81 | 2.03 | 1.88 | yes |
| `raw_buy:CCI` AND `raw_buy:KST` | 2.19 | 1.07 / 1.98 | 1.83 | 1.80 | yes |
| `raw_buy:KST` AND `raw_buy:WTO` | 2.19 | 1.98 / 1.03 | 1.81 | 1.86 | yes |
| `raw_buy:TRIX` AND `raw_buy:ZScore` | 2.17 | 1.87 / 1.07 | 1.93 | 1.81 | yes |

### C-pair:48, candidate bars: 2775 pairs, 3 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:VBM` | 1.27 | 1.04 / 1.15 | 1.23 | 1.18 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.22 | 1.04 / 1.06 | 1.25 | 1.16 | yes |
| `raw_buy:AO` AND `raw_buy:Ichimoku` | 1.22 | 0.98 / 1.07 | 0.99 | 1.11 | no |

### C-pair:672, all bars: 2775 pairs, 177 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VBM` | 2.38 | 1.87 / 0.96 | 2.34 | 1.98 | yes |
| `raw_buy:EMACross` AND `raw_buy:PPO` | 2.28 | 1.01 / 1.61 | 2.49 | 1.85 | yes |
| `raw_buy:COPP` AND `raw_buy:MI` | 2.26 | 1.87 / 1.02 | 2.39 | 1.84 | yes |
| `raw_buy:COPP` AND `raw_buy:UO` | 2.25 | 1.87 / 1.28 | 2.45 | 1.39 | early+late only |
| `raw_buy:COPP` AND `raw_buy:WTO` | 2.22 | 1.87 / 1.00 | 1.78 | 1.80 | no |
| `raw_buy:KST` AND `raw_buy:VBM` | 2.22 | 1.63 / 0.96 | 2.46 | 1.91 | yes |
| `raw_buy:COPP` AND `raw_buy:WFRACTAL` | 2.20 | 1.87 / 1.14 | 2.20 | 1.41 | early+late only |
| `raw_buy:COPP` AND `raw_buy:DCRevert` | 2.18 | 1.87 / 1.22 | 2.34 | 1.57 | early+late only |
| `raw_buy:PPO` AND `raw_buy:SMACross` | 2.18 | 1.61 / 0.98 | 1.43 | 1.87 | no |
| `raw_buy:ROC` AND `raw_buy:UO` | 2.17 | 1.69 / 1.28 | 2.32 | 1.45 | early+late only |
| `raw_buy:KST` AND `raw_buy:MI` | 2.16 | 1.63 / 1.02 | 2.40 | 1.75 | yes |
| `raw_buy:TRIX` AND `raw_buy:UO` | 2.16 | 1.57 / 1.28 | 2.13 | 1.19 | early+late only |
| `raw_buy:DCRevert` AND `raw_buy:KST` | 2.15 | 1.22 / 1.63 | 2.36 | 1.56 | early+late only |
| `raw_buy:FVE` AND `raw_buy:ROC` | 2.15 | 0.98 / 1.69 | 1.88 | 1.86 | yes |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:PPO` | 2.15 | 1.00 / 1.61 | 2.28 | 1.85 | yes |
| `raw_buy:MI` AND `raw_buy:ROC` | 2.14 | 1.02 / 1.69 | 1.96 | 1.87 | yes |
| `raw_buy:COPP` AND `raw_buy:FVE` | 2.12 | 1.87 / 0.98 | 2.33 | 1.85 | yes |
| `raw_buy:KCBreakout` AND `raw_buy:TRIX` | 2.09 | 0.98 / 1.57 | 2.45 | 1.62 | yes |
| `raw_buy:TRIX` AND `raw_buy:VBM` | 2.09 | 1.57 / 0.96 | 2.29 | 1.76 | yes |
| `raw_buy:DCRevert` AND `raw_buy:TRIX` | 2.08 | 1.22 / 1.57 | 2.29 | 1.41 | early+late only |

### C-pair:672, candidate bars: 2775 pairs, 43 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:COPP` AND `raw_buy:VBM` | 1.33 | 1.01 / 1.19 | 1.23 | 1.28 | yes |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:VBM` | 1.32 | 1.11 / 1.19 | 1.16 | 1.25 | no |
| `raw_buy:Ichimoku` AND `raw_buy:MI` | 1.31 | 1.13 / 1.16 | 1.14 | 1.22 | no |
| `raw_buy:EMACross` AND `raw_buy:PPO` | 1.29 | 1.14 / 0.92 | 1.44 | 1.21 | yes |
| `raw_buy:APO` AND `raw_buy:KCBreakout` | 1.29 | 0.98 / 1.14 | 1.06 | 0.96 | no |
| `raw_buy:Aroon` AND `raw_buy:MI` | 1.29 | 1.10 / 1.16 | 1.27 | 1.14 | yes |
| `raw_buy:PSAR` AND `raw_buy:UO` | 1.29 | 1.06 / 1.07 | 1.18 | 0.87 | no |
| `raw_buy:EMACross` AND `raw_buy:MFI` | 1.28 | 1.14 / 0.89 | 1.21 | 1.05 | early+late only |
| `raw_buy:AO` AND `raw_buy:KCBreakout` | 1.28 | 0.99 / 1.14 | 1.04 | 0.98 | no |
| `raw_buy:EMACross` AND `raw_buy:WTO` | 1.28 | 1.14 / 1.13 | 1.13 | 1.20 | no |
| `raw_buy:Aroon` AND `raw_buy:Ichimoku` | 1.28 | 1.10 / 1.13 | 1.09 | 1.17 | no |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:MFI` | 1.28 | 1.11 / 0.89 | 1.13 | 0.95 | no |
| `raw_buy:Ehlers_Ichimoku:trend` AND `raw_buy:UO` | 1.28 | 1.11 / 1.07 | 1.17 | 0.80 | no |
| `raw_buy:Ehlers_DC:cycle` AND `raw_buy:PSAR` | 1.28 | 1.00 / 1.06 | 1.15 | 0.90 | no |
| `raw_buy:KCBreakout` AND `raw_buy:MACDhist` | 1.28 | 1.14 / 1.01 | 1.17 | 1.11 | no |
| `raw_buy:EMACross` AND `raw_buy:UO` | 1.28 | 1.14 / 1.07 | 1.17 | 0.91 | no |
| `raw_buy:HTSINE` AND `raw_buy:UO` | 1.28 | 1.13 / 1.07 | 1.15 | 0.78 | no |
| `raw_buy:HTSINE` AND `raw_buy:WTO` | 1.27 | 1.13 / 1.13 | 1.17 | 1.17 | no |
| `raw_buy:SMACross` AND `raw_buy:WTO` | 1.27 | 1.15 / 1.13 | 1.22 | 1.19 | yes |
| `raw_buy:SMACross` AND `raw_buy:VFI` | 1.27 | 1.15 / 0.99 | 1.04 | 1.14 | no |

### label:buy, candidate bars: 2775 pairs, 76 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_buy:KST` AND `raw_buy:VBM` | 1.55 | 1.07 / 1.05 | 1.17 | 1.21 | no |
| `raw_buy:CCI` AND `raw_buy:COPP` | 1.47 | 1.24 / 1.13 | 1.50 | 1.27 | yes |
| `raw_buy:CCI` AND `raw_buy:WTO` | 1.47 | 1.24 / 1.18 | 1.29 | 1.15 | yes |
| `raw_buy:COPP` AND `raw_buy:WTO` | 1.46 | 1.13 / 1.18 | 1.33 | 1.22 | yes |
| `raw_buy:CCI` AND `raw_buy:KST` | 1.46 | 1.24 / 1.07 | 1.38 | 1.27 | yes |
| `raw_buy:KST` AND `raw_buy:ZScore` | 1.46 | 1.07 / 1.18 | 1.43 | 1.28 | yes |
| `raw_buy:WTO` AND `raw_buy:ZScore` | 1.44 | 1.18 / 1.18 | 1.35 | 1.15 | yes |
| `raw_buy:HTSINE` AND `raw_buy:ZScore` | 1.44 | 1.03 / 1.18 | 1.37 | 1.11 | yes |
| `raw_buy:DCRevert` AND `raw_buy:EMACross` | 1.40 | 1.10 / 1.10 | 1.10 | 1.19 | no |
| `raw_buy:DCRevert` AND `raw_buy:SMACross` | 1.40 | 1.10 / 1.09 | 1.12 | 1.15 | no |
| `raw_buy:KST` AND `raw_buy:WTO` | 1.39 | 1.07 / 1.18 | 1.27 | 1.18 | yes |
| `raw_buy:PVO` AND `raw_buy:VBM` | 1.38 | 1.07 / 1.05 | 1.05 | 1.07 | no |
| `raw_buy:FVE` AND `raw_buy:ZScore` | 1.38 | 1.01 / 1.18 | 1.33 | 1.14 | yes |
| `raw_buy:CMO` AND `raw_buy:FVE` | 1.38 | 1.06 / 1.01 | 1.22 | 0.83 | early+late only |
| `raw_buy:CCI` AND `raw_buy:MI` | 1.38 | 1.24 / 1.10 | 1.29 | 1.16 | yes |
| `raw_buy:KST` AND `raw_buy:LWRevert` | 1.37 | 1.07 / 1.13 | 1.22 | 1.20 | yes |
| `raw_buy:STC` AND `raw_buy:ZScore` | 1.36 | 1.00 / 1.18 | 1.22 | 1.11 | no |
| `raw_buy:COPP` AND `raw_buy:MI` | 1.36 | 1.13 / 1.10 | 1.29 | 1.22 | yes |
| `raw_buy:COPP` AND `raw_buy:VBM` | 1.35 | 1.13 / 1.05 | 1.22 | 1.22 | yes |
| `raw_buy:SMACross` AND `raw_buy:ZScore` | 1.35 | 1.09 / 1.18 | 1.29 | 1.16 | yes |

### label:sell, all bars: 2850 pairs, 525 pass the rule

| pair | early lift | components early | late lift | E0 lift | confirmed |
|---|---|---|---|---|---|
| `raw_sell:EWO` AND `raw_sell:MFI` | 15.17 | 12.85 / 2.18 | 8.63 | 7.86 | no |
| `raw_sell:KST` AND `raw_sell:WILLR` | 14.46 | 10.98 / 2.67 | 9.74 | 9.48 | yes |
| `raw_sell:EWO` AND `raw_sell:KST` | 14.20 | 12.85 / 10.98 | 8.84 | 9.15 | no |
| `raw_sell:COPP` AND `raw_sell:Ehlers_LW:cycle` | 13.82 | 11.13 / 1.67 | 10.01 | 7.80 | yes |
| `raw_sell:KST` AND `raw_sell:SAR` | 13.65 | 10.98 / 1.72 | 8.89 | 8.89 | yes |
| `raw_sell:MFI` AND `raw_sell:TRIX` | 13.65 | 2.18 / 10.71 | 8.30 | 7.27 | yes |
| `raw_sell:COPP` AND `raw_sell:LWRevert` | 13.57 | 11.13 / 1.64 | 10.37 | 7.77 | yes |
| `raw_sell:COPP` AND `raw_sell:UO` | 13.51 | 11.13 / 2.53 | 10.09 | 7.16 | early+late only |
| `raw_sell:TRIX` AND `raw_sell:UO` | 13.42 | 10.71 / 2.53 | 10.68 | 8.90 | yes |
| `raw_sell:TRIX` AND `raw_sell:VORTEX` | 13.12 | 10.71 / 4.17 | 8.22 | 8.56 | yes |
| `raw_sell:TRIX` AND `raw_sell:WILLR` | 12.82 | 10.71 / 2.67 | 8.90 | 8.09 | yes |
| `raw_sell:COPP` AND `raw_sell:KST` | 12.63 | 11.13 / 10.98 | 9.97 | 8.42 | yes |
| `raw_sell:SAR` AND `raw_sell:TRIX` | 12.60 | 1.72 / 10.71 | 8.24 | 7.93 | yes |
| `raw_sell:COPP` AND `raw_sell:TRIX` | 12.47 | 11.13 / 10.71 | 8.88 | 7.83 | no |
| `raw_sell:CMO` AND `raw_sell:COPP` | 12.47 | 3.22 / 11.13 | 10.30 | 6.07 | early+late only |
| `raw_sell:COPP` AND `raw_sell:VFI` | 12.37 | 11.13 / 1.86 | 11.40 | 7.81 | yes |
| `raw_sell:CMO` AND `raw_sell:PPO` | 10.64 | 3.22 / 7.60 | 8.84 | 5.23 | early+late only |
| `raw_sell:LWRevert` AND `raw_sell:ROC` | 10.49 | 1.64 / 9.06 | 9.03 | 6.77 | no |
| `raw_sell:Ehlers_LW:cycle` AND `raw_sell:ROC` | 10.43 | 1.67 / 9.06 | 9.26 | 6.70 | no |
| `raw_sell:BBBreakout:tm0` AND `raw_sell:PPO` | 10.43 | 2.11 / 7.60 | 7.64 | 7.43 | yes |

## Multivariate models (per fold; full vs bb_width + guard_metric)

### A1:288, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.595 | 0.512 | 0.558 | 0.509 | 0.560 / 0.512 | – / – | True |
| hgb | 0.535 | 0.516 | 0.550 | 0.512 | 0.560 / 0.512 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.531 / 0.430; beats it in both halves: tree False, hgb False.

Recurring tree splits: `ni:ni_ms_br` ×13, `btc:WOBV:wobv_raw` ×8, `btc:base:doy_cos` ×7, `btc:base:doy_sin` ×4, `base:doy_sin` ×3, `ni:ni_ms_vr` ×3, `base:doy_cos` ×2, `btc:Ichimoku:senkou_span_b` ×1, `ni:ni_ms_bv` ×1, `btc:Ehlers_KC:kc_lower` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### A1:288, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.540 | 0.533 | 0.564 | 0.548 | 0.541 / 0.558 | – / – | True |
| hgb | 0.552 | 0.511 | 0.605 | 0.530 | 0.541 / 0.558 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.513 / 0.398; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×10, `ni:ni_ms_br` ×8, `base:doy_cos` ×8, `btc:base:doy_cos` ×4, `btc:base:doy_sin` ×4, `btc:base:ema200` ×3, `btc:PSAR:ema200` ×3, `base:doy_sin` ×1, `btc:LWRevert:atr` ×1, `btc:KCBreakout:kc_lower` ×1 (tagged, not comparable across time or pairs: 9 of 10)

### A1:48, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.503 | 0.512 | 0.495 | 0.507 | 0.519 / 0.499 | – / – | False |
| hgb | 0.539 | 0.519 | 0.538 | 0.505 | 0.519 / 0.499 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.469 / 0.443; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:atr_pct` ×9, `btc:base:atr_pct_roll` ×5, `btc:Ehlers_LW:ema_slow` ×4, `btc:WOBV:wobv_raw` ×4, `ni:ni_ms_br` ×3, `ni:ni_ms_vr` ×3, `base:doy_sin` ×3, `btc:DCBreakout:dc_upper` ×2, `btc:Ehlers_Ichimoku:senkou_span_b` ×2, `btc:base:dow_cos` ×2 (tagged, not comparable across time or pairs: 6 of 10)

### A1:48, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.525 | 0.554 | 0.509 | 0.525 | 0.543 / 0.508 | – / – | False |
| hgb | 0.588 | 0.543 | 0.595 | 0.523 | 0.543 / 0.508 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.456 / 0.479; beats it in both halves: tree False, hgb True.

Recurring tree splits: `btc:base:atr_pct_roll` ×15, `btc:base:ema200` ×6, `base:doy_cos` ×5, `btc:PSAR:ema200` ×4, `btc:Ehlers_Ichimoku:cloud_top` ×4, `btc:Ichimoku:senkou_span_a` ×3, `ni:ni_ms_br` ×2, `base:dow_cos` ×2, `btc:base:doy_cos` ×1, `btc:ZScore:rolling_mean` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### A1:672, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.503 | 0.506 | 0.509 | 0.498 | 0.606 / 0.491 | – / – | False |
| hgb | 0.584 | 0.501 | 0.524 | 0.502 | 0.606 / 0.491 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.465 / 0.415; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_cos` ×11, `btc:WOBV:wobv_raw` ×10, `base:doy_cos` ×3, `btc:base:ema200` ×3, `base:doy_sin` ×3, `btc:base:doy_sin` ×3, `btc:Ehlers_KC:kc_lower` ×3, `btc:PSAR:ema200` ×2, `ni:ni_ms_br` ×2, `btc:DCBreakout:dc_lower` ×2 (tagged, not comparable across time or pairs: 9 of 10)

### A1:672, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.561 | 0.521 | 0.582 | 0.517 | 0.539 / 0.532 | – / – | True |
| hgb | 0.486 | 0.521 | 0.578 | 0.535 | 0.539 / 0.532 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.376 / 0.346; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×15, `base:doy_cos` ×8, `btc:base:doy_cos` ×8, `btc:base:doy_sin` ×4, `base:doy_sin` ×2, `btc:PSAR:ema200` ×2, `btc:Ehlers_KC:kc_lower` ×2, `btc:DCBreakout:dc_lower` ×1, `btc:base:ema200` ×1, `ni:ni_ms_vr` ×1 (tagged, not comparable across time or pairs: 9 of 10)

### A2:288, all bars (computable folds: 13)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.498 | 0.446 | 0.531 | 0.508 | 0.516 / 0.553 | – / – | True |
| hgb | 0.552 | 0.469 | 0.600 | 0.518 | 0.516 / 0.553 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.535 / 0.259; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_sin` ×6, `ni:ni_ms_br` ×5, `base:doy_sin` ×4, `btc:WOBV:wobv_raw` ×2, `ni:ni_ms_bv` ×2, `btc:KCBreakout:kc_lower` ×2, `btc:base:doy_cos` ×2, `base:doy_cos` ×2, `btc:Ehlers_LW:ss` ×1, `btc:Ehlers_Ichimoku:senkou_span_a` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### A2:288, candidate bars (computable folds: 13)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.500 | 0.497 | 0.555 | 0.551 | 0.519 / 0.560 | – / – | True |
| hgb | 0.522 | 0.497 | 0.768 | 0.537 | 0.519 / 0.560 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.555 / 0.313; beats it in both halves: tree False, hgb False.

Recurring tree splits: `base:doy_sin` ×6, `btc:base:doy_sin` ×6, `ni:ni_ms_br` ×5, `btc:base:doy_cos` ×3, `btc:EMACross:ema_long` ×2, `btc:base:ema200` ×2, `btc:PSAR:ema200` ×2, `btc:Ehlers_LW:ss` ×1, `btc:SMACross:sma_long` ×1, `btc:base:ss_close` ×1 (tagged, not comparable across time or pairs: 9 of 10)

### A2:48, all bars (computable folds: 13)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.465 | 0.507 | 0.473 | 0.627 | 0.541 / 0.658 | – / – | False |
| hgb | 0.552 | 0.547 | 0.522 | 0.590 | 0.541 / 0.658 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.446 / 0.493; beats it in both halves: tree False, hgb False.

Recurring tree splits: `ni:ni_ms_bv` ×5, `btc:base:doy_cos` ×4, `base:doy_cos` ×3, `btc:base:minus_di_scaled` ×2, `base:dow_sin` ×2, `base:doy_sin` ×2, `btc:WOBV:wobv_raw` ×2, `btc:base:doy_sin` ×2, `ni:ni_ms_vr` ×1, `btc:base:atr_pct_roll` ×1 (tagged, not comparable across time or pairs: 6 of 10)

### A2:48, candidate bars (computable folds: 13)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.516 | 0.507 | 0.520 | 0.547 | 0.482 / 0.569 | – / – | False |
| hgb | 0.491 | 0.543 | 0.674 | 0.543 | 0.482 / 0.569 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.386 / 0.477; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_cos` ×4, `btc:base:atr_pct_roll` ×3, `ni:ni_ms_bv` ×3, `base:dow_cos` ×3, `btc:VBM:vbm_zscore` ×2, `base:doy_cos` ×2, `btc:LWBreakout:atr` ×2, `btc:DCBreakout:dc_upper` ×2, `btc:PSAR:ema200` ×2, `btc:Ehlers_LW:lw_atr` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### A2:672, all bars (computable folds: 13)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.662 | 0.428 | 0.494 | 0.493 | 0.557 / 0.488 | – / – | True |
| hgb | 0.825 | 0.475 | 0.415 | 0.505 | 0.557 / 0.488 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.274 / 0.159; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_sin` ×7, `base:doy_sin` ×6, `ni:ni_ms_br` ×5, `base:doy_cos` ×3, `btc:WOBV:wobv_raw` ×3, `btc:base:ema200` ×3, `ni:ni_ms_vr` ×2, `btc:Ehlers_DYMI:ss` ×1, `Ehlers_ROC:cg` ×1, `btc:base:doy_cos` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### A2:672, candidate bars (computable folds: 13)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.653 | 0.508 | 0.550 | 0.538 | 0.547 / 0.553 | – / – | True |
| hgb | 0.806 | 0.508 | 0.700 | 0.543 | 0.547 / 0.553 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.244 / 0.234; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_sin` ×6, `base:doy_sin` ×4, `ni:ni_ms_br` ×4, `base:doy_cos` ×3, `ni:ni_ms_vr` ×3, `btc:base:doy_cos` ×3, `btc:PSAR:ema200` ×3, `btc:WOBV:wobv_raw` ×2, `btc:DCBreakout:dc_lower` ×2, `btc:SMACross:sma_long` ×2 (tagged, not comparable across time or pairs: 8 of 10)

### B-btc:288, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.520 | 0.495 | 0.506 | 0.510 | 0.504 / 0.507 | – / – | False |
| hgb | 0.554 | 0.499 | 0.598 | 0.499 | 0.504 / 0.507 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.488 / 0.445; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_cos` ×12, `btc:WOBV:wobv_raw` ×8, `btc:BBRevert:bb_lowerband` ×7, `btc:DCRevert:dc_lower` ×4, `base:doy_cos` ×4, `ni:ni_ms_vr` ×3, `base:doy_sin` ×2, `btc:base:doy_sin` ×1, `btc:TEMACross:tema_short/close` ×1, `btc:base:macdsignal` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### B-btc:288, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.544 | 0.512 | 0.523 | 0.533 | 0.533 / 0.546 | – / – | False |
| hgb | 0.483 | 0.508 | 0.638 | 0.511 | 0.533 / 0.546 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.481 / 0.426; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:doy_cos` ×9, `btc:PSAR:ema200` ×8, `btc:WOBV:wobv_raw` ×7, `btc:base:ema200` ×5, `ni:ni_ms_bv` ×4, `base:doy_cos` ×4, `ni:ni_ms_vr` ×3, `base:dow_cos` ×2, `btc:base:vwap_pos` ×1, `btc:base:dow_cos` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### B-btc:48, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.491 | 0.503 | 0.514 | 0.508 | 0.502 / 0.504 | – / – | False |
| hgb | 0.535 | 0.500 | 0.539 | 0.503 | 0.502 / 0.504 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.494 / 0.471; beats it in both halves: tree False, hgb True.

Recurring tree splits: `btc:WOBV:wobv_raw` ×9, `ni:ni_ms_vr` ×7, `btc:LWRevert:buy_level` ×5, `base:doy_cos` ×4, `btc:base:doy_cos` ×3, `btc:base:ema200` ×3, `btc:base:dow_sin` ×2, `base:dow_sin` ×2, `btc:base:atr_pct_roll` ×2, `ni:ni_ms_bv` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### B-btc:48, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.500 | 0.501 | 0.504 | 0.504 | 0.513 / 0.525 | – / – | False |
| hgb | 0.527 | 0.507 | 0.564 | 0.508 | 0.513 / 0.525 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.499 / 0.474; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:atr_pct` ×10, `ni:ni_ms_bv` ×8, `btc:base:atr_pct_roll` ×4, `btc:PSAR:ema200` ×3, `base:doy_sin` ×3, `btc:base:doy_cos` ×2, `ni:ni_ms_vr` ×2, `btc:Ehlers_DC:dc_upper` ×2, `base:doy_cos` ×2, `btc:TSI:tsi` ×2 (tagged, not comparable across time or pairs: 5 of 10)

### B-btc:672, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.508 | 0.489 | 0.500 | 0.499 | 0.528 / 0.484 | – / – | True |
| hgb | 0.536 | 0.494 | 0.552 | 0.497 | 0.528 / 0.484 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.355 / 0.419; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×10, `base:doy_cos` ×4, `btc:base:doy_cos` ×4, `ni:ni_po_tt` ×4, `btc:Ehlers_Fisher:ema_fast` ×3, `btc:PSAR:ema200` ×2, `base:doy_sin` ×2, `btc:FVE:fve` ×2, `btc:Ehlers_ind_DYMI:ema_fast` ×1, `btc:base:atr_pct_roll` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### B-btc:672, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.503 | 0.508 | 0.649 | 0.503 | 0.533 / 0.510 | – / – | False |
| hgb | 0.545 | 0.506 | 0.677 | 0.502 | 0.533 / 0.510 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.345 / 0.345; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×13, `btc:base:doy_sin` ×5, `base:doy_sin` ×3, `btc:base:ema200` ×3, `ni:ni_ms_vr` ×3, `btc:base:doy_cos` ×3, `btc:PSAR:ema200` ×2, `base:doy_cos` ×2, `btc:DCBreakout:dc_lower` ×2, `btc:DCRevert:dc_lower` ×2 (tagged, not comparable across time or pairs: 9 of 10)

### B-pair:288, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.501 | 0.515 | 0.491 | 0.498 | 0.526 / 0.496 | – / – | False |
| hgb | 0.540 | 0.514 | 0.576 | 0.504 | 0.526 / 0.496 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.494 / 0.428; beats it in both halves: tree False, hgb True.

Recurring tree splits: `btc:base:doy_cos` ×8, `base:doy_cos` ×6, `btc:WOBV:wobv_raw` ×6, `btc:base:doy_sin` ×4, `btc:PSAR:ema200` ×3, `ni:ni_ms_br` ×3, `ni:ni_ms_vr` ×3, `btc:base:ema200` ×3, `btc:Ichimoku:senkou_span_b` ×2, `btc:BBRevert:bb_lowerband` ×2 (tagged, not comparable across time or pairs: 8 of 10)

### B-pair:288, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.484 | 0.516 | 0.474 | 0.517 | 0.531 / 0.555 | – / – | False |
| hgb | 0.528 | 0.509 | 0.572 | 0.514 | 0.531 / 0.555 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.473 / 0.405; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:ema200` ×11, `btc:base:doy_cos` ×6, `btc:WOBV:wobv_raw` ×6, `base:doy_sin` ×4, `btc:base:doy_sin` ×4, `btc:EMACross:ema_long` ×3, `ni:ni_ms_br` ×3, `base:doy_cos` ×2, `btc:PSAR:ema200` ×2, `ni:ni_ms_bv` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### B-pair:48, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.505 | 0.529 | 0.523 | 0.528 | 0.518 / 0.508 | – / – | False |
| hgb | 0.540 | 0.524 | 0.532 | 0.528 | 0.518 / 0.508 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.472 / 0.457; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:BBBreakout:bb_upperband` ×7, `btc:base:atr_pct` ×7, `ni:ni_ms_vr` ×6, `btc:base:atr_pct_roll` ×5, `btc:Ehlers_KC:kc_mid` ×4, `base:dymi_scaled` ×3, `CMO:cmo` ×3, `btc:base:adx` ×2, `base:dow_sin` ×2, `btc:ZScore:rolling_std` ×1 (tagged, not comparable across time or pairs: 4 of 10)

### B-pair:48, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.510 | 0.520 | 0.531 | 0.528 | 0.537 / 0.519 | – / – | False |
| hgb | 0.570 | 0.515 | 0.547 | 0.523 | 0.537 / 0.519 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.448 / 0.474; beats it in both halves: tree False, hgb True.

Recurring tree splits: `ni:ni_ms_bv` ×9, `base:ema200/close` ×5, `btc:Ehlers_KC:kc_lower` ×4, `btc:PSAR:ema200` ×3, `Ehlers_Ichimoku:senkou_span_b/close` ×3, `btc:Ehlers_Ichimoku:cloud_bottom` ×3, `btc:base:doy_cos` ×3, `btc:base:ema200` ×2, `base:doy_cos` ×2, `btc:Ichimoku:senkou_span_b` ×2 (tagged, not comparable across time or pairs: 7 of 10)

### B-pair:672, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.502 | 0.530 | 0.556 | 0.489 | 0.545 / 0.483 | – / – | False |
| hgb | 0.559 | 0.522 | 0.551 | 0.485 | 0.545 / 0.483 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.410 / 0.418; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×10, `base:doy_cos` ×6, `btc:base:doy_cos` ×5, `btc:base:ema200` ×4, `btc:base:doy_sin` ×4, `btc:DCRevert:dc_lower` ×4, `base:doy_sin` ×3, `ni:ni_po_tt` ×3, `btc:Ehlers_KC:kc_lower` ×2, `btc:Ehlers_ind_DYMI:ema_fast` ×2 (tagged, not comparable across time or pairs: 9 of 10)

### B-pair:672, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.561 | 0.529 | 0.503 | 0.507 | 0.541 / 0.499 | – / – | False |
| hgb | 0.508 | 0.507 | 0.555 | 0.500 | 0.541 / 0.499 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.385 / 0.389; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×15, `base:doy_sin` ×8, `btc:PSAR:ema200` ×2, `btc:base:ema200` ×2, `btc:base:doy_sin` ×2, `btc:Ehlers_Ichimoku:senkou_span_b` ×2, `ni:ni_ms_br` ×2, `btc:DCRevert:dc_upper` ×2, `btc:DCBreakout:dc_lower` ×2, `btc:base:bb_middleband` ×1 (tagged, not comparable across time or pairs: 9 of 10)

### C-btc:288, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.631 | 0.651 | 0.678 | 0.637 | 0.653 / 0.641 | – / – | False |
| hgb | 0.645 | 0.662 | 0.757 | 0.649 | 0.653 / 0.641 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.723 / 0.823; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:base:atr_pct` ×15, `btc:WOBV:wobv_raw` ×15, `base:doy_cos` ×3, `btc:base:doy_sin` ×2, `btc:base:ema200/close` ×2, `base:doy_sin` ×2, `btc:base:ema200` ×2, `btc:base:doy_cos` ×1, `btc:DCBreakout:dc_upper` ×1, `btc:LWBreakout:atr` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### C-btc:288, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.653 | 0.597 | 0.694 | 0.589 | 0.605 / 0.602 | – / – | True |
| hgb | 0.680 | 0.593 | 0.766 | 0.585 | 0.605 / 0.602 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.769 / 0.864; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×13, `btc:base:atr_pct_roll` ×9, `btc:base:atr_pct` ×6, `btc:base:doy_cos` ×3, `btc:PSAR:ema200` ×3, `ni:ni_ms_vr` ×2, `base:doy_sin` ×2, `btc:base:ema200` ×2, `btc:base:dow_cos` ×2, `btc:LWBreakout:buy_level` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### C-btc:48, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.733 | 0.641 | 0.838 | 0.684 | 0.645 / 0.687 | – / – | True |
| hgb | 0.878 | 0.637 | 0.924 | 0.695 | 0.645 / 0.687 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.704 / 0.803; beats it in both halves: tree True, hgb True.

Recurring tree splits: `btc:base:atr_pct` ×15, `btc:WOBV:wobv_raw` ×7, `btc:base:dow_sin` ×5, `btc:base:mod_cos` ×5, `btc:PSAR:ema200/close` ×3, `base:dow_sin` ×3, `base:mod_cos` ×3, `btc:base:atr_pct_roll` ×1, `btc:base:ema200/close` ×1, `btc:Ehlers_LW:lw_atr` ×1 (tagged, not comparable across time or pairs: 6 of 10)

### C-btc:48, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.724 | 0.567 | 0.790 | 0.586 | 0.560 / 0.586 | – / – | True |
| hgb | 0.903 | 0.546 | 0.930 | 0.596 | 0.560 / 0.586 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.779 / 0.818; beats it in both halves: tree False, hgb True.

Recurring tree splits: `btc:base:atr_pct` ×15, `btc:WOBV:wobv_raw` ×11, `btc:base:mod_cos` ×5, `btc:Ehlers_LW:lw_atr` ×5, `btc:base:doy_sin` ×3, `base:mod_cos` ×3, `ni:ni_ms_vr` ×1, `btc:PSAR:ema200` ×1, `btc:LWBreakout:atr/close` ×1 (tagged, not comparable across time or pairs: 6 of 9)

### C-btc:672, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.500 | 0.685 | 0.652 | 0.631 | 0.695 / 0.636 | – / – | False |
| hgb | 0.652 | 0.702 | 0.720 | 0.638 | 0.695 / 0.636 | – / – | False |

Persistence baseline AUC on the same test rows (early / late): 0.845 / 0.756; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×14, `btc:base:atr_pct_roll` ×10, `base:doy_sin` ×7, `btc:base:doy_cos` ×3, `btc:base:doy_sin` ×3, `base:doy_cos` ×2, `btc:base:atr_pct` ×1, `btc:PSAR:ema200` ×1, `btc:Ehlers_KC:ema_slow` ×1, `btc:base:rsi` ×1 (tagged, not comparable across time or pairs: 7 of 10)

### C-btc:672, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.598 | 0.611 | 0.745 | 0.588 | 0.600 / 0.590 | – / – | False |
| hgb | 0.785 | 0.618 | 0.831 | 0.594 | 0.600 / 0.590 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.934 / 0.814; beats it in both halves: tree False, hgb False.

Recurring tree splits: `btc:WOBV:wobv_raw` ×13, `btc:base:atr_pct` ×6, `btc:base:atr_pct_roll` ×6, `base:doy_sin` ×3, `base:doy_cos` ×3, `btc:base:doy_cos` ×3, `btc:base:doy_sin` ×2, `ni:ni_ms_rr` ×1, `btc:PSAR:ema200/close` ×1, `btc:Ichimoku:kijun_sen` ×1 (tagged, not comparable across time or pairs: 6 of 10)

### C-pair:288, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.806 | 0.754 | 0.797 | 0.764 | 0.757 / 0.754 | – / – | True |
| hgb | 0.910 | 0.771 | 0.882 | 0.797 | 0.757 / 0.754 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.903 / 0.918; beats it in both halves: tree False, hgb False.

Recurring tree splits: `base:atr_pct` ×15, `WOBV:wobv_raw` ×14, `btc:WOBV:wobv_raw` ×5, `base:atr_pct_roll` ×5, `btc:base:doy_cos` ×2, `btc:PSAR:ema200` ×1, `Ehlers_KC:ema_slow` ×1, `btc:Ehlers_DC:dc_lower` ×1, `EhlersBase:ema_fast` ×1 (tagged, not comparable across time or pairs: 7 of 9)

### C-pair:288, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.844 | 0.681 | 0.828 | 0.648 | 0.614 / 0.655 | – / – | True |
| hgb | 0.926 | 0.696 | 0.904 | 0.662 | 0.614 / 0.655 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.925 / 0.914; beats it in both halves: tree False, hgb False.

Recurring tree splits: `base:atr_pct_roll` ×15, `WOBV:wobv_raw` ×14, `btc:WOBV:wobv_raw` ×5, `btc:base:dow_cos` ×5, `base:atr_pct` ×1, `btc:base:doy_sin` ×1, `base:dow_cos` ×1, `btc:Ehlers_DC:dc_mid` ×1, `btc:PSAR:ema200` ×1, `base:bb_upperband` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### C-pair:48, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.809 | 0.795 | 0.783 | 0.781 | 0.800 / 0.783 | – / – | True |
| hgb | 0.916 | 0.811 | 0.902 | 0.799 | 0.800 / 0.783 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.873 / 0.841; beats it in both halves: tree False, hgb True.

Recurring tree splits: `base:atr_pct` ×15, `WOBV:wobv_raw` ×9, `btc:WOBV:wobv_raw` ×6, `base:atr_pct_roll` ×6, `ZScore:rolling_std/close` ×4, `raw_sell:TSI` ×2, `base:bb_width` ×2, `base:doy_cos` ×1 (tagged, not comparable across time or pairs: 3 of 8)

### C-pair:48, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.816 | 0.673 | 0.745 | 0.666 | 0.635 / 0.628 | – / – | True |
| hgb | 0.919 | 0.685 | 0.889 | 0.683 | 0.635 / 0.628 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.843 / 0.795; beats it in both halves: tree False, hgb True.

Recurring tree splits: `base:atr_pct_roll` ×15, `WOBV:wobv_raw` ×12, `btc:WOBV:wobv_raw` ×8, `base:atr_pct` ×4, `base:minus_di_scaled` ×3, `raw_sell:TSI` ×2, `btc:base:mod_cos` ×1 (tagged, not comparable across time or pairs: 3 of 7)

### C-pair:672, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.829 | 0.730 | 0.773 | 0.747 | 0.735 / 0.714 | – / – | True |
| hgb | 0.844 | 0.751 | 0.832 | 0.765 | 0.735 / 0.714 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.877 / 0.890; beats it in both halves: tree False, hgb False.

Recurring tree splits: `WOBV:wobv_raw` ×14, `base:atr_pct_roll` ×13, `base:atr_pct` ×4, `btc:WOBV:wobv_raw` ×4, `EMACross:ema_long` ×3, `btc:base:doy_cos` ×2, `Ehlers_Ichimoku:senkou_span_b` ×2, `btc:PSAR:ema200` ×1, `btc:BBRevert:bb_upperband` ×1, `btc:Ehlers_KC:kc_upper` ×1 (tagged, not comparable across time or pairs: 8 of 10)

### C-pair:672, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.824 | 0.673 | 0.763 | 0.670 | 0.616 / 0.659 | – / – | True |
| hgb | 0.894 | 0.682 | 0.893 | 0.675 | 0.616 / 0.659 | – / – | True |

Persistence baseline AUC on the same test rows (early / late): 0.908 / 0.885; beats it in both halves: tree False, hgb False.

Recurring tree splits: `base:atr_pct_roll` ×15, `WOBV:wobv_raw` ×15, `btc:WOBV:wobv_raw` ×7, `btc:TEMACross:tema_short` ×2, `btc:Ehlers_Ichimoku:cloud_top` ×2, `ni:ni_po_tt` ×1, `LWBreakout:atr` ×1, `LWRevert:sell_level` ×1, `btc:base:doy_cos` ×1 (tagged, not comparable across time or pairs: 7 of 9)

### label:buy, candidate bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.599 | 0.550 | 0.592 | 0.548 | 0.549 / 0.554 | 0.607 / 0.597 | True |
| hgb | 0.619 | 0.552 | 0.625 | 0.551 | 0.549 / 0.554 | 0.607 / 0.597 | True |

Recurring tree splits: `base:atr_pct_roll` ×14, `base:fastk_scaled` ×7, `base:spread_ma` ×6, `SAR:sar/close` ×5, `base:atr_pct` ×2, `WOBV:wobv_raw` ×2, `ni:ni_of_tbu` ×2, `ni:ni_ms_bv` ×2, `ZScore:rolling_std/close` ×1, `Ehlers_KC:kc_upper/close` ×1 (tagged, not comparable across time or pairs: 1 of 10)

### label:sell, all bars (computable folds: 15)

| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |
|---|---|---|---|---|---|---|---|
| tree | 0.965 | 0.970 | 0.962 | 0.970 | 0.907 / 0.890 | – / – | False |
| hgb | 0.975 | 0.970 | 0.975 | 0.969 | 0.907 / 0.890 | – / – | True |

Recurring tree splits: `base:bb_lowerband/close` ×15, `base:bb_width` ×15, `base:guard_metric_pos` ×11, `base:guard_metric_neg` ×4

## Regime screen

Regime targets at 3 d / 7 d are heavily autocorrelated: one 90-day window holds only ~13–30 independent periods, so window medians and the same-direction share are the evidence, not row counts. `btc:` columns against market-wide targets are scored one row per timestamp.

### A1:48 (all bars): persistence baseline AUC 0.458

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:Ehlers_Ichimoku:tenkan_sen` | price_level | 0.443 | 0.430 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:Ichimoku:tenkan_sen` | price_level | 0.443 | 0.430 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:EMACross:ema_short` | price_level | 0.443 | 0.430 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:SMACross:sma_short` | price_level | 0.443 | 0.430 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:EhlersBase:ss` | price_level | 0.443 | 0.431 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:Ehlers_BB:ss` | price_level | 0.443 | 0.431 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:Ehlers_DC:ss` | price_level | 0.443 | 0.431 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |
| `btc:Ehlers_DYMI:ss` | price_level | 0.443 | 0.431 | 0.443 | 0.467 | 0.67 | 31 | 0.00 | 0.08 |

### A2:48 (all bars): persistence baseline AUC 0.471

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:base:dow_cos` | calendar | 0.546 | 0.556 | 0.494 | – | 0.54 | 12 | -0.03 | 0.00 |
| `base:atr_pct_roll` | indicator | 0.662 | 0.662 | 0.657 | – | 0.79 | 83 | 0.02 | 0.61 |
| `ni:ni_ms_bv` | indicator | 0.620 | 0.610 | 0.666 | – | 0.79 | 80 | 0.01 | 0.28 |
| `PSAR:ema200/close` | indicator | 0.505 | 0.492 | 0.615 | – | 0.57 | 152 | -0.45 | -0.06 |
| `base:ema200/close` | indicator | 0.505 | 0.492 | 0.615 | – | 0.57 | 152 | -0.45 | -0.06 |
| `btc:raw8:range` | indicator | 0.645 | 0.645 | 0.605 | – | 0.77 | 71 | 0.03 | 0.32 |
| `base:dow_cos` | calendar | 0.523 | 0.528 | 0.494 | – | 0.50 | 12 | -0.03 | 0.00 |
| `btc:base:atr_pct_roll` | indicator | 0.636 | 0.636 | 0.672 | – | 0.92 | 83 | 0.02 | 0.61 |

### B-btc:48 (all bars): persistence baseline AUC 0.488

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:Ehlers_LW:lw_cycle_sell` | price_level | 0.406 | 0.401 | 0.423 | 0.491 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:TEMACross:tema_short` | price_level | 0.406 | 0.399 | 0.422 | 0.490 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:LWRevert:sell_level` | price_level | 0.406 | 0.401 | 0.423 | 0.491 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:base:close_safe` | price_level | 0.406 | 0.399 | 0.422 | 0.490 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:LWBreakout:buy_level` | price_level | 0.406 | 0.401 | 0.423 | 0.491 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:Ehlers_LW:lw_trend_buy` | price_level | 0.406 | 0.400 | 0.423 | 0.491 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:base:ss_close` | price_level | 0.406 | 0.399 | 0.423 | 0.490 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:LWBreakout:sell_level` | price_level | 0.407 | 0.398 | 0.422 | 0.490 | 1.00 | 31 | 0.01 | 0.07 |

### B-pair:48 (all bars): persistence baseline AUC 0.472

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:SMACross:sma_long` | price_level | 0.453 | 0.456 | 0.453 | 0.493 | 0.93 | 31 | -0.00 | 0.08 |
| `btc:EMACross:ema_long` | price_level | 0.453 | 0.455 | 0.453 | 0.493 | 0.93 | 31 | -0.00 | 0.08 |
| `btc:Ehlers_LW:lw_cycle_buy` | price_level | 0.453 | 0.449 | 0.456 | 0.492 | 0.93 | 31 | 0.01 | 0.07 |
| `btc:LWRevert:buy_level` | price_level | 0.453 | 0.449 | 0.456 | 0.492 | 0.93 | 31 | 0.01 | 0.07 |
| `PSAR:ema200/close` | indicator | 0.546 | 0.550 | 0.546 | 0.524 | 1.00 | 152 | -0.45 | -0.06 |
| `base:ema200/close` | indicator | 0.546 | 0.550 | 0.546 | 0.524 | 1.00 | 152 | -0.45 | -0.06 |
| `btc:LWBreakout:sell_level` | price_level | 0.454 | 0.451 | 0.456 | 0.493 | 0.93 | 31 | 0.01 | 0.07 |
| `btc:Ehlers_LW:lw_trend_sell` | price_level | 0.454 | 0.451 | 0.456 | 0.493 | 0.93 | 31 | 0.01 | 0.07 |

### C-btc:48 (all bars): persistence baseline AUC 0.792

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:base:atr_pct` | indicator | 0.818 | 0.750 | 0.832 | 0.887 | 1.00 | 83 | 0.01 | 0.70 |
| `btc:base:atr_pct_roll` | indicator | 0.800 | 0.729 | 0.824 | 0.882 | 1.00 | 83 | 0.02 | 0.61 |
| `btc:Ehlers_LW:lw_atr/close` | indicator | 0.780 | 0.731 | 0.802 | 0.855 | 1.00 | 83 | 0.02 | 0.77 |
| `btc:VBM:vbm` | indicator | 0.778 | 0.723 | 0.808 | 0.857 | 1.00 | 83 | 0.04 | 0.73 |
| `btc:LWBreakout:atr/close` | indicator | 0.777 | 0.725 | 0.806 | 0.857 | 1.00 | 83 | 0.02 | 0.74 |
| `btc:LWBreakout:level_pct_diff` | indicator | 0.777 | 0.725 | 0.806 | 0.857 | 1.00 | 83 | 0.02 | 0.74 |
| `btc:LWRevert:atr/close` | indicator | 0.777 | 0.725 | 0.806 | 0.857 | 1.00 | 83 | 0.02 | 0.74 |
| `btc:ZScore:rolling_std/close` | indicator | 0.777 | 0.700 | 0.778 | 0.796 | 1.00 | 159 | 0.01 | 1.00 |

### C-pair:48 (all bars): persistence baseline AUC 0.864

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `base:atr_pct` | indicator | 0.785 | 0.828 | 0.778 | 0.885 | 1.00 | 83 | 0.01 | 0.70 |
| `ZScore:rolling_std/close` | indicator | 0.784 | 0.801 | 0.782 | 0.828 | 1.00 | 159 | 0.01 | 1.00 |
| `base:bb_width` | indicator | 0.784 | 0.800 | 0.782 | 0.828 | 1.00 | 159 | 0.02 | 1.00 |
| `base:atr_pct_roll` | indicator | 0.779 | 0.821 | 0.767 | 0.880 | 1.00 | 83 | 0.02 | 0.61 |
| `VBM:vbm` | indicator | 0.769 | 0.790 | 0.751 | 0.871 | 1.00 | 83 | 0.04 | 0.73 |
| `LWBreakout:level_pct_diff` | indicator | 0.761 | 0.787 | 0.750 | 0.869 | 1.00 | 83 | 0.02 | 0.74 |
| `LWBreakout:atr/close` | indicator | 0.761 | 0.787 | 0.750 | 0.869 | 1.00 | 83 | 0.02 | 0.74 |
| `LWRevert:atr/close` | indicator | 0.761 | 0.787 | 0.750 | 0.869 | 1.00 | 83 | 0.02 | 0.74 |

### A1:288 (all bars): persistence baseline AUC 0.498

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:DCBreakout:dc_upper` | price_level | 0.390 | 0.376 | 0.390 | 0.410 | 0.80 | 31 | -0.00 | 0.08 |
| `btc:DCRevert:dc_upper` | price_level | 0.390 | 0.376 | 0.390 | 0.410 | 0.80 | 31 | -0.00 | 0.08 |
| `btc:Ehlers_DC:dc_upper` | price_level | 0.390 | 0.376 | 0.390 | 0.409 | 0.80 | 31 | 0.00 | 0.08 |
| `btc:BBBreakout:bb_upperband` | price_level | 0.391 | 0.373 | 0.391 | 0.409 | 0.80 | 31 | -0.00 | 0.08 |
| `btc:BBRevert:bb_upperband` | price_level | 0.391 | 0.373 | 0.391 | 0.409 | 0.80 | 31 | -0.00 | 0.08 |
| `btc:base:bb_upperband` | price_level | 0.390 | 0.373 | 0.390 | 0.409 | 0.80 | 31 | -0.00 | 0.08 |
| `btc:Ehlers_KC:kc_upper` | price_level | 0.390 | 0.376 | 0.390 | 0.409 | 0.80 | 31 | 0.00 | 0.08 |
| `btc:KCBreakout:kc_upper` | price_level | 0.390 | 0.374 | 0.390 | 0.409 | 0.80 | 31 | -0.00 | 0.08 |

### A2:288 (all bars): persistence baseline AUC 0.297

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `ni:ni_ms_br` | indicator | 0.314 | 0.347 | 0.284 | – | 0.79 | 154 | 0.17 | 0.06 |
| `btc:Ehlers_DC:dc_lower` | price_level | 0.418 | 0.475 | 0.304 | – | 0.57 | 31 | -0.00 | 0.07 |
| `btc:DCBreakout:dc_lower` | price_level | 0.417 | 0.475 | 0.306 | – | 0.57 | 31 | -0.00 | 0.07 |
| `btc:DCRevert:dc_lower` | price_level | 0.417 | 0.475 | 0.306 | – | 0.57 | 31 | -0.00 | 0.07 |
| `base:doy_sin` | calendar | 0.428 | 0.510 | 0.396 | – | 0.64 | 21 | -0.00 | 0.07 |
| `btc:base:doy_sin` | calendar | 0.428 | 0.510 | 0.396 | – | 0.64 | 21 | -0.00 | 0.07 |
| `btc:BBBreakout:bb_lowerband` | price_level | 0.422 | 0.477 | 0.307 | – | 0.57 | 31 | -0.00 | 0.07 |
| `btc:BBRevert:bb_lowerband` | price_level | 0.422 | 0.477 | 0.307 | – | 0.57 | 31 | -0.00 | 0.07 |

### B-btc:288 (all bars): persistence baseline AUC 0.479

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:Ehlers_KC:kc_upper` | price_level | 0.314 | 0.297 | 0.323 | 0.457 | 1.00 | 31 | 0.00 | 0.08 |
| `btc:LWRevert:sell_level` | price_level | 0.314 | 0.298 | 0.324 | 0.457 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:Ehlers_LW:lw_cycle_sell` | price_level | 0.314 | 0.297 | 0.324 | 0.457 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:DCBreakout:dc_upper` | price_level | 0.314 | 0.301 | 0.322 | 0.457 | 1.00 | 31 | -0.00 | 0.08 |
| `btc:DCRevert:dc_upper` | price_level | 0.314 | 0.301 | 0.322 | 0.457 | 1.00 | 31 | -0.00 | 0.08 |
| `btc:Ehlers_DC:dc_upper` | price_level | 0.315 | 0.300 | 0.323 | 0.457 | 1.00 | 31 | 0.00 | 0.08 |
| `btc:LWBreakout:buy_level` | price_level | 0.315 | 0.298 | 0.325 | 0.457 | 1.00 | 31 | 0.01 | 0.08 |
| `btc:Ehlers_LW:lw_trend_buy` | price_level | 0.315 | 0.297 | 0.325 | 0.457 | 1.00 | 31 | 0.01 | 0.08 |

### B-pair:288 (all bars): persistence baseline AUC 0.462

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:Ehlers_KC:kc_upper` | price_level | 0.382 | 0.414 | 0.376 | 0.477 | 0.87 | 31 | 0.00 | 0.08 |
| `btc:DCBreakout:dc_upper` | price_level | 0.382 | 0.416 | 0.377 | 0.478 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:DCRevert:dc_upper` | price_level | 0.382 | 0.416 | 0.377 | 0.478 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:KCBreakout:kc_upper` | price_level | 0.382 | 0.415 | 0.377 | 0.478 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:KCRevert:kc_upper` | price_level | 0.382 | 0.415 | 0.377 | 0.478 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:Ehlers_DC:dc_upper` | price_level | 0.382 | 0.416 | 0.377 | 0.478 | 0.87 | 31 | 0.00 | 0.08 |
| `btc:BBBreakout:bb_upperband` | price_level | 0.383 | 0.415 | 0.378 | 0.478 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:BBRevert:bb_upperband` | price_level | 0.383 | 0.415 | 0.378 | 0.478 | 0.87 | 31 | -0.00 | 0.08 |

### C-btc:288 (all bars): persistence baseline AUC 0.755

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:base:atr_pct` | indicator | 0.763 | 0.753 | 0.796 | 0.859 | 0.93 | 83 | 0.01 | 0.70 |
| `btc:base:atr_pct_roll` | indicator | 0.751 | 0.743 | 0.790 | 0.856 | 0.93 | 83 | 0.02 | 0.61 |
| `btc:VBM:vbm` | indicator | 0.748 | 0.740 | 0.758 | 0.836 | 1.00 | 83 | 0.04 | 0.73 |
| `btc:LWBreakout:atr/close` | indicator | 0.746 | 0.737 | 0.758 | 0.836 | 1.00 | 83 | 0.02 | 0.74 |
| `btc:LWBreakout:level_pct_diff` | indicator | 0.746 | 0.737 | 0.758 | 0.836 | 1.00 | 83 | 0.02 | 0.74 |
| `btc:LWRevert:atr/close` | indicator | 0.746 | 0.737 | 0.758 | 0.836 | 1.00 | 83 | 0.02 | 0.74 |
| `btc:Ehlers_LW:lw_atr/close` | indicator | 0.743 | 0.736 | 0.755 | 0.833 | 1.00 | 83 | 0.02 | 0.77 |
| `base:doy_cos` | calendar | 0.469 | 0.369 | 0.481 | 0.342 | 0.57 | 85 | 0.03 | 0.16 |

### C-pair:288 (all bars): persistence baseline AUC 0.917

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `ZScore:rolling_std/close` | indicator | 0.755 | 0.758 | 0.755 | 0.811 | 1.00 | 159 | 0.01 | 1.00 |
| `base:bb_width` | indicator | 0.755 | 0.758 | 0.755 | 0.811 | 1.00 | 159 | 0.02 | 1.00 |
| `base:atr_pct` | indicator | 0.753 | 0.787 | 0.712 | 0.878 | 1.00 | 83 | 0.01 | 0.70 |
| `base:atr_pct_roll` | indicator | 0.752 | 0.790 | 0.713 | 0.876 | 1.00 | 83 | 0.02 | 0.61 |
| `VBM:vbm` | indicator | 0.715 | 0.744 | 0.681 | 0.861 | 1.00 | 83 | 0.04 | 0.73 |
| `LWBreakout:atr/close` | indicator | 0.711 | 0.738 | 0.673 | 0.860 | 1.00 | 83 | 0.02 | 0.74 |
| `LWRevert:atr/close` | indicator | 0.711 | 0.738 | 0.673 | 0.860 | 1.00 | 83 | 0.02 | 0.74 |
| `LWBreakout:level_pct_diff` | indicator | 0.711 | 0.738 | 0.673 | 0.860 | 1.00 | 83 | 0.02 | 0.74 |

### A1:672 (all bars): persistence baseline AUC 0.462

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:BBBreakout:bb_upperband` | price_level | 0.302 | 0.297 | 0.341 | 0.424 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:BBRevert:bb_upperband` | price_level | 0.302 | 0.297 | 0.341 | 0.424 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:base:bb_upperband` | price_level | 0.302 | 0.297 | 0.342 | 0.424 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:DCBreakout:dc_upper` | price_level | 0.303 | 0.298 | 0.342 | 0.424 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:DCRevert:dc_upper` | price_level | 0.303 | 0.298 | 0.342 | 0.424 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:Ehlers_DC:dc_upper` | price_level | 0.303 | 0.297 | 0.343 | 0.424 | 0.87 | 31 | 0.00 | 0.08 |
| `btc:Ehlers_KC:kc_upper` | price_level | 0.304 | 0.297 | 0.338 | 0.424 | 0.87 | 31 | 0.00 | 0.08 |
| `btc:KCBreakout:kc_upper` | price_level | 0.304 | 0.298 | 0.338 | 0.424 | 0.87 | 31 | -0.00 | 0.08 |

### A2:672 (all bars): persistence baseline AUC 0.157

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `base:doy_sin` | calendar | 0.516 | 0.502 | 0.516 | – | 0.57 | 21 | -0.00 | 0.07 |
| `btc:base:doy_sin` | calendar | 0.516 | 0.502 | 0.516 | – | 0.57 | 21 | -0.00 | 0.07 |
| `base:doy_cos` | calendar | 0.525 | 0.502 | 0.525 | – | 0.50 | 85 | 0.03 | 0.16 |
| `btc:base:doy_cos` | calendar | 0.525 | 0.502 | 0.525 | – | 0.50 | 85 | 0.03 | 0.16 |
| `ni:ni_ms_br` | indicator | 0.397 | 0.360 | 0.429 | – | 0.79 | 154 | 0.17 | 0.06 |
| `btc:WOBV:wobv_raw` | non_stationary | 0.592 | 0.645 | 0.512 | – | 0.57 | 36 | 0.00 | 0.09 |
| `btc:Ehlers_KC:kc_upper` | price_level | 0.452 | 0.509 | 0.349 | – | 0.57 | 31 | 0.00 | 0.08 |
| `btc:KCBreakout:kc_upper` | price_level | 0.453 | 0.509 | 0.350 | – | 0.57 | 31 | -0.00 | 0.08 |

### B-btc:672 (all bars): persistence baseline AUC 0.397

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:LWRevert:buy_level` | price_level | 0.218 | 0.207 | 0.266 | 0.428 | 1.00 | 31 | 0.01 | 0.07 |
| `btc:Ehlers_LW:lw_cycle_buy` | price_level | 0.219 | 0.207 | 0.266 | 0.428 | 1.00 | 31 | 0.01 | 0.07 |
| `btc:Ehlers_LW:lw_trend_sell` | price_level | 0.220 | 0.208 | 0.265 | 0.428 | 1.00 | 31 | 0.01 | 0.07 |
| `btc:LWBreakout:sell_level` | price_level | 0.221 | 0.208 | 0.265 | 0.428 | 1.00 | 31 | 0.01 | 0.07 |
| `btc:base:bb_upperband` | price_level | 0.222 | 0.213 | 0.262 | 0.429 | 1.00 | 31 | -0.00 | 0.08 |
| `btc:TEMACross:tema_long` | price_level | 0.222 | 0.209 | 0.265 | 0.429 | 1.00 | 31 | 0.00 | 0.08 |
| `btc:Ehlers_KC:kc_lower` | price_level | 0.222 | 0.207 | 0.266 | 0.428 | 1.00 | 31 | 0.00 | 0.07 |
| `btc:BBBreakout:bb_upperband` | price_level | 0.222 | 0.214 | 0.261 | 0.430 | 1.00 | 31 | -0.00 | 0.08 |

### B-pair:672 (all bars): persistence baseline AUC 0.416

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `btc:LWRevert:sell_level` | price_level | 0.361 | 0.406 | 0.299 | 0.453 | 0.87 | 31 | 0.01 | 0.08 |
| `btc:Ehlers_LW:lw_cycle_sell` | price_level | 0.361 | 0.406 | 0.299 | 0.453 | 0.87 | 31 | 0.01 | 0.08 |
| `btc:Ehlers_LW:lw_trend_buy` | price_level | 0.361 | 0.405 | 0.298 | 0.453 | 0.87 | 31 | 0.01 | 0.08 |
| `btc:LWBreakout:buy_level` | price_level | 0.361 | 0.405 | 0.298 | 0.453 | 0.87 | 31 | 0.01 | 0.08 |
| `btc:Ehlers_KC:kc_upper` | price_level | 0.361 | 0.408 | 0.300 | 0.454 | 0.87 | 31 | 0.00 | 0.08 |
| `btc:TEMACross:tema_short` | price_level | 0.361 | 0.405 | 0.298 | 0.453 | 0.87 | 31 | 0.01 | 0.08 |
| `btc:KCBreakout:kc_upper` | price_level | 0.362 | 0.408 | 0.300 | 0.454 | 0.87 | 31 | -0.00 | 0.08 |
| `btc:KCRevert:kc_upper` | price_level | 0.362 | 0.408 | 0.300 | 0.454 | 0.87 | 31 | -0.00 | 0.08 |

### C-btc:672 (all bars): persistence baseline AUC 0.777

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `base:doy_sin` | calendar | 0.835 | 0.835 | 0.790 | 0.607 | 0.91 | 21 | -0.00 | 0.07 |
| `btc:base:doy_sin` | calendar | 0.835 | 0.835 | 0.790 | 0.597 | 0.91 | 21 | -0.00 | 0.07 |
| `base:doy_cos` | calendar | 0.406 | 0.247 | 0.426 | 0.316 | 0.64 | 85 | 0.03 | 0.16 |
| `btc:base:doy_cos` | calendar | 0.406 | 0.247 | 0.426 | 0.285 | 0.64 | 85 | 0.03 | 0.16 |
| `btc:PSAR:ema200` | price_level | 0.347 | 0.597 | 0.152 | 0.729 | 0.55 | 31 | -0.00 | 0.07 |
| `btc:base:ema200` | price_level | 0.347 | 0.597 | 0.152 | 0.729 | 0.55 | 31 | -0.00 | 0.07 |
| `btc:DCBreakout:dc_lower` | price_level | 0.380 | 0.495 | 0.137 | 0.720 | 0.73 | 31 | -0.00 | 0.07 |
| `btc:DCRevert:dc_lower` | price_level | 0.380 | 0.495 | 0.137 | 0.720 | 0.73 | 31 | -0.00 | 0.07 |

### C-pair:672 (all bars): persistence baseline AUC 0.883

| column | tag | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |
|---|---|---|---|---|---|---|---|---|---|
| `ZScore:rolling_std/close` | indicator | 0.720 | 0.736 | 0.714 | 0.789 | 1.00 | 159 | 0.01 | 1.00 |
| `base:bb_width` | indicator | 0.720 | 0.736 | 0.714 | 0.789 | 1.00 | 159 | 0.02 | 1.00 |
| `base:atr_pct_roll` | indicator | 0.692 | 0.740 | 0.690 | 0.863 | 1.00 | 83 | 0.02 | 0.61 |
| `base:atr_pct` | indicator | 0.690 | 0.738 | 0.689 | 0.863 | 1.00 | 83 | 0.01 | 0.70 |
| `VBM:vbm` | indicator | 0.671 | 0.698 | 0.661 | 0.846 | 1.00 | 83 | 0.04 | 0.73 |
| `LWBreakout:atr/close` | indicator | 0.668 | 0.693 | 0.654 | 0.844 | 1.00 | 83 | 0.02 | 0.74 |
| `LWRevert:atr/close` | indicator | 0.668 | 0.693 | 0.654 | 0.844 | 1.00 | 83 | 0.02 | 0.74 |
| `LWBreakout:level_pct_diff` | indicator | 0.668 | 0.693 | 0.654 | 0.844 | 1.00 | 83 | 0.02 | 0.74 |

## SC-004 answers

**(a) Columns that beat the width sort on candidate bars in both halves (E0 agrees):** 122 columns in 14 redundancy groups: `CCI:cci`, `DCRevert:dc_upper/close (+8 in group)`, `Ehlers_DC:dc_mid/close (+2 in group)`, `HTDCP:cg (+11 in group)`, `HTDCP:itrend/close (+44 in group)`, `Ichimoku:tenkan_sen/close (+1 in group)`, `LWRevert:sell_level/close (+2 in group)`, `SAR:sar/close (+2 in group)`, `VBM:vbm (+6 in group)`, `WILLR:wobv (+2 in group)`, `ZScore:rolling_std/close`, `base:fast_diff`, `base:macdhist/close (+3 in group)`, `raw8:ret4 (+27 in group)`

**(b) Columns with residual strength beyond bb_width + guard_metric in both halves (E0 agrees):** 282 columns in 52 redundancy groups: `Aroon:aroondown`, `Aroon:aroonup`, `BOP:bop (+1 in group)`, `CCI:cci (+7 in group)`, `CMO:cmo (+67 in group)`, `COPP:copp`, `DCRevert:dc_upper/close (+8 in group)`, `Ehlers_DC:dc_mid/close (+2 in group)`, `Ehlers_DYMI:dymi_fisher`, `Ehlers_Fisher:fisher_cci (+4 in group)`, `Ehlers_Fisher:fisher_dsp`, `Ehlers_Fisher:fisher_mfi_diff`, `Ehlers_Fisher:fisher_roc_diff (+1 in group)`, `Ehlers_Fisher:fisher_stoch`, `Ehlers_ROC:roc_fisher (+2 in group)`, `Ehlers_ROC:roc_slope`, `Ehlers_ind_DYMI:dymi_slope (+4 in group)`, `HTDCP:cg (+23 in group)`, `HTDCP:cg_slope (+23 in group)`, `HTDCP:ebsw (+11 in group)`, `HTDCP:ebsw_slope (+11 in group)`, `HTDCP:regime_score (+11 in group)`, `Ichimoku:tenkan_sen/close (+1 in group)`, `LWRevert:sell_level/close (+3 in group)`, `MI:mi (+2 in group)`, `PZO:pzo`, `SAR:sar/close (+2 in group)`, `STC:stc`, `TEMACross:tema_short/close`, `UO:uo`, `VBM:vbm (+6 in group)`, `WILLR:wobv (+6 in group)`, `WTO:wt2`, `ZScore:rolling_std/close`, `base:atr_norm`, `base:di_diff_scaled`, `base:fast_diff`, `base:fisher_ss`, `base:gain_norm`, `base:macdhist_norm (+4 in group)`, `base:macdsignal_norm`, `base:roc (+1 in group)`, `base:spread_ma`, `base:vwap_pos (+1 in group)`, `ni:ni_ms_bv`, `ni:ni_ms_vr`, `ni:ni_of_tbu`, `raw8:log_hc`, `raw8:ret1`, `raw8:ret4 (+28 in group)`, `raw_buy:WILLR`, `raw_sell:VBM (+1 in group)`

*Tagged columns that met (a) or (b) but are not comparable across time or pairs (tags.json):*
  - (a) price_scaled: `HTDCP:dsp (+11 in group)`
  - (b) price_scaled: `APO:apo`, `HTDCP:dsp (+11 in group)`, `HTDCP:ema_slope (+11 in group)`, `base:macdhist (+2 in group)`

**(c) Signal pairs that survive late confirmation (E0 agrees), per target:**
  - A1:288 [all]: `raw_buy:CCI AND raw_buy:KST`, `raw_buy:BBRevert AND raw_buy:KST`, `raw_buy:BBBreakout:tm0 AND raw_buy:KST`, `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:COPP AND raw_buy:WTO`, `raw_buy:CCI AND raw_buy:TRIX`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:CCI AND raw_buy:PPO`, `raw_buy:Ehlers_LW:cycle AND raw_buy:KST`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:EhlersBase:cycle AND raw_buy:PPO`
  - A1:288 [candidate]: `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:ROC AND raw_buy:TEMACross`, `raw_buy:TRIX AND raw_buy:ZScore`, `raw_buy:BBBreakout:tm0 AND raw_buy:KST`, `raw_buy:BBRevert AND raw_buy:KST`
  - A1:48 [all]: `raw_buy:EhlersBase:cycle AND raw_buy:KST`, `raw_buy:ROC AND raw_buy:STC`, `raw_buy:CCI AND raw_buy:KST`
  - A1:48 [candidate]: `raw_buy:COPP AND raw_buy:STC`, `raw_buy:COPP AND raw_buy:ZScore`, `raw_buy:ROC AND raw_buy:STC`, `raw_buy:KST AND raw_buy:STC`
  - A1:672 [all]: `raw_buy:COPP AND raw_buy:WFRACTAL`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:BOP AND raw_buy:TEMACross`, `raw_buy:EWO AND raw_buy:HTSINE`
  - A1:672 [candidate]: `raw_buy:CCI AND raw_buy:KST`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:CCI AND raw_buy:TRIX`
  - B-btc:288 [all]: `raw_buy:ROC AND raw_buy:TEMACross`, `raw_buy:HTSINE AND raw_buy:KST`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:COPP AND raw_buy:TEMACross`, `raw_buy:MI AND raw_buy:ROC`
  - B-btc:288 [candidate]: `raw_buy:ROC AND raw_buy:TEMACross`, `raw_buy:HTSINE AND raw_buy:KST`, `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:COPP AND raw_buy:TEMACross`, `raw_buy:CCI AND raw_buy:TRIX`, `raw_buy:CCI AND raw_buy:COPP`
  - B-btc:48 [all]: `raw_buy:CCI AND raw_buy:KST`, `raw_buy:CCI AND raw_buy:COPP`
  - B-pair:288 [all]: `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:CCI AND raw_buy:PPO`, `raw_buy:BBRevert AND raw_buy:KST`, `raw_buy:BBBreakout:tm0 AND raw_buy:KST`, `raw_buy:COPP AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:ZScore`, `raw_buy:ROC AND raw_buy:UO`, `raw_buy:TRIX AND raw_buy:ZScore`
  - B-pair:288 [candidate]: `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:VFI AND raw_buy:WTO`, `raw_buy:TRIX AND raw_buy:ZScore`, `raw_buy:KST AND raw_buy:UO`
  - B-pair:48 [all]: `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:CMO AND raw_buy:PVO`, `raw_buy:COPP AND raw_buy:WTO`, `raw_buy:CCI AND raw_buy:TRIX`, `raw_buy:COPP AND raw_buy:LWRevert`, `raw_buy:TRIX AND raw_buy:ZScore`, `raw_buy:COPP AND raw_buy:UO`, `raw_buy:CCI AND raw_buy:PPO`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:CMO AND raw_buy:TEMACross`, `raw_buy:PPO AND raw_buy:ZScore`
  - B-pair:48 [candidate]: `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:FVE AND raw_buy:ZScore`
  - B-pair:672 [all]: `raw_buy:KST AND raw_buy:VBM`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:CCI AND raw_buy:PPO`
  - C-btc:288 [all]: `raw_buy:COPP AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:ZScore`, `raw_buy:PPO AND raw_buy:ZScore`, `raw_buy:BOP AND raw_buy:PPO`, `raw_buy:COPP AND raw_buy:Ehlers_LW:cycle`, `raw_buy:CCI AND raw_buy:PPO`, `raw_buy:HTTREND AND raw_buy:ZScore`, `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:HTTREND AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:Ehlers_Fisher:trend`, `raw_buy:HTTREND AND raw_buy:TSI`, `raw_buy:HTTREND AND raw_buy:VFI`, `raw_buy:BOP AND raw_buy:TSI`
  - C-btc:288 [candidate]: `raw_buy:CCI AND raw_buy:KST`, `raw_buy:Ehlers_LW:cycle AND raw_buy:KST`, `raw_buy:EMV AND raw_buy:LWRevert`, `raw_buy:PPO AND raw_buy:ZScore`, `raw_buy:LWRevert AND raw_buy:TRIX`
  - C-btc:48 [all]: `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:COPP AND raw_buy:ZScore`, `raw_buy:BBRevert AND raw_buy:KST`, `raw_buy:BBBreakout:tm0 AND raw_buy:KST`, `raw_buy:PPO AND raw_buy:ZScore`, `raw_buy:COPP AND raw_buy:LWRevert`, `raw_buy:TRIX AND raw_buy:ZScore`, `raw_buy:BOP AND raw_buy:COPP`, `raw_buy:COPP AND raw_buy:HTTREND`, `raw_buy:ROC AND raw_buy:STC`, `raw_buy:Ehlers_LW:cycle AND raw_buy:KST`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:BOP AND raw_buy:PPO`, `raw_buy:HTTREND AND raw_buy:PPO`, `raw_buy:BOP AND raw_buy:TRIX`, `raw_buy:HTTREND AND raw_buy:TRIX`, `raw_buy:Ehlers_BB:cycle AND raw_buy:TRIX`
  - C-btc:48 [candidate]: `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:BBBreakout:tm0 AND raw_buy:KST`, `raw_buy:BBRevert AND raw_buy:KST`, `raw_buy:CCI AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:HTTREND`, `raw_buy:HTTREND AND raw_buy:KST`, `raw_buy:BBBreakout:tm0 AND raw_buy:TRIX`, `raw_buy:BBRevert AND raw_buy:TRIX`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:Ehlers_BB:cycle AND raw_buy:TRIX`, `raw_buy:Ehlers_LW:cycle AND raw_buy:Ehlers_LW:trend`, `raw_buy:EMV AND raw_buy:Ehlers_BB:cycle`
  - C-pair:288 [all]: `raw_buy:COPP AND raw_buy:VBM`, `raw_buy:KST AND raw_buy:VBM`, `raw_buy:FVE AND raw_buy:ROC`, `raw_buy:EMACross AND raw_buy:PPO`, `raw_buy:COPP AND raw_buy:MI`, `raw_buy:COPP AND raw_buy:FVE`, `raw_buy:MI AND raw_buy:ROC`, `raw_buy:KST AND raw_buy:MI`, `raw_buy:PPO AND raw_buy:SMACross`, `raw_buy:Ehlers_Ichimoku:trend AND raw_buy:PPO`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:STC`, `raw_buy:ROC AND raw_buy:STC`
  - C-pair:288 [candidate]: `raw_buy:Ehlers_Ichimoku:trend AND raw_buy:VBM`, `raw_buy:Ichimoku AND raw_buy:MI`, `raw_buy:EMACross AND raw_buy:WTO`, `raw_buy:EhlersBase:cycle AND raw_buy:MI`, `raw_buy:EMACross AND raw_buy:PPO`
  - C-pair:48 [all]: `raw_buy:COPP AND raw_buy:VBM`, `raw_buy:KST AND raw_buy:VBM`, `raw_buy:COPP AND raw_buy:WTO`, `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:FVE AND raw_buy:ROC`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:MI`, `raw_buy:TRIX AND raw_buy:VBM`, `raw_buy:Ehlers_BB:cycle AND raw_buy:KST`, `raw_buy:PPO AND raw_buy:SMACross`, `raw_buy:Ehlers_Ichimoku:trend AND raw_buy:PPO`, `raw_buy:ROC AND raw_buy:STC`, `raw_buy:Ehlers_LW:cycle AND raw_buy:KST`, `raw_buy:MI AND raw_buy:ROC`, `raw_buy:EMACross AND raw_buy:PPO`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:TRIX AND raw_buy:ZScore`
  - C-pair:48 [candidate]: `raw_buy:KST AND raw_buy:VBM`, `raw_buy:KST AND raw_buy:ZScore`
  - C-pair:672 [all]: `raw_buy:COPP AND raw_buy:VBM`, `raw_buy:EMACross AND raw_buy:PPO`, `raw_buy:COPP AND raw_buy:MI`, `raw_buy:KST AND raw_buy:VBM`, `raw_buy:KST AND raw_buy:MI`, `raw_buy:FVE AND raw_buy:ROC`, `raw_buy:Ehlers_Ichimoku:trend AND raw_buy:PPO`, `raw_buy:MI AND raw_buy:ROC`, `raw_buy:COPP AND raw_buy:FVE`, `raw_buy:KCBreakout AND raw_buy:TRIX`, `raw_buy:TRIX AND raw_buy:VBM`
  - C-pair:672 [candidate]: `raw_buy:COPP AND raw_buy:VBM`, `raw_buy:EMACross AND raw_buy:PPO`, `raw_buy:Aroon AND raw_buy:MI`, `raw_buy:SMACross AND raw_buy:WTO`
  - label:buy [candidate]: `raw_buy:CCI AND raw_buy:COPP`, `raw_buy:CCI AND raw_buy:WTO`, `raw_buy:COPP AND raw_buy:WTO`, `raw_buy:CCI AND raw_buy:KST`, `raw_buy:KST AND raw_buy:ZScore`, `raw_buy:WTO AND raw_buy:ZScore`, `raw_buy:HTSINE AND raw_buy:ZScore`, `raw_buy:KST AND raw_buy:WTO`, `raw_buy:FVE AND raw_buy:ZScore`, `raw_buy:CCI AND raw_buy:MI`, `raw_buy:KST AND raw_buy:LWRevert`, `raw_buy:COPP AND raw_buy:MI`, `raw_buy:COPP AND raw_buy:VBM`, `raw_buy:SMACross AND raw_buy:ZScore`
  - label:sell [all]: `raw_sell:KST AND raw_sell:WILLR`, `raw_sell:COPP AND raw_sell:Ehlers_LW:cycle`, `raw_sell:KST AND raw_sell:SAR`, `raw_sell:MFI AND raw_sell:TRIX`, `raw_sell:COPP AND raw_sell:LWRevert`, `raw_sell:TRIX AND raw_sell:UO`, `raw_sell:TRIX AND raw_sell:VORTEX`, `raw_sell:TRIX AND raw_sell:WILLR`, `raw_sell:COPP AND raw_sell:KST`, `raw_sell:SAR AND raw_sell:TRIX`, `raw_sell:COPP AND raw_sell:VFI`, `raw_sell:BBBreakout:tm0 AND raw_sell:PPO`

  Early + late only (E0 fails or not computable): A1:288 [all]: 4; A1:288 [candidate]: 3; A1:48 [all]: 2; A1:48 [candidate]: 2; A1:672 [all]: 1; A2:288 [all]: 18; A2:288 [candidate]: 14; A2:48 [all]: 18; A2:48 [candidate]: 11; A2:672 [all]: 15; A2:672 [candidate]: 13; B-btc:288 [all]: 8; B-btc:288 [candidate]: 6; B-btc:48 [candidate]: 1; B-pair:288 [all]: 7; C-btc:288 [candidate]: 1; C-pair:288 [all]: 4; C-pair:672 [all]: 7; C-pair:672 [candidate]: 1; label:buy [candidate]: 1; label:sell [all]: 3

**(d) Multivariate models that beat the restricted model in both halves:** `tree [A1:288, all] (distant era not tested)`, `hgb [A1:288, all] (distant era not tested)`, `tree [A1:288, candidate] (distant era not tested)`, `hgb [A1:288, candidate] (distant era not tested)`, `hgb [A1:48, all] (distant era not tested)`, `hgb [A1:48, candidate] (distant era not tested)`, `hgb [A1:672, all] (distant era not tested)`, `tree [A1:672, candidate] (distant era not tested)`, `tree [A2:288, all] (distant era not tested)`, `hgb [A2:288, all] (distant era not tested)`, `tree [A2:288, candidate] (distant era not tested)`, `hgb [A2:288, candidate] (distant era not tested)`, `tree [A2:672, all] (distant era not tested)`, `tree [A2:672, candidate] (distant era not tested)`, `hgb [A2:672, candidate] (distant era not tested)`, `hgb [B-btc:288, all] (distant era not tested)`, `hgb [B-btc:48, all] (distant era not tested)`, `hgb [B-btc:48, candidate] (distant era not tested)`, `tree [B-btc:672, all] (distant era not tested)`, `hgb [B-btc:672, all] (distant era not tested)`, `hgb [B-btc:672, candidate] (distant era not tested)`, `hgb [B-pair:288, all] (distant era not tested)`, `hgb [B-pair:288, candidate] (distant era not tested)`, `hgb [B-pair:48, all] (distant era not tested)`, `hgb [B-pair:48, candidate] (distant era not tested)`, `hgb [B-pair:672, all] (distant era not tested)`, `hgb [B-pair:672, candidate] (distant era not tested)`, `tree [C-btc:288, candidate] (distant era not tested)`, `hgb [C-btc:288, candidate] (distant era not tested)`, `tree [C-btc:48, all] (distant era not tested)`, `hgb [C-btc:48, all] (distant era not tested)`, `tree [C-btc:48, candidate] (distant era not tested)`, `hgb [C-btc:48, candidate] (distant era not tested)`, `hgb [C-btc:672, candidate] (distant era not tested)`, `tree [C-pair:288, all] (distant era not tested)`, `hgb [C-pair:288, all] (distant era not tested)`, `tree [C-pair:288, candidate] (distant era not tested)`, `hgb [C-pair:288, candidate] (distant era not tested)`, `tree [C-pair:48, all] (distant era not tested)`, `hgb [C-pair:48, all] (distant era not tested)`, `tree [C-pair:48, candidate] (distant era not tested)`, `hgb [C-pair:48, candidate] (distant era not tested)`, `tree [C-pair:672, all] (distant era not tested)`, `hgb [C-pair:672, all] (distant era not tested)`, `tree [C-pair:672, candidate] (distant era not tested)`, `hgb [C-pair:672, candidate] (distant era not tested)`, `tree [label:buy, candidate] (distant era not tested)`, `hgb [label:buy, candidate] (distant era not tested)`, `hgb [label:sell, all] (distant era not tested)`

**(e) Regime columns/models that beat the persistence baseline in both halves:**
  - A1:288 [all]: `btc:VFI:vfi`
  - A1:288 [candidate]: `PSAR:ema200/close (+1 in group)`, `ni:ni_ms_br`
  - A1:48 [all]: `ni:ni_ms_br`
  - A1:48 [candidate]: `PSAR:ema200/close (+1 in group)`, `VBM:vbm (+5 in group)`, `VBM:vbm_zscore`, `base:atr_norm`, `base:bb_width (+1 in group)`, `base:spread_ma`, `hgb model (distant era not tested)`, `ni:ni_ms_br`, `ni:ni_ms_bv`, `ni:ni_ms_vr`
  - B-btc:288 [all]: `btc:VBM:vbm (+6 in group)`
  - B-btc:48 [all]: `btc:HTDCP:htdcp (+13 in group)`, `btc:Ichimoku:senkou_span_b/close (+1 in group)`, `btc:PSAR:ema200/close (+1 in group)`, `hgb model (distant era not tested)`, `ni:ni_ms_vr`
  - B-pair:288 [all]: `hgb model (distant era not tested)`, `ni:ni_ms_br`, `ni:ni_ms_vr`
  - B-pair:288 [candidate]: `ni:ni_ms_br`
  - B-pair:48 [all]: `CCI:cci (+4 in group)`, `CMO:cmo (+68 in group)`, `EMACross:ema_short/close (+1 in group)`, `Ehlers_DC:dc_mid/close (+2 in group)`, `Ichimoku:senkou_span_b/close (+3 in group)`, `Ichimoku:tenkan_sen/close (+1 in group)`, `KCRevert:kc_upper/close (+2 in group)`, `PSAR:ema200/close (+1 in group)`, `SAR:sar/close (+1 in group)`, `UO:uo`, `WILLR:wobv (+6 in group)`, `base:rsi_sma (+3 in group)`, `ni:ni_ms_vr`
  - B-pair:48 [candidate]: `CMO:cmo (+47 in group)`, `Ichimoku:senkou_span_b/close (+2 in group)`, `PSAR:ema200/close (+1 in group)`, `btc:LWRevert:atr/close (+3 in group)`, `hgb model (distant era not tested)`
  - C-btc:48 [all]: `btc:base:atr_pct (+1 in group)`, `hgb model (distant era not tested)`, `tree model (distant era not tested)`
  - C-btc:48 [candidate]: `hgb model (distant era not tested)`
  - C-pair:48 [all]: `hgb model (distant era not tested)`
  - C-pair:48 [candidate]: `hgb model (distant era not tested)`

  *Tagged columns that also beat the baseline (not counted):* A1:288 [all]: non_stationary 1, price_level 85; A1:48 [all]: price_level 93; A1:48 [candidate]: non_stationary 1, price_level 93; A1:672 [candidate]: non_stationary 1; B-btc:288 [all]: calendar 2, non_stationary 1, price_level 93; B-btc:288 [candidate]: non_stationary 1, price_level 93; B-btc:48 [all]: price_level 93; B-btc:48 [candidate]: non_stationary 1, price_level 93; B-btc:672 [all]: price_level 93; B-btc:672 [candidate]: non_stationary 1, price_level 93; B-pair:288 [all]: non_stationary 1, price_level 93; B-pair:288 [candidate]: price_level 93; B-pair:48 [all]: price_level 91, price_scaled 12; B-pair:672 [all]: non_stationary 1, price_level 93; B-pair:672 [candidate]: price_level 93; C-btc:288 [candidate]: calendar 2; C-btc:672 [all]: calendar 4; C-btc:672 [candidate]: calendar 4

**SCREEN (descriptive).** Association with a label or a regime decides nothing about acceptance (Principle III). Every candidate below needs a backtest in a later feature before any P&L claim (FR-025).
