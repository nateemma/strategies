## Stage: coexistence

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_fam_combo`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_coex_cap3 | +16.62 | 1828 | +7.37 | False | pair_cap refusals 24204; recency-dropped rows 0 -> LIVE |
| st_coex_cap5 | +22.72 | 2320 | +12.58 | False | pair_cap refusals 9312; recency-dropped rows 0 -> LIVE |
| st_coex_cap7 | +34.33 | 2551 | +23.70 | False | pair_cap refusals 2630; recency-dropped rows 0 -> LIVE |
| st_coex_win4 | +33.24 | 2600 | +21.79 | False | pair_cap refusals 0; recency-dropped rows 21956 -> LIVE |
| st_coex_win16 | +27.10 | 2569 | +12.48 | False | pair_cap refusals 0; recency-dropped rows 74099 -> LIVE |
| st_coex_win48 | +23.19 | 2360 | +2.92 | False | pair_cap refusals 0; recency-dropped rows 128017 -> LIVE |

chosen: `st_coex_cap7` (best early sum)

late (f08-f14) vs `st_fam_combo`: **INCONCLUSIVE_ARM_LOSES**, delta -4.851912813495821 pp, CI [-11.271589007582417, 5.379679561204845] pp/yr, p 0.5004

adopted: **no**

## Stage: tune1

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_fam_combo`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_tune1_tp015 | +46.85 | 2559 | +23.76 | False | reversion exit counts moved {'force_exit': (20, 16), 'metric_overbought': (11, 7), 'roi': (20, 16), 'take_profit': (2699, 3671), 'trailing_stop_loss': (2485, 2307), 'unclog_12h': (1094, 684), 'unclog_1d': (60, 55)}; folds with different trades 15/15 -> LIVE |
| st_tune1_tp030 | +38.24 | 2723 | +18.47 | False | reversion exit counts moved {'force_exit': (20, 21), 'metric_overbought': (11, 13), 'roi': (20, 35), 'take_profit': (2699, 1875), 'trailing_stop_loss': (2485, 2600), 'unclog_12h': (1094, 1561)}; folds with different trades 15/15 -> LIVE |
| st_tune1_tpoff | +30.77 | 2670 | +20.28 | False | reversion exit counts moved {'force_exit': (20, 24), 'metric_overbought': (11, 52), 'roi': (20, 609), 'stop_loss': (11, 10), 'take_profit': (2699, 0), 'trailing_stop_loss': (2485, 2739), 'unclog_12h': (1094, 2590), 'unclog_1d': (60, 59)}; folds with different trades 15/15 -> LIVE |

chosen: `st_tune1_tp015` (best early sum)

late (f08-f14) vs `st_fam_combo`: **INCONCLUSIVE_ARM_LOSES**, delta -3.5851516154003975 pp, CI [-11.597272482128941, 7.237874091001415] pp/yr, p 0.6612

adopted: **no**

## Stage: tune2

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_fam_combo`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_tune2_u6 | +49.80 | 2603 | +32.33 | False | reversion exit counts moved {'force_exit': (20, 10), 'metric_overbought': (11, 5), 'roi': (20, 15), 'take_profit': (2699, 1901), 'trailing_stop_loss': (2485, 2045), 'unclog_12h': (1094, 2418), 'unclog_1d': (60, 45)}; folds with different trades 15/15 -> LIVE |
| st_tune2_u24 | +33.77 | 2663 | +22.28 | False | reversion exit counts moved {'force_exit': (20, 21), 'metric_overbought': (11, 26), 'roi': (20, 23), 'take_profit': (2699, 3172), 'trailing_stop_loss': (2485, 2723), 'unclog_12h': (1094, 224), 'unclog_1d': (60, 181)}; folds with different trades 15/15 -> LIVE |

chosen: `st_tune2_u6` (best early sum)

late (f08-f14) vs `st_fam_combo`: **INCONCLUSIVE_ARM_LOSES**, delta 0.16174091631424606 pp, CI [-7.845063912882403, 8.236157444851374] pp/yr, p 0.988

adopted: **no**

## Stage: tune3

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_fam_combo`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_tune3_f05 | +35.48 | 2655 | +23.61 | False | reversion exit counts moved {'take_profit': (2699, 2691), 'trailing_stop_loss': (2485, 2489), 'unclog_12h': (1094, 1093)}; folds with different trades 12/15 -> LIVE |
| st_tune3_f08 | +36.11 | 2659 | +24.40 | False | reversion exit counts moved {'take_profit': (2699, 2695), 'trailing_stop_loss': (2485, 2482), 'unclog_12h': (1094, 1096)}; folds with different trades 10/15 -> LIVE |

chosen: `st_tune3_f08` (best early sum)

late (f08-f14) vs `st_fam_combo`: **INCONCLUSIVE_ARM_LOSES**, delta 0.07078904943989064 pp, CI [-1.436385896174522, 1.6157128418611317] pp/yr, p 0.9473

adopted: **no**

## Stage: tune4

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_fam_combo`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_tune4_vcoff | +32.51 | 2670 | +20.19 | False | reversion exit counts moved {'force_exit': (20, 26), 'metric_overbought': (11, 14), 'roi': (20, 24), 'stop_loss': (11, 10), 'take_profit': (2699, 2589), 'trailing_stop_loss': (2485, 1772), 'unclog_12h': (1094, 1404), 'unclog_1d': (60, 201)}; folds with different trades 15/15 -> LIVE |

chosen: `st_tune4_vcoff` (best early sum)

late (f08-f14) vs `st_fam_combo`: **INCONCLUSIVE_ARM_LOSES**, delta 2.8340819011856433 pp, CI [-8.547228791201096, 11.69637107516107] pp/yr, p 0.7476

adopted: **no**

