## Stage: regime

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_bgo_mx`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_reg_volreg | +44.19 | 2526 | +33.51 | False | reversion exit counts moved {'force_exit': (20, 14), 'metric_overbought': (11, 8), 'roi': (20, 15), 'take_profit': (2695, 2024), 'trailing_stop_loss': (2479, 1637), 'unclog_12h': (1087, 675), 'unclog_1d': (59, 40)}; folds with different trades 15/15 -> LIVE |

chosen: `st_reg_volreg` (best early sum)

late (f08-f14) vs `st_bgo_mx`: **INCONCLUSIVE_ARM_LOSES**, delta -8.557234451677608 pp, CI [-20.596815539527316, 10.556850181647357] pp/yr, p 0.5259

(reported only) late vs the previous arm: **INCONCLUSIVE_ARM_LOSES**, delta 14.323943079505007 pp, CI [-2.5720929434569677, 19.184573724689784] pp/yr, p 0.1342

adopted: **no**

