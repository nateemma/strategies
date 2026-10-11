## Stage: regime

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

previous arm: `st_bgo_mx`

| candidate | early sum (f00-f07) | lost to other family (early) | reversion early pp | degenerate | liveness |
|---|---|---|---|---|---|
| st_reg_volreg2 | +53.68 | 2376 | +44.67 | False | reversion exit counts moved {'force_exit': (14, 4), 'metric_overbought': (8, 2), 'roi': (15, 6), 'stop_loss': (11, 13), 'take_profit': (2024, 1253), 'trailing_stop_loss': (1637, 763), 'unclog_12h': (675, 321), 'unclog_1d': (40, 13)}; folds with different trades 15/15 -> LIVE |

chosen: `st_reg_volreg2` (best early sum)

late (f08-f14) vs `st_bgo_mx`: **INCONCLUSIVE_ARM_LOSES**, delta -8.80041529350269 pp, CI [-19.07668851761364, 8.209766846817347] pp/yr, p 0.4534

(reported only) late vs the previous arm: **INCONCLUSIVE_ARM_LOSES**, delta -0.24318947401552082 pp, CI [-12.259733084782932, 11.978755316772268] pp/yr, p 0.9943

adopted: **no**

