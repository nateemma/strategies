# 027 exploratory arms (option 1) — EXPLORATORY, not out-of-sample

*Chosen after the late-half replay results were seen (027 report). PROVISIONAL, gate 2. Nothing is adopted from this file.*

## C4 grace 12 h (`st_xpl_rev_c4`)

- arm: early +58.67, late +10.46 (reference early +50.23, late +4.20)
- replay predicted gain: early +19.05, late +12.76; arm gain: early +8.44, late +6.27
- early paired: **UNCHANGED**, delta 8.439609884106876, p 0.2019
- late paired: **UNCHANGED**, delta 6.267508753435851, CI [-3.565101266194407, 11.051967129208974] pp/yr, p 0.3182
- late risk (624 days, realised daily pp): Sharpe **UNCHANGED** 0.90 vs 0.31 (delta +0.59, CI [-0.25, +1.46], p 0.184); max drawdown **UNCHANGED** 7.46 vs 9.19 (delta -1.72, CI [-9.22, +2.42], p 0.393)
- whole span: **UNCHANGED**, delta 14.706509571761764, p 0.1146
- liveness: folds with different trades 15/15; exit reasons arm {'roi': 32, 'take_profit': 1514, 'trailing_stop_loss': 786, 'unclog_12h': 572, 'unclog_1d': 22, 'stop_loss': 17, 'force_exit': 7, 'metric_overbought': 3} vs reference {'roi': 29, 'take_profit': 1715, 'trailing_stop_loss': 1160, 'unclog_12h': 444, 'stop_loss': 13, 'unclog_1d': 13, 'force_exit': 7, 'metric_overbought': 3}

## C2 TP = 2.90 x ATR (`st_xpl_rev_c2`)

- arm: early +54.70, late +8.37 (reference early +50.23, late +4.20)
- replay predicted gain: early +8.21, late +8.31; arm gain: early +4.46, late +4.17
- early paired: **UNCHANGED**, delta 4.462524657925386, p 0.3839
- late paired: **UNCHANGED**, delta 4.171529515713098, CI [-5.773460705150049, 10.75171951472269] pp/yr, p 0.5606
- late risk (624 days, realised daily pp): Sharpe **UNCHANGED** 0.71 vs 0.31 (delta +0.40, CI [-0.22, +1.08], p 0.244); max drawdown **UNCHANGED** 6.31 vs 9.19 (delta -2.87, CI [-8.54, +1.37], p 0.292)
- whole span: **UNCHANGED**, delta 8.633648791534963, p 0.3326
- liveness: folds with different trades 15/15; exit reasons arm {'roi': 136, 'take_profit': 1682, 'trailing_stop_loss': 1145, 'unclog_12h': 349, 'stop_loss': 12, 'unclog_1d': 13, 'force_exit': 9, 'metric_overbought': 1} vs reference {'roi': 29, 'take_profit': 1715, 'trailing_stop_loss': 1160, 'unclog_12h': 444, 'stop_loss': 13, 'unclog_1d': 13, 'force_exit': 7, 'metric_overbought': 3}

## C1 stop no tighter than -3.71% (`st_xpl_rev_c1`)

- arm: early +53.52, late +6.07 (reference early +50.23, late +4.20)
- replay predicted gain: early +4.86, late +3.23; arm gain: early +3.29, late +1.88
- early paired: **UNCHANGED**, delta 3.2846668466807305, p 0.4272
- late paired: **UNCHANGED**, delta 1.8774423666761066, CI [-2.144050658701244, 4.312088492434031] pp/yr, p 0.5025
- late risk (624 days, realised daily pp): Sharpe **UNCHANGED** 0.46 vs 0.31 (delta +0.14, CI [-0.26, +0.53], p 0.458); max drawdown **UNCHANGED** 9.05 vs 9.19 (delta -0.14, CI [-4.54, +1.85], p 0.614)
- whole span: **UNCHANGED**, delta 5.161926766718743, p 0.3066
- liveness: folds with different trades 15/15; exit reasons arm {'roi': 29, 'take_profit': 1691, 'trailing_stop_loss': 900, 'unclog_12h': 589, 'unclog_1d': 44, 'stop_loss': 13, 'force_exit': 7, 'metric_overbought': 4} vs reference {'roi': 29, 'take_profit': 1715, 'trailing_stop_loss': 1160, 'unclog_12h': 444, 'stop_loss': 13, 'unclog_1d': 13, 'force_exit': 7, 'metric_overbought': 3}
