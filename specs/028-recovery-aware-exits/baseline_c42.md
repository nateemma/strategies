# 028 setup: C4+C2 on 2023–26 (T004) — EXPLORATORY, not out-of-sample

*Both parts were chosen after their late-half replay results were seen (027). Measured to fill rule (d) for C42 before the 2022 pre-registration. PROVISIONAL, gate 2. Nothing is adopted from this file.*

## C4+C2 grace 12 h + TP = 2.90 x ATR (`st_xpl_rev_c42`)

- arm: early +54.46, late +8.28 (reference early +50.23, late +4.20)
- (no 027 replay for the combination); arm gain: early +4.23, late +4.08
- early paired: **UNCHANGED**, delta 4.228004394289962, p 0.5809
- late paired: **UNCHANGED**, delta 4.0822563928749, CI [-7.8117746563592165, 12.417075248423764] pp/yr, p 0.6528
- late risk (624 days, realised daily pp): Sharpe **UNCHANGED** 0.71 vs 0.31 (delta +0.40, CI [-0.45, +1.25], p 0.378); max drawdown **UNCHANGED** 6.33 vs 9.19 (delta -2.86, CI [-10.44, +2.76], p 0.404)
- whole span: **UNCHANGED**, delta 8.309864080470732, p 0.4701
- liveness: folds with different trades 15/15; exit reasons arm {'roi': 120, 'take_profit': 1522, 'trailing_stop_loss': 787, 'unclog_12h': 456, 'unclog_1d': 22, 'stop_loss': 17, 'force_exit': 9, 'metric_overbought': 1} vs reference {'roi': 29, 'take_profit': 1715, 'trailing_stop_loss': 1160, 'unclog_12h': 444, 'stop_loss': 13, 'unclog_1d': 13, 'force_exit': 7, 'metric_overbought': 3}
