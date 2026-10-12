# NNNC switch experiment — verdicts

*PROVISIONAL -- values were chosen on SimpleTrigger trades over these same 2023-26 folds; this tests carry-over to model entries, not out-of-sample value*

## C2: **NOT ADOPTED** (arm D + 1/3 replications)

| model | trades arm / ctl | Σ pp arm / ctl | early gain | late gain | paired (p) | risk Sharpe / maxDD | folds changed | verdict |
|---|---|---|---|---|---|---|---|---|
| ARO | 1374 / 1420 | +50.60 / +42.71 | +7.27 | +0.62 | UNCHANGED (0.135) | UNCHANGED / UNCHANGED | 15/15 | **PASS** |
| GBR | 2331 / 2395 | +44.06 / +43.65 | -3.53 | +3.95 | UNCHANGED (0.954) | UNCHANGED / UNCHANGED | 15/15 | **FAIL** (early gain) |
| GFM | 1969 / 2051 | +20.89 / +25.61 | -0.65 | -4.06 | UNCHANGED (0.319) | UNCHANGED / UNCHANGED | 15/15 | **FAIL** (early gain, late gain) |
| MLX | 374 / 378 | +29.01 / +25.32 | +3.38 | +0.31 | IMPROVED (0.042) | UNCHANGED / UNCHANGED | 15/15 | **PASS** |

Per half (reported only):

- ARO: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'roi': 177, 'take_profit': 586, 'trailing_stop_loss': 495, 'unclog_12h': 109, 'force_exit': 3, 'stop_loss': 3, 'unclog_1d': 1} vs {'roi': 37, 'take_profit': 824, 'trailing_stop_loss': 469, 'unclog_12h': 84, 'force_exit': 2, 'stop_loss': 3, 'unclog_1d': 1}
- GBR: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'roi': 191, 'take_profit': 1150, 'trailing_stop_loss': 808, 'unclog_12h': 164, 'unclog_1d': 2, 'force_exit': 9, 'stop_loss': 7} vs {'roi': 44, 'take_profit': 1316, 'trailing_stop_loss': 818, 'unclog_12h': 199, 'unclog_1d': 3, 'force_exit': 8, 'stop_loss': 7}
- GFM: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'roi': 137, 'take_profit': 866, 'trailing_stop_loss': 770, 'unclog_12h': 184, 'unclog_1d': 4, 'force_exit': 6, 'stop_loss': 1, 'metric_overbought': 1} vs {'roi': 31, 'take_profit': 1047, 'trailing_stop_loss': 769, 'unclog_12h': 193, 'unclog_1d': 4, 'force_exit': 5, 'stop_loss': 1, 'metric_overbought': 1}
- MLX: early IMPROVED, risk IMPROVED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'roi': 32, 'take_profit': 168, 'trailing_stop_loss': 98, 'unclog_12h': 24, 'model_exit ': 50, 'stop_loss': 1, 'metric_overbought': 1} vs {'take_profit': 195, 'trailing_stop_loss': 99, 'unclog_12h': 26, 'force_exit': 1, 'model_exit ': 45, 'roi': 10, 'stop_loss': 1, 'metric_overbought': 1}

## Gate: **NOT ADOPTED** (arm D + 0/3 replications)

| model | trades arm / ctl | Σ pp arm / ctl | early gain | late gain | paired (p) | risk Sharpe / maxDD | folds changed | verdict |
|---|---|---|---|---|---|---|---|---|
| ARO | 848 / 1422 | +35.09 / +42.11 | -0.81 | -6.21 | UNCHANGED (0.245) | UNCHANGED / UNCHANGED | 15/15 | **FAIL** (early gain, late gain) |
| GBR | 1246 / 2392 | +41.42 / +43.83 | +5.44 | -7.85 | UNCHANGED (0.808) | UNCHANGED / UNCHANGED | 15/15 | **FAIL** (late gain) |
| GFM | 1009 / 2044 | +21.93 / +25.23 | -3.21 | -0.08 | UNCHANGED (0.569) | UNCHANGED / UNCHANGED | 15/15 | **FAIL** (early gain, late gain) |
| MLX | 238 / 377 | +20.17 / +25.18 | -3.42 | -1.59 | UNCHANGED (0.123) | UNCHANGED / UNCHANGED | 15/15 | **FAIL** (early gain, late gain) |

Per half (reported only):

- ARO: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'roi': 25, 'take_profit': 526, 'trailing_stop_loss': 256, 'unclog_12h': 38, 'stop_loss': 3} vs {'roi': 36, 'take_profit': 827, 'trailing_stop_loss': 471, 'unclog_12h': 82, 'force_exit': 2, 'stop_loss': 3, 'unclog_1d': 1}
- GBR: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'take_profit': 742, 'trailing_stop_loss': 380, 'unclog_12h': 84, 'unclog_1d': 1, 'roi': 30, 'force_exit': 2, 'stop_loss': 7} vs {'roi': 44, 'take_profit': 1315, 'trailing_stop_loss': 817, 'unclog_12h': 198, 'unclog_1d': 3, 'force_exit': 8, 'stop_loss': 7}
- GFM: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'take_profit': 562, 'trailing_stop_loss': 354, 'unclog_12h': 73, 'roi': 16, 'unclog_1d': 1, 'force_exit': 2, 'stop_loss': 1} vs {'roi': 31, 'take_profit': 1045, 'trailing_stop_loss': 765, 'unclog_12h': 192, 'unclog_1d': 4, 'force_exit': 5, 'stop_loss': 1, 'metric_overbought': 1}
- MLX: early UNCHANGED, risk UNCHANGED/UNCHANGED; late UNCHANGED, risk UNCHANGED/UNCHANGED; exits {'take_profit': 128, 'unclog_12h': 15, 'trailing_stop_loss': 55, 'model_exit ': 30, 'roi': 8, 'stop_loss': 1, 'metric_overbought': 1} vs {'take_profit': 195, 'trailing_stop_loss': 99, 'unclog_12h': 26, 'force_exit': 1, 'model_exit ': 44, 'roi': 10, 'stop_loss': 1, 'metric_overbought': 1}

## Reported only: the 4096-bar startup alone (GateOff vs model control)

| model | Σ pp gateoff / ctl | early / late gain | folds changed | paired |
|---|---|---|---|---|
| ARO | +42.11 / +42.71 | -0.47 / -0.13 | 6/15 | UNCHANGED |
| GBR | +43.83 / +43.65 | +0.30 / -0.11 | 5/15 | INCONCLUSIVE_UNDERPOWERED |
| GFM | +25.23 / +25.61 | -0.46 / +0.07 | 6/15 | UNCHANGED |
| MLX | +25.18 / +25.32 | -0.15 / +0.00 | 1/15 | INCONCLUSIVE_UNDERPOWERED |

## Reported only: same-day native controls vs their earlier records

| model | record | trades_hash equal | trades now / record |
|---|---|---|---|
| GBR | `wf_gbr` | 15/15 | 2395 / 2395 |
| GFM | `wf_gfm` | 15/15 | 2051 / 2051 |
| MLX | `wf_nnnc` | 5/15 | 378 / 378 |
