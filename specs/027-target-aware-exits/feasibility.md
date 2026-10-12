# Feasibility: target-aware exits (027 US1)

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

## momentum (`st_bgo_mx`)

fidelity: null replay reproduces 0.996 of recorded exits (reason AND bar); stop model agrees on 0.996 of stop exits (median gap 0.000 pp)

| metric | early | late |
|---|---|---|
| trades | 3411 | 2745 |
| ambiguous | 9 | 2 |
| label_positive | 1864 | 1421 |
| a_stop_exits | 1836 | 1507 |
| a_touch_after_stop_h | 0.020 | 0.015 |
| a_touch_after_stop_maxhold | 0.465 | 0.451 |
| b_killed_share | 0.019 | 0.017 |
| b_killed_after_grace_share_of_killed | 1.000 | 1.000 |
| b_killed_after_grace_share_of_lp | 0.019 | 0.017 |
| b_mae_depth_p50_p80_p90_p95 | 0.007/0.020/0.029/0.039 | 0.007/0.019/0.029/0.040 |
| b_stop_at_touch_median | -0.270 | -0.270 |
| c_bars_to_touch_p25_p50_p75 | 3.000/9.000/21.000 | 3.000/10.000/22.000 |
| c_open_past_horizon_no_touch | 609 | 509 |
| c_open_past_horizon_pp | -12.576 | -8.168 |
| d_touched_exited_below | 908 | 675 |
| d_giveback_mean | 0.033 | 0.033 |
| pp_recorded | 22.683 | 2.505 |

| coupling | value (early-chosen) | gain pp early / late | verdict |
|---|---|---|---|
| C1 | {"percentile": 80, "depth": -0.01995817875256361} | +0.00 / +0.00 | **NOT SUPPORTED** |
| C2 | null | — | **N/A** |
| C3 | {"action": "breakeven"} | +0.64 / -0.63 | **NOT SUPPORTED** |
| C4 | {"depth_choice": "base", "grace_level": null} | +16.94 / -2.44 | **NOT SUPPORTED** |
| C5 | {"lambda": 0.5} | -0.50 / -4.64 | **NOT SUPPORTED** |

all grid values, replay gain pp (early / late; reported, not used to choose):

- C1 `80`: +0.00 / +0.00
- C1 `90`: -0.12 / +1.21
- C1 conditions (early / late): [False, False] / [False, False]
- C3 `breakeven`: +0.64 / -0.63
- C3 `close`: -0.31 / -2.22
- C3 conditions (early / late): [True] / [False]
- C4 `base`: +16.94 / -2.44
- C4 `p95`: -25.14 / -10.34
- C4 conditions (early / late): [True, False, True] / [True, False, False]
- C5 `0.0`: -8.07 / -6.59
- C5 `0.5`: -0.50 / -4.64
- C5 conditions (early / late): [False] / [False]

null-replay agreement by recorded reason: `{"force_exit": {"count": 46, "mean": 1.0}, "metric_overbought": {"count": 641, "mean": 0.9969}, "roi": {"count": 1122, "mean": 0.9991}, "stop": {"count": 3353, "mean": 0.994}, "unclog_12h": {"count": 737, "mean": 1.0}, "unclog_1d": {"count": 257, "mean": 1.0}}`

## reversion (`st_book_rev2`)

fidelity: null replay reproduces 0.996 of recorded exits (reason AND bar); stop model agrees on 0.994 of stop exits (median gap 0.000 pp)

| metric | early | late |
|---|---|---|
| trades | 2092 | 1292 |
| ambiguous | 5 | 3 |
| label_positive | 1384 | 832 |
| a_stop_exits | 699 | 466 |
| a_touch_after_stop_h | 0.067 | 0.045 |
| a_touch_after_stop_maxhold | 0.634 | 0.575 |
| b_killed_share | 0.034 | 0.024 |
| b_killed_after_grace_share_of_killed | 1.000 | 0.900 |
| b_killed_after_grace_share_of_lp | 0.034 | 0.022 |
| b_mae_depth_p50_p80_p90_p95 | 0.009/0.025/0.037/0.048 | 0.007/0.022/0.034/0.049 |
| b_stop_at_touch_median | -0.270 | -0.270 |
| c_bars_to_touch_p25_p50_p75 | 3.000/9.000/21.000 | 3.000/8.000/19.000 |
| c_open_past_horizon_no_touch | 287 | 134 |
| c_open_past_horizon_pp | -2.184 | -1.681 |
| d_touched_exited_below | 264 | 142 |
| d_giveback_mean | 0.032 | 0.038 |
| pp_recorded | 50.235 | 4.196 |

| coupling | value (early-chosen) | gain pp early / late | verdict |
|---|---|---|---|
| C1 | {"percentile": 90, "depth": -0.03710764139923586} | +4.86 / +3.23 | **NOT SUPPORTED** |
| C2 | {"k": 2.8970711921574543} | +8.21 / +8.31 | **NOT SUPPORTED** |
| C3 | {"action": "breakeven"} | -0.82 / +0.26 | **NOT SUPPORTED** |
| C4 | {"depth_choice": "base", "grace_level": null} | +19.05 / +12.76 | **NOT SUPPORTED** |
| C5 | {"lambda": 0.0} | -4.32 / -2.77 | **NOT SUPPORTED** |

all grid values, replay gain pp (early / late; reported, not used to choose):

- C1 `80`: +2.03 / +0.06
- C1 `90`: +4.86 / +3.23
- C1 conditions (early / late): [False, True] / [False, True]
- C2 `k`: +8.21 / +8.31
- C2 conditions (early / late): [False, True] / [False, True]
- C3 `breakeven`: -0.82 / +0.26
- C3 `close`: -1.21 / +0.23
- C3 conditions (early / late): [False] / [True]
- C4 `base`: +19.05 / +12.76
- C4 `p95`: -12.18 / +6.85
- C4 conditions (early / late): [True, False, True] / [True, False, True]
- C5 `0.0`: -4.32 / -2.77
- C5 `0.5`: -4.71 / -2.64
- C5 conditions (early / late): [False] / [False]

null-replay agreement by recorded reason: `{"force_exit": {"count": 7, "mean": 1.0}, "metric_overbought": {"count": 3, "mean": 0.6667}, "roi": {"count": 29, "mean": 1.0}, "stop": {"count": 1173, "mean": 0.9949}, "take_profit": {"count": 1715, "mean": 1.0}, "unclog_12h": {"count": 444, "mean": 0.9887}, "unclog_1d": {"count": 13, "mean": 1.0}}`

overall: **STOP (none supported)** (exit 3)
