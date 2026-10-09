# Leaderboard: phase 28

labels: tb_h12_u0.015_d0.015_cr1_pf0

Classification metrics against fixed tb_h12_u0.015_d0.015_cr1_pf0 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=1, LSTM_FAM=2, LSTM_FAM=2, LSTM_FAM=2, LSTM_FAM=2, LSTM_FAM=2, RIDGE=3, MAJORITY=1)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 21666484c86b | {} | raw8 | 16 | 1 | 0.759 ± NaN | 0.845 ± NaN | 0.845 ± NaN | 0.955 | 0.125 | 0.881 | 0.817 | 0.168 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | LSTM_FAM | f1baf072c9e6 | {"depth": 1, "hidden": 32, "input_norm": false} | raw8 | 32 | 2 | 0.231 ± 0.003 | 0.463 ± 0.004 | 0.475 ± 0.001 | 0.976 | 0.108 | 0.693 | NaN | 0.000 | 5411 | 2.868 | 0.004 | 21.7 | - |
| 3 | LSTM_FAM | d2fe0737a3c4 | {"depth": 2, "hidden": 16, "input_norm": false} | raw8 | 32 | 2 | 0.230 ± 0.001 | 0.464 ± 0.002 | 0.474 ± 0.004 | 0.975 | 0.114 | 0.693 | NaN | 0.000 | 3795 | 6.340 | 0.006 | 26.0 | - |
| 4 | LSTM_FAM | 563c42a2f80c | {"bidirectional": true, "depth": 2, "hidden": 32, "input_norm": false} | raw8 | 32 | 2 | 0.229 ± 0.001 | 0.462 ± 0.001 | 0.474 ± 0.003 | 0.972 | 0.101 | 0.692 | NaN | 0.000 | 27459 | 10.521 | 0.013 | 21.0 | - |
| 5 | LSTM_FAM | 477f5a46bd6f | {"depth": 2, "hidden": 64, "input_norm": false} | raw8 | 32 | 2 | 0.228 ± 0.002 | 0.460 ± 0.003 | 0.473 ± 0.003 | 0.976 | 0.107 | 0.692 | NaN | 0.000 | 52035 | 5.711 | 0.008 | 15.7 | - |
| 6 | LSTM_FAM | 6c157ee909e8 | {"depth": 3, "hidden": 32, "input_norm": false} | raw8 | 32 | 2 | 0.228 ± 0.001 | 0.462 ± 0.001 | 0.474 ± 0.003 | 0.978 | 0.106 | 0.691 | NaN | 0.000 | 22051 | 6.153 | 0.009 | 17.5 | - |
| 7 | RIDGE | de187e3fb32a | {} | raw8 | 16 | 3 | 0.168 ± 0.001 | 0.416 ± 0.002 | 0.417 ± 0.001 | 1.027 | 0.189 | 0.618 | NaN | 0.000 | 387 | 0.512 | 0.001 | 22.0 | baseline |
| 8 | MAJORITY | 3578723b8987 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.260 ± NaN | 0.333 ± NaN | 0.909 | 0.079 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 21666484c86b | 0.761 ± NaN | 0.744 ± NaN | 0.773 ± NaN | 0.845 ± NaN | 0.836 ± NaN | 0.853 ± NaN |
| LSTM_FAM | f1baf072c9e6 | 0.281 ± 0.003 | 0.178 ± 0.006 | 0.234 ± 0.002 | 0.489 ± 0.002 | 0.432 ± 0.008 | 0.470 ± 0.002 |
| LSTM_FAM | d2fe0737a3c4 | 0.277 ± 0.001 | 0.178 ± 0.001 | 0.235 ± 0.001 | 0.485 ± 0.002 | 0.434 ± 0.003 | 0.471 ± 0.001 |
| LSTM_FAM | 563c42a2f80c | 0.278 ± 0.001 | 0.176 ± 0.002 | 0.233 ± 0.001 | 0.486 ± 0.000 | 0.429 ± 0.002 | 0.470 ± 0.000 |
| LSTM_FAM | 477f5a46bd6f | 0.277 ± 0.001 | 0.173 ± 0.003 | 0.235 ± 0.001 | 0.485 ± 0.002 | 0.424 ± 0.005 | 0.472 ± 0.002 |
| LSTM_FAM | 6c157ee909e8 | 0.276 ± 0.003 | 0.175 ± 0.000 | 0.233 ± 0.000 | 0.485 ± 0.001 | 0.429 ± 0.000 | 0.471 ± 0.002 |
| RIDGE | de187e3fb32a | 0.224 ± 0.001 | 0.132 ± 0.000 | 0.147 ± 0.001 | 0.440 ± 0.003 | 0.403 ± 0.001 | 0.405 ± 0.002 |
| MAJORITY | 3578723b8987 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.275 ± NaN | 0.240 ± NaN | 0.264 ± NaN |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 21666484c86b | PERSISTENCE | 0.192 | 0.639 | 0.169 |
| 3578723b8987 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 477f5a46bd6f | LSTM_FAM | 0.218 | 0.544 | 0.238 |
| 563c42a2f80c | LSTM_FAM | 0.219 | 0.544 | 0.238 |
| 6c157ee909e8 | LSTM_FAM | 0.219 | 0.545 | 0.236 |
| d2fe0737a3c4 | LSTM_FAM | 0.221 | 0.554 | 0.224 |
| de187e3fb32a | RIDGE | 0.159 | 0.719 | 0.121 |
| f1baf072c9e6 | LSTM_FAM | 0.212 | 0.547 | 0.241 |
