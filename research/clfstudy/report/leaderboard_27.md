# Leaderboard: phase 27

labels: tb_h12_u0.015_d0.015_cr1_pf0

Classification metrics against fixed tb_h12_u0.015_d0.015_cr1_pf0 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=1, LSTM_FAM=2, RIDGE=3, MAJORITY=1)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 21666484c86b | {} | raw8 | 16 | 1 | 0.759 ± NaN | 0.845 ± NaN | 0.845 ± NaN | 0.955 | 0.125 | 0.881 | 0.817 | 0.168 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | LSTM_FAM | 9077bccb6895 | {"depth": 2, "hidden": 32, "input_norm": false} | raw8 | 128 | 2 | 0.236 ± 0.002 | 0.466 ± 0.002 | 0.479 ± 0.003 | 0.972 | 0.107 | 0.697 | NaN | 0.000 | 13731 | 54.591 | 0.035 | 19.5 | - |
| 3 | RIDGE | de187e3fb32a | {} | raw8 | 16 | 3 | 0.168 ± 0.001 | 0.416 ± 0.002 | 0.417 ± 0.001 | 1.027 | 0.189 | 0.618 | NaN | 0.000 | 387 | 0.512 | 0.001 | 22.0 | baseline |
| 4 | MAJORITY | 3578723b8987 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.260 ± NaN | 0.333 ± NaN | 0.909 | 0.079 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 21666484c86b | 0.761 ± NaN | 0.744 ± NaN | 0.773 ± NaN | 0.845 ± NaN | 0.836 ± NaN | 0.853 ± NaN |
| LSTM_FAM | 9077bccb6895 | 0.283 ± 0.002 | 0.181 ± 0.001 | 0.244 ± 0.002 | 0.488 ± 0.004 | 0.433 ± 0.002 | 0.477 ± 0.000 |
| RIDGE | de187e3fb32a | 0.224 ± 0.001 | 0.132 ± 0.000 | 0.147 ± 0.001 | 0.440 ± 0.003 | 0.403 ± 0.001 | 0.405 ± 0.002 |
| MAJORITY | 3578723b8987 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.275 ± NaN | 0.240 ± NaN | 0.264 ± NaN |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 21666484c86b | PERSISTENCE | 0.192 | 0.639 | 0.169 |
| 3578723b8987 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 9077bccb6895 | LSTM_FAM | 0.231 | 0.544 | 0.225 |
| de187e3fb32a | RIDGE | 0.159 | 0.719 | 0.121 |
