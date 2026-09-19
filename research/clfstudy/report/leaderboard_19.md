# Leaderboard: phase 19

labels: tb_h48_u0.02_d0.02_cr1_pf0

Classification metrics against fixed tb_h48_u0.02_d0.02_cr1_pf0 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=3, LSTM_FAM=2, MLP_FAM=2, RIDGE=3, TRANSFORMER_FAM=2, TCN_FAM=2, Wavenet=2, MAJORITY=3)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 5a7196f85699 | {} | ind24 | 16 | 3 | 0.850 ± 0.000 | 0.900 ± 0.000 | 0.900 ± 0.000 | 0.748 | 0.097 | 0.925 | 0.920 | 0.297 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | LSTM_FAM | 01596109699e | {"depth": 2, "hidden": 32, "input_norm": false} | ind24 | 16 | 2 | 0.169 ± 0.003 | 0.434 ± 0.002 | 0.438 ± 0.002 | 1.038 | 0.033 | 0.630 | 0.667 | 0.000 | 15779 | 2.610 | 0.004 | 20.2 | - |
| 3 | MLP_FAM | c1aa32b8dcea | {"width": 64} | ind24 | 16 | 2 | 0.168 ± 0.003 | 0.434 ± 0.002 | 0.437 ± 0.002 | 1.038 | 0.035 | 0.630 | NaN | 0.000 | 28995 | 0.472 | 0.001 | 17.7 | - |
| 4 | RIDGE | e700753b5111 | {} | ind24 | 16 | 3 | 0.161 ± 0.001 | 0.422 ± 0.002 | 0.432 ± 0.001 | 1.059 | 0.046 | 0.617 | NaN | 0.000 | 1155 | 0.531 | 0.001 | 25.6 | baseline |
| 5 | TRANSFORMER_FAM | 1e9e0b212d86 | {"d_model": 32} | ind24 | 16 | 2 | 0.159 ± 0.003 | 0.426 ± 0.005 | 0.431 ± 0.001 | 1.041 | 0.033 | 0.625 | NaN | 0.000 | 18243 | 2.239 | 0.005 | 16.2 | - |
| 6 | TCN_FAM | ae29f8c1fcd8 | {"channels": 64} | ind24 | 16 | 2 | 0.153 ± 0.006 | 0.421 ± 0.005 | 0.428 ± 0.003 | 1.045 | 0.038 | 0.621 | NaN | 0.000 | 29571 | 0.735 | 0.002 | 17.0 | - |
| 7 | Wavenet | 467d23581f7b | {} | ind24 | 16 | 2 | 0.151 ± 0.005 | 0.402 ± 0.014 | 0.422 ± 0.001 | 1.041 | 0.031 | 0.624 | NaN | 0.000 | 16227 | 0.953 | 0.002 | 14.0 | - |
| 8 | MAJORITY | bbae4b588c23 | {} | ind24 | 16 | 3 | 0.000 ± 0.000 | 0.151 ± 0.000 | 0.333 ± 0.000 | 1.108 | 0.072 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 5a7196f85699 | 0.859 ± 0.000 | 0.831 ± 0.000 | 0.860 ± 0.000 | 0.909 ± 0.000 | 0.882 ± 0.000 | 0.909 ± 0.000 |
| LSTM_FAM | 01596109699e | 0.217 ± 0.001 | 0.101 ± 0.002 | 0.189 ± 0.005 | 0.451 ± 0.001 | 0.405 ± 0.000 | 0.445 ± 0.004 |
| MLP_FAM | c1aa32b8dcea | 0.220 ± 0.001 | 0.098 ± 0.006 | 0.187 ± 0.001 | 0.452 ± 0.002 | 0.403 ± 0.003 | 0.446 ± 0.001 |
| RIDGE | e700753b5111 | 0.202 ± 0.002 | 0.102 ± 0.001 | 0.179 ± 0.001 | 0.437 ± 0.002 | 0.397 ± 0.001 | 0.433 ± 0.004 |
| TRANSFORMER_FAM | 1e9e0b212d86 | 0.207 ± 0.006 | 0.091 ± 0.001 | 0.178 ± 0.000 | 0.438 ± 0.010 | 0.399 ± 0.000 | 0.441 ± 0.003 |
| TCN_FAM | ae29f8c1fcd8 | 0.208 ± 0.004 | 0.084 ± 0.012 | 0.168 ± 0.003 | 0.443 ± 0.003 | 0.387 ± 0.008 | 0.431 ± 0.004 |
| Wavenet | 467d23581f7b | 0.208 ± 0.007 | 0.088 ± 0.005 | 0.158 ± 0.002 | 0.427 ± 0.014 | 0.374 ± 0.018 | 0.404 ± 0.012 |
| MAJORITY | bbae4b588c23 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.143 ± 0.000 | 0.138 ± 0.000 | 0.172 ± 0.000 |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 01596109699e | LSTM_FAM | 0.290 | 0.436 | 0.274 |
| 1e9e0b212d86 | TRANSFORMER_FAM | 0.285 | 0.431 | 0.284 |
| 467d23581f7b | Wavenet | 0.329 | 0.467 | 0.204 |
| 5a7196f85699 | PERSISTENCE | 0.341 | 0.361 | 0.297 |
| ae29f8c1fcd8 | TCN_FAM | 0.264 | 0.442 | 0.295 |
| bbae4b588c23 | MAJORITY | 0.668 | 0.332 | 0.000 |
| c1aa32b8dcea | MLP_FAM | 0.271 | 0.433 | 0.296 |
| e700753b5111 | RIDGE | 0.266 | 0.506 | 0.228 |
