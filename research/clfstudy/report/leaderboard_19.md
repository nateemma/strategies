# Leaderboard: phase 19

labels: tb_h48_u0.02_d0.02_cr1_pf0

Classification metrics against fixed tb_h48_u0.02_d0.02_cr1_pf0 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=3, LSTM_FAM=2, MLP_FAM=2, RIDGE=3, TRANSFORMER_FAM=2, TCN_FAM=2, Wavenet=2, MAJORITY=3)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 5a7196f85699 | {} | ind24 | 16 | 3 | 0.865 ± 0.000 | 0.913 ± 0.000 | 0.913 ± 0.000 | 0.636 | 0.083 | 0.934 | 0.908 | 0.183 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | LSTM_FAM | 01596109699e | {"depth": 2, "hidden": 32, "input_norm": false} | ind24 | 16 | 2 | 0.177 ± 0.003 | 0.425 ± 0.003 | 0.436 ± 0.001 | 1.018 | 0.051 | 0.641 | NaN | 0.000 | 15779 | 2.114 | 0.004 | 15.5 | - |
| 3 | MLP_FAM | c1aa32b8dcea | {"width": 64} | ind24 | 16 | 2 | 0.176 ± 0.002 | 0.420 ± 0.005 | 0.437 ± 0.001 | 1.026 | 0.059 | 0.641 | NaN | 0.000 | 28995 | 0.375 | 0.001 | 13.7 | - |
| 4 | RIDGE | e700753b5111 | {} | ind24 | 16 | 3 | 0.172 ± 0.003 | 0.425 ± 0.003 | 0.430 ± 0.004 | 1.042 | 0.074 | 0.627 | NaN | 0.000 | 1155 | 0.372 | 0.001 | 18.8 | baseline |
| 5 | TRANSFORMER_FAM | 1e9e0b212d86 | {"d_model": 32} | ind24 | 16 | 2 | 0.167 ± 0.004 | 0.415 ± 0.012 | 0.430 ± 0.001 | 1.019 | 0.042 | 0.634 | NaN | 0.000 | 18243 | 2.243 | 0.005 | 16.5 | - |
| 6 | TCN_FAM | ae29f8c1fcd8 | {"channels": 64} | ind24 | 16 | 2 | 0.161 ± 0.001 | 0.407 ± 0.005 | 0.425 ± 0.002 | 1.024 | 0.047 | 0.631 | NaN | 0.000 | 29571 | 0.744 | 0.002 | 16.7 | - |
| 7 | Wavenet | 467d23581f7b | {} | ind24 | 16 | 2 | 0.161 ± 0.002 | 0.335 ± 0.005 | 0.426 ± 0.002 | 1.021 | 0.068 | 0.640 | NaN | 0.000 | 16227 | 1.210 | 0.002 | 17.2 | - |
| 8 | MAJORITY | bbae4b588c23 | {} | ind24 | 16 | 3 | 0.000 ± 0.000 | 0.179 ± 0.000 | 0.333 ± 0.000 | 1.049 | 0.090 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 5a7196f85699 | 0.872 ± 0.000 | 0.848 ± 0.000 | 0.875 ± 0.000 | 0.919 ± 0.000 | 0.901 ± 0.000 | 0.920 ± 0.000 |
| LSTM_FAM | 01596109699e | 0.224 ± 0.003 | 0.111 ± 0.003 | 0.197 ± 0.001 | 0.443 ± 0.007 | 0.392 ± 0.000 | 0.440 ± 0.001 |
| MLP_FAM | c1aa32b8dcea | 0.225 ± 0.002 | 0.112 ± 0.001 | 0.192 ± 0.001 | 0.440 ± 0.008 | 0.386 ± 0.006 | 0.433 ± 0.001 |
| RIDGE | e700753b5111 | 0.212 ± 0.004 | 0.117 ± 0.003 | 0.185 ± 0.003 | 0.440 ± 0.004 | 0.402 ± 0.003 | 0.432 ± 0.001 |
| TRANSFORMER_FAM | 1e9e0b212d86 | 0.219 ± 0.005 | 0.101 ± 0.001 | 0.181 ± 0.006 | 0.444 ± 0.004 | 0.370 ± 0.025 | 0.430 ± 0.006 |
| TCN_FAM | ae29f8c1fcd8 | 0.216 ± 0.000 | 0.100 ± 0.002 | 0.167 ± 0.002 | 0.438 ± 0.003 | 0.368 ± 0.006 | 0.413 ± 0.005 |
| Wavenet | 467d23581f7b | 0.214 ± 0.001 | 0.109 ± 0.003 | 0.158 ± 0.001 | 0.374 ± 0.002 | 0.301 ± 0.012 | 0.330 ± 0.001 |
| MAJORITY | bbae4b588c23 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.149 ± 0.000 | 0.174 ± 0.000 | 0.215 ± 0.000 |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 01596109699e | LSTM_FAM | 0.226 | 0.501 | 0.274 |
| 1e9e0b212d86 | TRANSFORMER_FAM | 0.218 | 0.483 | 0.299 |
| 467d23581f7b | Wavenet | 0.021 | 0.534 | 0.445 |
| 5a7196f85699 | PERSISTENCE | 0.361 | 0.456 | 0.183 |
| ae29f8c1fcd8 | TCN_FAM | 0.190 | 0.500 | 0.311 |
| bbae4b588c23 | MAJORITY | 0.303 | 0.697 | 0.000 |
| c1aa32b8dcea | MLP_FAM | 0.199 | 0.498 | 0.302 |
| e700753b5111 | RIDGE | 0.260 | 0.538 | 0.203 |
