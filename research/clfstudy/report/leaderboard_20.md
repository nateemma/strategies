# Leaderboard: phase 20

labels: tb_h48_u0.005_d0.005_cr1_pf0_atr1, tb_h48_u0.005_d0.005_cr1_pf0_atr2, tb_h48_u0.005_d0.005_cr1_pf0_atr3, tb_h48_u0.02_d0.02_cr1_pf0

Classification metrics against fixed tb_h48_u0.005_d0.005_cr1_pf0_atr1, tb_h48_u0.005_d0.005_cr1_pf0_atr2, tb_h48_u0.005_d0.005_cr1_pf0_atr3, tb_h48_u0.02_d0.02_cr1_pf0 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=3, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, MLP_FAM=2, RIDGE=3, RIDGE=3, MLP_FAM=2, MLP_FAM=2, RIDGE=3, RIDGE=3, RIDGE=3, MLP_FAM=2, MAJORITY=1, MAJORITY=3, MAJORITY=1, MAJORITY=1)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 5a7196f85699 | {} | ind24 | 16 | 3 | 0.850 ± 0.000 | 0.900 ± 0.000 | 0.900 ± 0.000 | 0.748 | 0.097 | 0.925 | 0.920 | 0.297 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | PERSISTENCE | 806f9100265b | {} | ind24 | 16 | 1 | 0.759 ± NaN | 0.795 ± NaN | 0.795 ± NaN | 1.115 | 0.146 | 0.860 | 0.904 | 0.383 | 0 | 0.000 | NaN | 0.0 | baseline |
| 3 | PERSISTENCE | 38aa9b6f18b2 | {} | ind24 | 16 | 1 | 0.721 ± NaN | 0.754 ± NaN | 0.754 ± NaN | 1.266 | 0.165 | 0.834 | 0.893 | 0.377 | 0 | 0.000 | NaN | 0.0 | baseline |
| 4 | PERSISTENCE | 99459e508ede | {} | ind24 | 16 | 1 | 0.671 ± NaN | 0.729 ± NaN | 0.729 ± NaN | 1.502 | 0.197 | 0.812 | 0.870 | 0.356 | 0 | 0.000 | NaN | 0.0 | baseline |
| 5 | MLP_FAM | 8cf98280ff82 | {"width": 64} | raw8 | 16 | 2 | 0.163 ± 0.002 | 0.430 ± 0.001 | 0.435 ± 0.001 | 1.048 | 0.036 | 0.625 | 0.500 | 0.000 | 12611 | 0.515 | 0.001 | 21.5 | - |
| 6 | RIDGE | e700753b5111 | {} | ind24 | 16 | 3 | 0.161 ± 0.001 | 0.422 ± 0.002 | 0.432 ± 0.001 | 1.059 | 0.046 | 0.617 | NaN | 0.000 | 1155 | 0.531 | 0.001 | 25.6 | baseline |
| 7 | RIDGE | 4b5fca8c62e8 | {} | raw8 | 16 | 3 | 0.119 ± 0.001 | 0.365 ± 0.005 | 0.399 ± 0.001 | 1.076 | 0.060 | 0.584 | NaN | 0.000 | 387 | 0.361 | 0.001 | 18.2 | baseline |
| 8 | MLP_FAM | 0e557fb2d854 | {"width": 64} | ind24 | 16 | 2 | 0.088 ± 0.003 | 0.389 ± 0.003 | 0.401 ± 0.001 | 1.075 | 0.050 | 0.569 | NaN | 0.000 | 28995 | 0.479 | 0.001 | 17.8 | - |
| 9 | MLP_FAM | 7f59acfb484c | {"width": 64} | ind24 | 16 | 2 | 0.083 ± 0.002 | 0.374 ± 0.005 | 0.391 ± 0.003 | 1.077 | 0.028 | 0.567 | NaN | 0.000 | 28995 | 0.549 | 0.001 | 19.5 | - |
| 10 | RIDGE | 9225aa2a6874 | {} | ind24 | 16 | 3 | 0.081 ± 0.002 | 0.374 ± 0.006 | 0.397 ± 0.001 | 1.080 | 0.028 | 0.566 | NaN | 0.000 | 1155 | 0.426 | 0.001 | 22.1 | baseline |
| 11 | RIDGE | 0351fd438202 | {} | ind24 | 16 | 3 | 0.079 ± 0.003 | 0.377 ± 0.003 | 0.384 ± 0.002 | 1.080 | 0.041 | 0.562 | NaN | 0.000 | 1155 | 0.474 | 0.001 | 21.3 | baseline |
| 12 | RIDGE | ea0c913f117e | {} | ind24 | 16 | 3 | 0.069 ± 0.002 | 0.369 ± 0.003 | 0.375 ± 0.002 | 1.081 | 0.048 | 0.553 | NaN | 0.000 | 1155 | 0.442 | 0.001 | 21.2 | baseline |
| 13 | MLP_FAM | dca68572f4fe | {"width": 64} | ind24 | 16 | 2 | 0.066 ± 0.004 | 0.362 ± 0.005 | 0.375 ± 0.001 | 1.086 | 0.037 | 0.552 | NaN | 0.000 | 28995 | 0.473 | 0.001 | 17.5 | - |
| 14 | MAJORITY | 7cdbea8c4aa7 | {} | ind24 | 16 | 1 | 0.000 ± NaN | 0.214 ± NaN | 0.333 ± NaN | 1.002 | 0.022 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 15 | MAJORITY | bbae4b588c23 | {} | ind24 | 16 | 3 | 0.000 ± 0.000 | 0.151 ± 0.000 | 0.333 ± 0.000 | 1.108 | 0.072 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 16 | MAJORITY | dddeca7df34c | {} | ind24 | 16 | 1 | 0.000 ± NaN | 0.223 ± NaN | 0.333 ± NaN | 0.992 | 0.012 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 17 | MAJORITY | 22a3f4f3054b | {} | ind24 | 16 | 1 | 0.000 ± NaN | 0.221 ± NaN | 0.333 ± NaN | 0.979 | 0.016 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 5a7196f85699 | 0.859 ± 0.000 | 0.831 ± 0.000 | 0.860 ± 0.000 | 0.909 ± 0.000 | 0.882 ± 0.000 | 0.909 ± 0.000 |
| PERSISTENCE | 806f9100265b | 0.763 ± NaN | 0.751 ± NaN | 0.763 ± NaN | 0.804 ± NaN | 0.782 ± NaN | 0.800 ± NaN |
| PERSISTENCE | 38aa9b6f18b2 | 0.731 ± NaN | 0.704 ± NaN | 0.727 ± NaN | 0.763 ± NaN | 0.740 ± NaN | 0.760 ± NaN |
| PERSISTENCE | 99459e508ede | 0.687 ± NaN | 0.641 ± NaN | 0.686 ± NaN | 0.738 ± NaN | 0.712 ± NaN | 0.736 ± NaN |
| MLP_FAM | 8cf98280ff82 | 0.213 ± 0.006 | 0.104 ± 0.000 | 0.173 ± 0.001 | 0.450 ± 0.001 | 0.403 ± 0.000 | 0.436 ± 0.001 |
| RIDGE | e700753b5111 | 0.202 ± 0.002 | 0.102 ± 0.001 | 0.179 ± 0.001 | 0.437 ± 0.002 | 0.397 ± 0.001 | 0.433 ± 0.004 |
| RIDGE | 4b5fca8c62e8 | 0.188 ± 0.002 | 0.067 ± 0.001 | 0.103 ± 0.001 | 0.419 ± 0.006 | 0.333 ± 0.005 | 0.344 ± 0.003 |
| MLP_FAM | 0e557fb2d854 | 0.084 ± 0.006 | 0.070 ± 0.002 | 0.111 ± 0.001 | 0.393 ± 0.007 | 0.368 ± 0.001 | 0.405 ± 0.001 |
| MLP_FAM | 7f59acfb484c | 0.074 ± 0.000 | 0.070 ± 0.001 | 0.105 ± 0.003 | 0.370 ± 0.003 | 0.360 ± 0.007 | 0.391 ± 0.006 |
| RIDGE | 9225aa2a6874 | 0.084 ± 0.003 | 0.057 ± 0.002 | 0.103 ± 0.002 | 0.387 ± 0.004 | 0.347 ± 0.006 | 0.390 ± 0.006 |
| RIDGE | 0351fd438202 | 0.075 ± 0.004 | 0.061 ± 0.003 | 0.101 ± 0.001 | 0.375 ± 0.005 | 0.367 ± 0.002 | 0.389 ± 0.002 |
| RIDGE | ea0c913f117e | 0.059 ± 0.003 | 0.050 ± 0.000 | 0.096 ± 0.001 | 0.364 ± 0.004 | 0.358 ± 0.003 | 0.385 ± 0.001 |
| MLP_FAM | dca68572f4fe | 0.056 ± 0.003 | 0.046 ± 0.003 | 0.095 ± 0.005 | 0.359 ± 0.001 | 0.345 ± 0.006 | 0.380 ± 0.009 |
| MAJORITY | 7cdbea8c4aa7 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.205 ± NaN | 0.217 ± NaN | 0.219 ± NaN |
| MAJORITY | bbae4b588c23 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.143 ± 0.000 | 0.138 ± 0.000 | 0.172 ± 0.000 |
| MAJORITY | dddeca7df34c | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.218 ± NaN | 0.224 ± NaN | 0.226 ± NaN |
| MAJORITY | 22a3f4f3054b | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.214 ± NaN | 0.223 ± NaN | 0.225 ± NaN |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 0351fd438202 | RIDGE | 0.412 | 0.205 | 0.383 |
| 0e557fb2d854 | MLP_FAM | 0.432 | 0.231 | 0.337 |
| 22a3f4f3054b | MAJORITY | 1.000 | 0.000 | 0.000 |
| 38aa9b6f18b2 | PERSISTENCE | 0.496 | 0.127 | 0.376 |
| 4b5fca8c62e8 | RIDGE | 0.206 | 0.658 | 0.136 |
| 5a7196f85699 | PERSISTENCE | 0.341 | 0.361 | 0.297 |
| 7cdbea8c4aa7 | MAJORITY | 1.000 | 0.000 | 0.000 |
| 7f59acfb484c | MLP_FAM | 0.353 | 0.263 | 0.385 |
| 806f9100265b | PERSISTENCE | 0.474 | 0.144 | 0.382 |
| 8cf98280ff82 | MLP_FAM | 0.275 | 0.450 | 0.276 |
| 9225aa2a6874 | RIDGE | 0.393 | 0.301 | 0.306 |
| 99459e508ede | PERSISTENCE | 0.503 | 0.142 | 0.355 |
| bbae4b588c23 | MAJORITY | 0.668 | 0.332 | 0.000 |
| dca68572f4fe | MLP_FAM | 0.437 | 0.259 | 0.304 |
| dddeca7df34c | MAJORITY | 1.000 | 0.000 | 0.000 |
| e700753b5111 | RIDGE | 0.266 | 0.506 | 0.228 |
| ea0c913f117e | RIDGE | 0.439 | 0.193 | 0.368 |
