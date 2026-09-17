# Leaderboard: phase 2

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=3, ROCKET=3, RIDGE=3, ROCKET=3, MAJORITY=3)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | ffb5c40e8d6b | {} | ind24 | 16 | 3 | 0.805 ± 0.000 | 0.869 ± 0.000 | 0.869 ± 0.000 | 0.351 | 0.045 | 0.902 | 0.834 | 0.081 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | ROCKET | 12c3636e7110 | {"feature_norm": "standardize"} | ind24 | 16 | 3 | 0.400 ± 0.030 | 0.591 ± 0.019 | 0.597 ± 0.063 | 0.320 | 0.048 | 0.928 | 0.628 | 0.015 | 6003 | 9.445 | 0.023 | 24.8 | baseline |
| 3 | RIDGE | d1dc0339b10b | {} | ind24 | 16 | 3 | 0.380 ± 0.003 | 0.560 ± 0.003 | 0.598 ± 0.003 | 0.381 | 0.094 | 0.932 | 0.616 | 0.014 | 1155 | 0.862 | 0.001 | 23.9 | baseline |
| 4 | ROCKET | c43f5e55fe61 | {} | ind24 | 16 | 3 | 0.266 ± 0.157 | 0.469 ± 0.140 | 0.549 ± 0.097 | 0.551 | 0.127 | 0.767 | 0.536 | 0.019 | 6003 | 4.957 | 0.022 | 14.3 | collapsed,baseline |
| 5 | MAJORITY | 6f0c110ce13c | {} | ind24 | 16 | 3 | 0.000 ± 0.000 | 0.310 ± 0.000 | 0.333 ± 0.000 | 0.482 | 0.037 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | ffb5c40e8d6b | 0.801 ± 0.000 | 0.802 ± 0.000 | 0.812 ± 0.000 | 0.864 ± 0.000 | 0.868 ± 0.000 | 0.874 ± 0.000 |
| ROCKET | 12c3636e7110 | 0.450 ± 0.082 | 0.380 ± 0.007 | 0.371 ± 0.000 | 0.611 ± 0.048 | 0.584 ± 0.006 | 0.577 ± 0.004 |
| RIDGE | d1dc0339b10b | 0.533 ± 0.003 | 0.307 ± 0.004 | 0.299 ± 0.003 | 0.656 ± 0.003 | 0.517 ± 0.004 | 0.508 ± 0.003 |
| ROCKET | c43f5e55fe61 | 0.539 ± 0.006 | 0.144 ± 0.247 | 0.115 ± 0.219 | 0.671 ± 0.004 | 0.407 ± 0.180 | 0.329 ± 0.236 |
| MAJORITY | 6f0c110ce13c | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.315 ± 0.000 | 0.303 ± 0.000 | 0.311 ± 0.000 |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 12c3636e7110 | ROCKET | 0.046 | 0.886 | 0.068 |
| 6f0c110ce13c | MAJORITY | 0.000 | 1.000 | 0.000 |
| c43f5e55fe61 | ROCKET | 0.032 | 0.786 | 0.182 |
| d1dc0339b10b | RIDGE | 0.048 | 0.889 | 0.063 |
| ffb5c40e8d6b | PERSISTENCE | 0.052 | 0.867 | 0.081 |
