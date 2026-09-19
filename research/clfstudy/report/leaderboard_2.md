# Leaderboard: phase 2

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=3, RIDGE=3, ROCKET=3, ROCKET=3, MAJORITY=3)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | ffb5c40e8d6b | {} | ind24 | 16 | 3 | 0.805 ± 0.000 | 0.869 ± 0.000 | 0.869 ± 0.000 | 0.351 | 0.045 | 0.902 | 0.834 | 0.081 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | RIDGE | d1dc0339b10b | {} | ind24 | 16 | 3 | 0.486 ± 0.092 | 0.626 ± 0.057 | 0.775 ± 0.154 | 0.498 | 0.133 | 0.935 | 0.598 | 0.029 | 1155 | 1.060 | 0.001 | 28.7 | baseline |
| 3 | ROCKET | 12c3636e7110 | {"feature_norm": "standardize"} | ind24 | 16 | 3 | 0.465 ± 0.084 | 0.618 ± 0.041 | 0.735 ± 0.181 | 0.417 | 0.052 | 0.928 | 0.606 | 0.032 | 6003 | 14.101 | 0.022 | 31.2 | baseline |
| 4 | ROCKET | c43f5e55fe61 | {} | ind24 | 16 | 3 | 0.266 ± 0.157 | 0.469 ± 0.140 | 0.549 ± 0.097 | 0.551 | 0.127 | 0.767 | 0.536 | 0.019 | 6003 | 4.957 | 0.022 | 14.3 | collapsed,baseline |
| 5 | MAJORITY | 6f0c110ce13c | {} | ind24 | 16 | 3 | 0.000 ± 0.000 | 0.310 ± 0.000 | 0.333 ± 0.000 | 0.482 | 0.037 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | ffb5c40e8d6b | 0.801 ± 0.000 | 0.802 ± 0.000 | 0.812 ± 0.000 | 0.864 ± 0.000 | 0.868 ± 0.000 | 0.874 ± 0.000 |
| RIDGE | d1dc0339b10b | 0.533 ± 0.003 | 0.459 ± 0.132 | 0.465 ± 0.141 | 0.656 ± 0.003 | 0.606 ± 0.079 | 0.614 ± 0.089 |
| ROCKET | 12c3636e7110 | 0.450 ± 0.082 | 0.469 ± 0.080 | 0.475 ± 0.090 | 0.611 ± 0.048 | 0.619 ± 0.033 | 0.625 ± 0.042 |
| ROCKET | c43f5e55fe61 | 0.539 ± 0.006 | 0.144 ± 0.247 | 0.115 ± 0.219 | 0.671 ± 0.004 | 0.407 ± 0.180 | 0.329 ± 0.236 |
| MAJORITY | 6f0c110ce13c | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.315 ± 0.000 | 0.303 ± 0.000 | 0.311 ± 0.000 |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 12c3636e7110 | ROCKET | 0.095 | 0.774 | 0.132 |
| 6f0c110ce13c | MAJORITY | 0.000 | 1.000 | 0.000 |
| c43f5e55fe61 | ROCKET | 0.032 | 0.786 | 0.182 |
| d1dc0339b10b | RIDGE | 0.103 | 0.765 | 0.131 |
| ffb5c40e8d6b | PERSISTENCE | 0.052 | 0.867 | 0.081 |
