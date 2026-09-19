# Leaderboard: phase 2

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=3, RIDGE=3, ROCKET=3, ROCKET=3, MAJORITY=3)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | ffb5c40e8d6b | {} | ind24 | 16 | 3 | 0.805 ± 0.000 | 0.869 ± 0.000 | 0.869 ± 0.000 | 0.351 | 0.045 | 0.902 | 0.834 | 0.081 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | RIDGE | d1dc0339b10b | {} | ind24 | 16 | 3 | 0.538 ± 0.002 | 0.658 ± 0.002 | 0.864 ± 0.002 | 0.556 | 0.153 | 0.937 | 0.588 | 0.036 | 1155 | 1.155 | 0.001 | 30.6 | baseline |
| 3 | ROCKET | 12c3636e7110 | {"feature_norm": "standardize"} | ind24 | 16 | 3 | 0.497 ± 0.029 | 0.633 ± 0.017 | 0.805 ± 0.061 | 0.465 | 0.053 | 0.929 | 0.592 | 0.040 | 6003 | 17.109 | 0.021 | 35.9 | baseline |
| 4 | ROCKET | c43f5e55fe61 | {} | ind24 | 16 | 3 | 0.313 ± 0.189 | 0.525 ± 0.132 | 0.575 ± 0.116 | 0.463 | 0.098 | 0.816 | 0.620 | 0.015 | 6003 | 5.326 | 0.023 | 14.8 | collapsed,baseline |
| 5 | MAJORITY | 6f0c110ce13c | {} | ind24 | 16 | 3 | 0.000 ± 0.000 | 0.310 ± 0.000 | 0.333 ± 0.000 | 0.482 | 0.037 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | ffb5c40e8d6b | 0.801 ± 0.000 | 0.802 ± 0.000 | 0.812 ± 0.000 | 0.864 ± 0.000 | 0.868 ± 0.000 | 0.874 ± 0.000 |
| RIDGE | d1dc0339b10b | 0.533 ± 0.003 | 0.535 ± 0.000 | 0.547 ± 0.003 | 0.656 ± 0.003 | 0.652 ± 0.000 | 0.666 ± 0.002 |
| ROCKET | 12c3636e7110 | 0.450 ± 0.082 | 0.516 ± 0.003 | 0.526 ± 0.003 | 0.611 ± 0.048 | 0.638 ± 0.002 | 0.650 ± 0.002 |
| ROCKET | c43f5e55fe61 | 0.539 ± 0.006 | 0.214 ± 0.303 | 0.186 ± 0.256 | 0.671 ± 0.004 | 0.459 ± 0.221 | 0.446 ± 0.172 |
| MAJORITY | 6f0c110ce13c | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.315 ± 0.000 | 0.303 ± 0.000 | 0.311 ± 0.000 |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 12c3636e7110 | ROCKET | 0.113 | 0.733 | 0.154 |
| 6f0c110ce13c | MAJORITY | 0.000 | 1.000 | 0.000 |
| c43f5e55fe61 | ROCKET | 0.042 | 0.866 | 0.092 |
| d1dc0339b10b | RIDGE | 0.129 | 0.712 | 0.159 |
| ffb5c40e8d6b | PERSISTENCE | 0.052 | 0.867 | 0.081 |
