# Leaderboard: phase 26

labels: fmae_h12_dd0.1_g0.015_l0.015

Classification metrics against fixed fmae_h12_dd0.1_g0.015_l0.015 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=1, MLP_FAM=2, RIDGE=3, MAJORITY=1)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 821e38a0e71f | {} | raw8 | 16 | 1 | 0.767 ± NaN | 0.849 ± NaN | 0.849 ± NaN | 0.928 | 0.121 | 0.885 | 0.816 | 0.165 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | MLP_FAM | f116cdd07388 | {"width": 64} | raw8 | 16 | 2 | 0.216 ± 0.003 | 0.452 ± 0.002 | 0.461 ± 0.002 | 0.993 | 0.136 | 0.678 | NaN | 0.000 | 12611 | 0.455 | 0.001 | 17.5 | - |
| 3 | RIDGE | d7a384ada075 | {} | raw8 | 16 | 3 | 0.170 ± 0.001 | 0.419 ± 0.002 | 0.418 ± 0.002 | 1.028 | 0.186 | 0.621 | NaN | 0.000 | 387 | 0.528 | 0.001 | 25.2 | baseline |
| 4 | MAJORITY | 05aceb34ab69 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.259 ± NaN | 0.333 ± NaN | 0.911 | 0.080 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 821e38a0e71f | 0.768 ± NaN | 0.752 ± NaN | 0.781 ± NaN | 0.849 ± NaN | 0.841 ± NaN | 0.858 ± NaN |
| MLP_FAM | f116cdd07388 | 0.263 ± 0.006 | 0.167 ± 0.001 | 0.219 ± 0.002 | 0.468 ± 0.005 | 0.428 ± 0.000 | 0.459 ± 0.001 |
| RIDGE | d7a384ada075 | 0.224 ± 0.001 | 0.135 ± 0.000 | 0.152 ± 0.001 | 0.447 ± 0.002 | 0.404 ± 0.001 | 0.407 ± 0.003 |
| MAJORITY | 05aceb34ab69 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.275 ± NaN | 0.239 ± NaN | 0.264 ± NaN |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 05aceb34ab69 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 821e38a0e71f | PERSISTENCE | 0.198 | 0.637 | 0.165 |
| d7a384ada075 | RIDGE | 0.154 | 0.715 | 0.131 |
| f116cdd07388 | MLP_FAM | 0.197 | 0.571 | 0.231 |
