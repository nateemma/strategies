# Leaderboard: phase 24

labels: fmae_h12_dd0.03_g0.0125_l0.0125, fmae_h12_dd0.03_g0.015_l0.015, fmae_h12_dd0.05_g0.015_l0.015, fmae_h16_dd0.03_g0.015_l0.015, qfut_h12_g0.001_l0.001_q0.75, qfut_h12_g0.001_l0.001_q0.8

Classification metrics against fixed fmae_h12_dd0.03_g0.0125_l0.0125, fmae_h12_dd0.03_g0.015_l0.015, fmae_h12_dd0.05_g0.015_l0.015, fmae_h16_dd0.03_g0.015_l0.015, qfut_h12_g0.001_l0.001_q0.75, qfut_h12_g0.001_l0.001_q0.8 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, MLP_FAM=2, MLP_FAM=2, MLP_FAM=2, MLP_FAM=2, MLP_FAM=2, MLP_FAM=2, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 8efa7c7e7083 | {} | raw8 | 16 | 1 | 0.792 ± NaN | 0.867 ± NaN | 0.867 ± NaN | 0.907 | 0.118 | 0.898 | 0.853 | 0.210 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | PERSISTENCE | adfdd544a8e0 | {} | raw8 | 16 | 1 | 0.767 ± NaN | 0.850 ± NaN | 0.850 ± NaN | 0.924 | 0.120 | 0.885 | 0.818 | 0.167 | 0 | 0.000 | NaN | 0.0 | baseline |
| 3 | PERSISTENCE | a5c468262d1e | {} | raw8 | 16 | 1 | 0.766 ± NaN | 0.850 ± NaN | 0.850 ± NaN | 0.910 | 0.119 | 0.885 | 0.821 | 0.173 | 0 | 0.000 | NaN | 0.0 | baseline |
| 4 | PERSISTENCE | e10635b984d9 | {} | raw8 | 16 | 1 | 0.763 ± NaN | 0.847 ± NaN | 0.847 ± NaN | 0.918 | 0.120 | 0.883 | 0.812 | 0.162 | 0 | 0.000 | NaN | 0.0 | baseline |
| 5 | PERSISTENCE | 4075582cd4ea | {} | raw8 | 16 | 1 | 0.759 ± NaN | 0.845 ± NaN | 0.845 ± NaN | 1.061 | 0.138 | 0.881 | 0.826 | 0.205 | 0 | 0.000 | NaN | 0.0 | baseline |
| 6 | PERSISTENCE | 1d32174ecfc9 | {} | raw8 | 16 | 1 | 0.755 ± NaN | 0.843 ± NaN | 0.843 ± NaN | 1.069 | 0.140 | 0.880 | 0.819 | 0.192 | 0 | 0.000 | NaN | 0.0 | baseline |
| 7 | MLP_FAM | 230564a443c3 | {"width": 64} | raw8 | 16 | 2 | 0.211 ± 0.003 | 0.448 ± 0.002 | 0.460 ± 0.003 | 0.997 | 0.125 | 0.677 | 0.500 | 0.000 | 12611 | 0.609 | 0.001 | 26.0 | - |
| 8 | MLP_FAM | ea8eb12e4d63 | {"width": 64} | raw8 | 16 | 2 | 0.200 ± 0.001 | 0.441 ± 0.004 | 0.459 ± 0.001 | 0.992 | 0.112 | 0.673 | 0.750 | 0.000 | 12611 | 0.712 | 0.001 | 28.5 | - |
| 9 | MLP_FAM | aed541d0648b | {"width": 64} | raw8 | 16 | 2 | 0.200 ± 0.001 | 0.448 ± 0.001 | 0.456 ± 0.001 | 1.007 | 0.089 | 0.663 | 0.556 | 0.000 | 12611 | 0.802 | 0.001 | 34.7 | - |
| 10 | MLP_FAM | 63aade7b549e | {"width": 64} | raw8 | 16 | 2 | 0.197 ± 0.001 | 0.444 ± 0.002 | 0.455 ± 0.001 | 1.004 | 0.084 | 0.664 | 0.500 | 0.000 | 12611 | 0.974 | 0.001 | 38.2 | - |
| 11 | MLP_FAM | 50eae37770c7 | {"width": 64} | raw8 | 16 | 2 | 0.196 ± 0.002 | 0.441 ± 0.002 | 0.448 ± 0.002 | 0.995 | 0.146 | 0.666 | NaN | 0.000 | 12611 | 0.375 | 0.001 | 15.2 | - |
| 12 | MLP_FAM | aa0d585660c0 | {"width": 64} | raw8 | 16 | 2 | 0.193 ± 0.002 | 0.443 ± 0.002 | 0.446 ± 0.002 | 1.012 | 0.107 | 0.656 | NaN | 0.000 | 12611 | 0.528 | 0.001 | 20.0 | - |
| 13 | RIDGE | dc2bfb3b75d1 | {} | raw8 | 16 | 3 | 0.173 ± 0.002 | 0.421 ± 0.003 | 0.421 ± 0.001 | 1.016 | 0.187 | 0.632 | NaN | 0.000 | 387 | 0.469 | 0.001 | 23.2 | baseline |
| 14 | RIDGE | 259dd973647f | {} | raw8 | 16 | 3 | 0.169 ± 0.001 | 0.417 ± 0.002 | 0.419 ± 0.001 | 1.035 | 0.145 | 0.623 | NaN | 0.000 | 387 | 0.581 | 0.001 | 27.3 | baseline |
| 15 | RIDGE | fad269dfdcd2 | {} | raw8 | 16 | 3 | 0.156 ± 0.002 | 0.409 ± 0.002 | 0.409 ± 0.002 | 1.032 | 0.183 | 0.609 | 0.000 | 0.000 | 387 | 0.414 | 0.001 | 22.1 | baseline |
| 16 | RIDGE | 0cf073afdb60 | {} | raw8 | 16 | 3 | 0.147 ± 0.001 | 0.408 ± 0.001 | 0.409 ± 0.001 | 1.037 | 0.174 | 0.599 | 0.330 | 0.000 | 387 | 0.470 | 0.001 | 24.9 | baseline |
| 17 | RIDGE | 7b35b240fcdc | {} | raw8 | 16 | 3 | 0.141 ± 0.002 | 0.404 ± 0.001 | 0.406 ± 0.002 | 1.053 | 0.130 | 0.591 | 0.539 | 0.000 | 387 | 0.443 | 0.001 | 23.0 | baseline |
| 18 | RIDGE | dd5940dc6c98 | {} | raw8 | 16 | 3 | 0.135 ± 0.002 | 0.401 ± 0.002 | 0.405 ± 0.001 | 1.052 | 0.122 | 0.589 | 0.195 | 0.000 | 387 | 0.400 | 0.001 | 19.6 | baseline |
| 19 | MAJORITY | 9817325ab612 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.242 ± NaN | 0.333 ± NaN | 0.988 | 0.084 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 20 | MAJORITY | 11af8b1e0357 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.242 ± NaN | 0.333 ± NaN | 0.985 | 0.092 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 21 | MAJORITY | afe23de442e0 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.260 ± NaN | 0.333 ± NaN | 0.909 | 0.079 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 22 | MAJORITY | 434666168724 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.243 ± NaN | 0.333 ± NaN | 0.984 | 0.084 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 23 | MAJORITY | 37ce659065ab | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.262 ± NaN | 0.333 ± NaN | 0.896 | 0.077 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 24 | MAJORITY | 1f0553d218cd | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.262 ± NaN | 0.333 ± NaN | 0.895 | 0.085 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 8efa7c7e7083 | 0.794 ± NaN | 0.776 ± NaN | 0.807 ± NaN | 0.868 ± NaN | 0.857 ± NaN | 0.876 ± NaN |
| PERSISTENCE | adfdd544a8e0 | 0.768 ± NaN | 0.752 ± NaN | 0.781 ± NaN | 0.849 ± NaN | 0.842 ± NaN | 0.858 ± NaN |
| PERSISTENCE | a5c468262d1e | 0.767 ± NaN | 0.751 ± NaN | 0.781 ± NaN | 0.849 ± NaN | 0.842 ± NaN | 0.858 ± NaN |
| PERSISTENCE | e10635b984d9 | 0.765 ± NaN | 0.745 ± NaN | 0.779 ± NaN | 0.848 ± NaN | 0.837 ± NaN | 0.857 ± NaN |
| PERSISTENCE | 4075582cd4ea | 0.761 ± NaN | 0.739 ± NaN | 0.776 ± NaN | 0.846 ± NaN | 0.833 ± NaN | 0.855 ± NaN |
| PERSISTENCE | 1d32174ecfc9 | 0.759 ± NaN | 0.735 ± NaN | 0.772 ± NaN | 0.845 ± NaN | 0.830 ± NaN | 0.853 ± NaN |
| MLP_FAM | 230564a443c3 | 0.248 ± 0.007 | 0.167 ± 0.001 | 0.219 ± 0.002 | 0.458 ± 0.002 | 0.427 ± 0.003 | 0.460 ± 0.001 |
| MLP_FAM | ea8eb12e4d63 | 0.235 ± 0.001 | 0.157 ± 0.000 | 0.209 ± 0.001 | 0.451 ± 0.009 | 0.419 ± 0.002 | 0.454 ± 0.001 |
| MLP_FAM | aed541d0648b | 0.238 ± 0.002 | 0.152 ± 0.000 | 0.210 ± 0.001 | 0.461 ± 0.001 | 0.423 ± 0.001 | 0.460 ± 0.000 |
| MLP_FAM | 63aade7b549e | 0.233 ± 0.000 | 0.153 ± 0.000 | 0.203 ± 0.002 | 0.456 ± 0.004 | 0.422 ± 0.000 | 0.453 ± 0.002 |
| MLP_FAM | 50eae37770c7 | 0.238 ± 0.004 | 0.163 ± 0.000 | 0.188 ± 0.003 | 0.458 ± 0.003 | 0.424 ± 0.001 | 0.442 ± 0.001 |
| MLP_FAM | aa0d585660c0 | 0.245 ± 0.000 | 0.149 ± 0.001 | 0.187 ± 0.004 | 0.469 ± 0.000 | 0.419 ± 0.003 | 0.442 ± 0.003 |
| RIDGE | dc2bfb3b75d1 | 0.231 ± 0.001 | 0.154 ± 0.003 | 0.133 ± 0.000 | 0.449 ± 0.004 | 0.413 ± 0.003 | 0.400 ± 0.001 |
| RIDGE | 259dd973647f | 0.231 ± 0.001 | 0.143 ± 0.001 | 0.133 ± 0.001 | 0.453 ± 0.001 | 0.406 ± 0.001 | 0.393 ± 0.003 |
| RIDGE | fad269dfdcd2 | 0.209 ± 0.002 | 0.125 ± 0.001 | 0.136 ± 0.003 | 0.435 ± 0.001 | 0.396 ± 0.002 | 0.396 ± 0.003 |
| RIDGE | 0cf073afdb60 | 0.195 ± 0.002 | 0.123 ± 0.000 | 0.124 ± 0.000 | 0.431 ± 0.001 | 0.399 ± 0.001 | 0.393 ± 0.002 |
| RIDGE | 7b35b240fcdc | 0.191 ± 0.002 | 0.111 ± 0.002 | 0.120 ± 0.003 | 0.432 ± 0.002 | 0.391 ± 0.000 | 0.388 ± 0.001 |
| RIDGE | dd5940dc6c98 | 0.179 ± 0.002 | 0.115 ± 0.000 | 0.112 ± 0.002 | 0.427 ± 0.003 | 0.393 ± 0.000 | 0.383 ± 0.003 |
| MAJORITY | 9817325ab612 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.259 ± NaN | 0.218 ± NaN | 0.248 ± NaN |
| MAJORITY | 11af8b1e0357 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.262 ± NaN | 0.214 ± NaN | 0.249 ± NaN |
| MAJORITY | afe23de442e0 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.275 ± NaN | 0.240 ± NaN | 0.264 ± NaN |
| MAJORITY | 434666168724 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.261 ± NaN | 0.220 ± NaN | 0.248 ± NaN |
| MAJORITY | 37ce659065ab | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.277 ± NaN | 0.244 ± NaN | 0.267 ± NaN |
| MAJORITY | 1f0553d218cd | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.279 ± NaN | 0.239 ± NaN | 0.268 ± NaN |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 0cf073afdb60 | RIDGE | 0.187 | 0.683 | 0.129 |
| 11af8b1e0357 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 1d32174ecfc9 | PERSISTENCE | 0.237 | 0.571 | 0.193 |
| 1f0553d218cd | MAJORITY | 0.000 | 1.000 | 0.000 |
| 230564a443c3 | MLP_FAM | 0.229 | 0.550 | 0.221 |
| 259dd973647f | RIDGE | 0.163 | 0.694 | 0.143 |
| 37ce659065ab | MAJORITY | 0.000 | 1.000 | 0.000 |
| 4075582cd4ea | PERSISTENCE | 0.225 | 0.570 | 0.205 |
| 434666168724 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 50eae37770c7 | MLP_FAM | 0.155 | 0.608 | 0.238 |
| 63aade7b549e | MLP_FAM | 0.318 | 0.490 | 0.193 |
| 7b35b240fcdc | RIDGE | 0.217 | 0.654 | 0.128 |
| 8efa7c7e7083 | PERSISTENCE | 0.215 | 0.575 | 0.210 |
| 9817325ab612 | MAJORITY | 0.000 | 1.000 | 0.000 |
| a5c468262d1e | PERSISTENCE | 0.177 | 0.650 | 0.173 |
| aa0d585660c0 | MLP_FAM | 0.175 | 0.556 | 0.269 |
| adfdd544a8e0 | PERSISTENCE | 0.193 | 0.640 | 0.167 |
| aed541d0648b | MLP_FAM | 0.294 | 0.498 | 0.208 |
| afe23de442e0 | MAJORITY | 0.000 | 1.000 | 0.000 |
| dc2bfb3b75d1 | RIDGE | 0.154 | 0.718 | 0.128 |
| dd5940dc6c98 | RIDGE | 0.240 | 0.634 | 0.126 |
| e10635b984d9 | PERSISTENCE | 0.188 | 0.650 | 0.162 |
| ea8eb12e4d63 | MLP_FAM | 0.266 | 0.534 | 0.200 |
| fad269dfdcd2 | RIDGE | 0.163 | 0.715 | 0.122 |
