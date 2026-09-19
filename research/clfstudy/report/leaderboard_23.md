# Leaderboard: phase 23

labels: tb_h12_u0.015_d0.015_cr1_pf0, tb_h16_u0.015_d0.015_cr1_pf0, tb_h24_u0.02_d0.02_cr1_pf0, tb_h32_u0.025_d0.025_cr1_pf0, tb_h48_u0.03_d0.03_cr1_pf0, tb_h8_u0.01_d0.01_cr1_pf0

Classification metrics against fixed tb_h12_u0.015_d0.015_cr1_pf0, tb_h16_u0.015_d0.015_cr1_pf0, tb_h24_u0.02_d0.02_cr1_pf0, tb_h32_u0.025_d0.025_cr1_pf0, tb_h48_u0.03_d0.03_cr1_pf0, tb_h8_u0.01_d0.01_cr1_pf0 labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, PERSISTENCE=1, MLP_FAM=2, MLP_FAM=2, MLP_FAM=2, MLP_FAM=2, RIDGE=3, MLP_FAM=2, MLP_FAM=2, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, RIDGE=3, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1, MAJORITY=1)

| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 | bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params | train (min) | infer ms (b256) | epochs_run | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PERSISTENCE | 17883d5cc292 | {} | raw8 | 16 | 1 | 0.878 ± NaN | 0.922 ± NaN | 0.922 ± NaN | 0.523 | 0.068 | 0.940 | 0.915 | 0.202 | 0 | 0.000 | NaN | 0.0 | baseline |
| 2 | PERSISTENCE | ae1f2642da95 | {} | ind24 | 16 | 1 | 0.878 ± NaN | 0.922 ± NaN | 0.922 ± NaN | 0.523 | 0.068 | 0.940 | 0.915 | 0.202 | 0 | 0.000 | NaN | 0.0 | baseline |
| 3 | PERSISTENCE | f028a05576bd | {} | raw8 | 16 | 1 | 0.851 ± NaN | 0.905 ± NaN | 0.905 ± NaN | 0.618 | 0.080 | 0.927 | 0.893 | 0.188 | 0 | 0.000 | NaN | 0.0 | baseline |
| 4 | PERSISTENCE | cb0d692383b8 | {} | ind24 | 16 | 1 | 0.825 ± NaN | 0.887 ± NaN | 0.887 ± NaN | 0.763 | 0.099 | 0.914 | 0.878 | 0.201 | 0 | 0.000 | NaN | 0.0 | baseline |
| 5 | PERSISTENCE | 239617d19dd5 | {} | raw8 | 16 | 1 | 0.825 ± NaN | 0.887 ± NaN | 0.887 ± NaN | 0.763 | 0.099 | 0.914 | 0.878 | 0.201 | 0 | 0.000 | NaN | 0.0 | baseline |
| 6 | PERSISTENCE | 926de3d8bd2a | {} | raw8 | 16 | 1 | 0.782 ± NaN | 0.860 ± NaN | 0.860 ± NaN | 0.968 | 0.126 | 0.893 | 0.849 | 0.206 | 0 | 0.000 | NaN | 0.0 | baseline |
| 7 | PERSISTENCE | 21666484c86b | {} | raw8 | 16 | 1 | 0.759 ± NaN | 0.845 ± NaN | 0.845 ± NaN | 0.955 | 0.125 | 0.881 | 0.817 | 0.168 | 0 | 0.000 | NaN | 0.0 | baseline |
| 8 | PERSISTENCE | 051c7a9a9e62 | {} | raw8 | 16 | 1 | 0.692 ± NaN | 0.801 ± NaN | 0.801 ± NaN | 1.327 | 0.173 | 0.848 | 0.772 | 0.185 | 0 | 0.000 | NaN | 0.0 | baseline |
| 9 | MLP_FAM | 478eb86b7d49 | {"width": 64} | raw8 | 16 | 2 | 0.215 ± 0.002 | 0.451 ± 0.002 | 0.462 ± 0.002 | 0.995 | 0.130 | 0.678 | 1.000 | 0.000 | 12611 | 0.855 | 0.001 | 21.0 | - |
| 10 | MLP_FAM | 9a06cb27f5ab | {"width": 64} | raw8 | 16 | 2 | 0.207 ± 0.001 | 0.454 ± 0.002 | 0.456 ± 0.001 | 1.016 | 0.103 | 0.663 | NaN | 0.000 | 12611 | 0.746 | 0.001 | 19.0 | - |
| 11 | MLP_FAM | 4c8d9b3bbf29 | {"width": 64} | raw8 | 16 | 2 | 0.204 ± 0.001 | 0.449 ± 0.003 | 0.452 ± 0.002 | 1.010 | 0.111 | 0.665 | 0.000 | 0.000 | 12611 | 0.963 | 0.001 | 23.3 | - |
| 12 | MLP_FAM | 6519489a3625 | {"width": 64} | raw8 | 16 | 2 | 0.204 ± 0.001 | 0.449 ± 0.001 | 0.457 ± 0.001 | 1.009 | 0.098 | 0.663 | NaN | 0.000 | 12611 | 0.641 | 0.001 | 24.7 | - |
| 13 | RIDGE | 947d9f2d2796 | {} | ind24 | 16 | 3 | 0.200 ± 0.002 | 0.444 ± 0.003 | 0.446 ± 0.002 | 1.032 | 0.122 | 0.653 | NaN | 0.000 | 1155 | 0.489 | 0.001 | 23.9 | baseline |
| 14 | MLP_FAM | 8398df06227c | {"width": 64} | raw8 | 16 | 2 | 0.198 ± 0.002 | 0.442 ± 0.003 | 0.446 ± 0.002 | 1.003 | 0.126 | 0.666 | 1.000 | 0.000 | 12611 | 1.029 | 0.001 | 24.8 | - |
| 15 | MLP_FAM | 775fff906eac | {"width": 64} | raw8 | 16 | 2 | 0.185 ± 0.001 | 0.435 ± 0.001 | 0.437 ± 0.001 | 1.018 | 0.120 | 0.654 | NaN | 0.000 | 12611 | 0.673 | 0.001 | 18.8 | - |
| 16 | RIDGE | 3f84ab72c7d7 | {} | ind24 | 16 | 3 | 0.180 ± 0.003 | 0.430 ± 0.004 | 0.432 ± 0.002 | 1.030 | 0.112 | 0.642 | NaN | 0.000 | 1155 | 0.429 | 0.001 | 21.2 | baseline |
| 17 | RIDGE | de187e3fb32a | {} | raw8 | 16 | 3 | 0.168 ± 0.001 | 0.416 ± 0.002 | 0.417 ± 0.001 | 1.027 | 0.189 | 0.618 | NaN | 0.000 | 387 | 0.512 | 0.001 | 22.0 | baseline |
| 18 | RIDGE | 4f6a7fd0d1d1 | {} | raw8 | 16 | 3 | 0.165 ± 0.002 | 0.418 ± 0.003 | 0.418 ± 0.002 | 1.043 | 0.157 | 0.615 | NaN | 0.000 | 387 | 0.546 | 0.001 | 26.3 | baseline |
| 19 | RIDGE | f95dc722a734 | {} | raw8 | 16 | 3 | 0.161 ± 0.002 | 0.410 ± 0.003 | 0.412 ± 0.001 | 1.035 | 0.169 | 0.618 | NaN | 0.000 | 387 | 0.609 | 0.001 | 21.1 | baseline |
| 20 | RIDGE | d6345ba4fc98 | {} | raw8 | 16 | 3 | 0.159 ± 0.002 | 0.410 ± 0.004 | 0.413 ± 0.002 | 1.042 | 0.152 | 0.613 | NaN | 0.000 | 387 | 0.821 | 0.001 | 28.2 | baseline |
| 21 | RIDGE | b35f797da79f | {} | raw8 | 16 | 3 | 0.157 ± 0.001 | 0.408 ± 0.003 | 0.412 ± 0.002 | 1.045 | 0.143 | 0.608 | NaN | 0.000 | 387 | 0.654 | 0.001 | 23.1 | baseline |
| 22 | RIDGE | 1b3a1fd4aea7 | {} | raw8 | 16 | 3 | 0.153 ± 0.001 | 0.405 ± 0.001 | 0.408 ± 0.001 | 1.041 | 0.152 | 0.611 | NaN | 0.000 | 387 | 0.699 | 0.001 | 24.4 | baseline |
| 23 | MAJORITY | 07ee811ebaeb | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.238 ± NaN | 0.333 ± NaN | 1.000 | 0.090 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 24 | MAJORITY | 702d88601ddb | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.244 ± NaN | 0.333 ± NaN | 0.983 | 0.112 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 25 | MAJORITY | 6e2ee8809702 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.246 ± NaN | 0.333 ± NaN | 0.971 | 0.081 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 26 | MAJORITY | c35b8a913c91 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.244 ± NaN | 0.333 ± NaN | 0.982 | 0.092 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 27 | MAJORITY | 50ede9b5436a | {} | ind24 | 16 | 1 | 0.000 ± NaN | 0.244 ± NaN | 0.333 ± NaN | 0.983 | 0.112 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 28 | MAJORITY | 3578723b8987 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.260 ± NaN | 0.333 ± NaN | 0.909 | 0.079 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 29 | MAJORITY | f6efd7cb9daa | {} | ind24 | 16 | 1 | 0.000 ± NaN | 0.244 ± NaN | 0.333 ± NaN | 0.982 | 0.092 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |
| 30 | MAJORITY | 72ca30e64366 | {} | raw8 | 16 | 1 | 0.000 ± NaN | 0.252 ± NaN | 0.333 ± NaN | 0.950 | 0.099 | 0.500 | NaN | 0.000 | 0 | 0.000 | NaN | 0.0 | collapsed,baseline |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A | MCC B | MCC C | macro-F1 A | macro-F1 B | macro-F1 C |
|---|---|---|---|---|---|---|---|
| PERSISTENCE | 17883d5cc292 | 0.881 ± NaN | 0.870 ± NaN | 0.885 ± NaN | 0.923 ± NaN | 0.917 ± NaN | 0.927 ± NaN |
| PERSISTENCE | ae1f2642da95 | 0.881 ± NaN | 0.870 ± NaN | 0.885 ± NaN | 0.923 ± NaN | 0.917 ± NaN | 0.927 ± NaN |
| PERSISTENCE | f028a05576bd | 0.852 ± NaN | 0.841 ± NaN | 0.861 ± NaN | 0.905 ± NaN | 0.899 ± NaN | 0.911 ± NaN |
| PERSISTENCE | cb0d692383b8 | 0.830 ± NaN | 0.809 ± NaN | 0.835 ± NaN | 0.890 ± NaN | 0.878 ± NaN | 0.894 ± NaN |
| PERSISTENCE | 239617d19dd5 | 0.830 ± NaN | 0.809 ± NaN | 0.835 ± NaN | 0.890 ± NaN | 0.878 ± NaN | 0.894 ± NaN |
| PERSISTENCE | 926de3d8bd2a | 0.786 ± NaN | 0.765 ± NaN | 0.797 ± NaN | 0.862 ± NaN | 0.849 ± NaN | 0.869 ± NaN |
| PERSISTENCE | 21666484c86b | 0.761 ± NaN | 0.744 ± NaN | 0.773 ± NaN | 0.845 ± NaN | 0.836 ± NaN | 0.853 ± NaN |
| PERSISTENCE | 051c7a9a9e62 | 0.695 ± NaN | 0.671 ± NaN | 0.710 ± NaN | 0.802 ± NaN | 0.789 ± NaN | 0.812 ± NaN |
| MLP_FAM | 478eb86b7d49 | 0.260 ± 0.002 | 0.167 ± 0.002 | 0.217 ± 0.001 | 0.473 ± 0.000 | 0.424 ± 0.006 | 0.457 ± 0.001 |
| MLP_FAM | 9a06cb27f5ab | 0.254 ± 0.001 | 0.156 ± 0.002 | 0.211 ± 0.000 | 0.473 ± 0.002 | 0.428 ± 0.002 | 0.460 ± 0.001 |
| MLP_FAM | 4c8d9b3bbf29 | 0.253 ± 0.001 | 0.152 ± 0.002 | 0.207 ± 0.000 | 0.470 ± 0.001 | 0.421 ± 0.006 | 0.457 ± 0.000 |
| MLP_FAM | 6519489a3625 | 0.249 ± 0.002 | 0.154 ± 0.000 | 0.209 ± 0.001 | 0.471 ± 0.001 | 0.422 ± 0.000 | 0.453 ± 0.002 |
| RIDGE | 947d9f2d2796 | 0.241 ± 0.003 | 0.147 ± 0.001 | 0.211 ± 0.001 | 0.462 ± 0.001 | 0.416 ± 0.004 | 0.455 ± 0.003 |
| MLP_FAM | 8398df06227c | 0.251 ± 0.001 | 0.148 ± 0.005 | 0.196 ± 0.000 | 0.461 ± 0.003 | 0.419 ± 0.005 | 0.446 ± 0.000 |
| MLP_FAM | 775fff906eac | 0.235 ± 0.001 | 0.135 ± 0.002 | 0.187 ± 0.000 | 0.454 ± 0.001 | 0.413 ± 0.000 | 0.439 ± 0.001 |
| RIDGE | 3f84ab72c7d7 | 0.215 ± 0.003 | 0.133 ± 0.003 | 0.191 ± 0.002 | 0.445 ± 0.002 | 0.409 ± 0.002 | 0.437 ± 0.007 |
| RIDGE | de187e3fb32a | 0.224 ± 0.001 | 0.132 ± 0.000 | 0.147 ± 0.001 | 0.440 ± 0.003 | 0.403 ± 0.001 | 0.405 ± 0.002 |
| RIDGE | 4f6a7fd0d1d1 | 0.220 ± 0.005 | 0.128 ± 0.002 | 0.148 ± 0.000 | 0.447 ± 0.005 | 0.400 ± 0.003 | 0.406 ± 0.001 |
| RIDGE | f95dc722a734 | 0.227 ± 0.003 | 0.109 ± 0.002 | 0.147 ± 0.002 | 0.446 ± 0.004 | 0.384 ± 0.004 | 0.401 ± 0.002 |
| RIDGE | d6345ba4fc98 | 0.225 ± 0.002 | 0.112 ± 0.001 | 0.142 ± 0.004 | 0.446 ± 0.004 | 0.386 ± 0.004 | 0.397 ± 0.004 |
| RIDGE | b35f797da79f | 0.220 ± 0.001 | 0.115 ± 0.001 | 0.137 ± 0.000 | 0.443 ± 0.003 | 0.389 ± 0.002 | 0.392 ± 0.004 |
| RIDGE | 1b3a1fd4aea7 | 0.222 ± 0.002 | 0.099 ± 0.001 | 0.138 ± 0.001 | 0.442 ± 0.001 | 0.377 ± 0.002 | 0.394 ± 0.001 |
| MAJORITY | 07ee811ebaeb | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.258 ± NaN | 0.213 ± NaN | 0.244 ± NaN |
| MAJORITY | 702d88601ddb | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.266 ± NaN | 0.213 ± NaN | 0.253 ± NaN |
| MAJORITY | 6e2ee8809702 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.261 ± NaN | 0.224 ± NaN | 0.253 ± NaN |
| MAJORITY | c35b8a913c91 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.263 ± NaN | 0.219 ± NaN | 0.250 ± NaN |
| MAJORITY | 50ede9b5436a | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.266 ± NaN | 0.213 ± NaN | 0.253 ± NaN |
| MAJORITY | 3578723b8987 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.275 ± NaN | 0.240 ± NaN | 0.264 ± NaN |
| MAJORITY | f6efd7cb9daa | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.263 ± NaN | 0.219 ± NaN | 0.250 ± NaN |
| MAJORITY | 72ca30e64366 | 0.000 ± NaN | 0.000 ± NaN | 0.000 ± NaN | 0.271 ± NaN | 0.226 ± NaN | 0.259 ± NaN |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| 051c7a9a9e62 | PERSISTENCE | 0.230 | 0.585 | 0.185 |
| 07ee811ebaeb | MAJORITY | 0.000 | 1.000 | 0.000 |
| 17883d5cc292 | PERSISTENCE | 0.218 | 0.581 | 0.202 |
| 1b3a1fd4aea7 | RIDGE | 0.178 | 0.699 | 0.123 |
| 21666484c86b | PERSISTENCE | 0.192 | 0.639 | 0.169 |
| 239617d19dd5 | PERSISTENCE | 0.222 | 0.578 | 0.201 |
| 3578723b8987 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 3f84ab72c7d7 | RIDGE | 0.239 | 0.568 | 0.193 |
| 478eb86b7d49 | MLP_FAM | 0.182 | 0.560 | 0.258 |
| 4c8d9b3bbf29 | MLP_FAM | 0.206 | 0.547 | 0.247 |
| 4f6a7fd0d1d1 | RIDGE | 0.165 | 0.690 | 0.144 |
| 50ede9b5436a | MAJORITY | 0.000 | 1.000 | 0.000 |
| 6519489a3625 | MLP_FAM | 0.217 | 0.511 | 0.272 |
| 6e2ee8809702 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 702d88601ddb | MAJORITY | 0.000 | 1.000 | 0.000 |
| 72ca30e64366 | MAJORITY | 0.000 | 1.000 | 0.000 |
| 775fff906eac | MLP_FAM | 0.229 | 0.574 | 0.197 |
| 8398df06227c | MLP_FAM | 0.214 | 0.574 | 0.212 |
| 926de3d8bd2a | PERSISTENCE | 0.237 | 0.558 | 0.206 |
| 947d9f2d2796 | RIDGE | 0.250 | 0.563 | 0.187 |
| 9a06cb27f5ab | MLP_FAM | 0.223 | 0.524 | 0.252 |
| ae1f2642da95 | PERSISTENCE | 0.218 | 0.581 | 0.202 |
| b35f797da79f | RIDGE | 0.170 | 0.699 | 0.131 |
| c35b8a913c91 | MAJORITY | 0.000 | 1.000 | 0.000 |
| cb0d692383b8 | PERSISTENCE | 0.222 | 0.578 | 0.201 |
| d6345ba4fc98 | RIDGE | 0.168 | 0.698 | 0.133 |
| de187e3fb32a | RIDGE | 0.159 | 0.719 | 0.121 |
| f028a05576bd | PERSISTENCE | 0.203 | 0.609 | 0.188 |
| f6efd7cb9daa | MAJORITY | 0.000 | 1.000 | 0.000 |
| f95dc722a734 | RIDGE | 0.171 | 0.707 | 0.122 |
