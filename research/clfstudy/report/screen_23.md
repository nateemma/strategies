# Screen: phase 23

Label/input-variant screen (RIDGE + MLP_FAM w64, RIDGE/MLP_FAM only) against baseline phase 19's RIDGE pooled MCC = 0.161 (gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading edge (Constitution III).

Rule: PROMOTE if MLP - RIDGE >= 0.02 on >= 2 of 3 windows, OR ridge_shift >= +0.05 (a rise; a drop is noted, not promoted); else NO. A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE.

| variant | input_set | seq_len | RIDGE A | RIDGE B | RIDGE C | RIDGE pooled | MLP A | MLP B | MLP C | MLP pooled | delta A | delta B | delta C | ridge_shift | balance (S/H/B) | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tb_h12_u0.015_d0.015_cr1_pf0 | raw8 | 16 | 0.224 | 0.132 | 0.147 | 0.168 | 0.260 | 0.167 | 0.217 | 0.215 | 0.037 | 0.035 | 0.070 | 0.007 | 0.19/0.64/0.17 | PROMOTE | - |
| tb_h16_u0.015_d0.015_cr1_pf0 | raw8 | 16 | 0.220 | 0.115 | 0.137 | 0.157 | 0.254 | 0.156 | 0.211 | 0.207 | 0.035 | 0.040 | 0.074 | -0.004 | 0.24/0.56/0.21 | PROMOTE | - |
| tb_h24_u0.02_d0.02_cr1_pf0 | raw8 | 16 | 0.225 | 0.112 | 0.142 | 0.159 | 0.253 | 0.152 | 0.207 | 0.204 | 0.028 | 0.040 | 0.066 | -0.002 | 0.22/0.58/0.20 | PROMOTE | - |
| tb_h32_u0.025_d0.025_cr1_pf0 | raw8 | 16 | 0.227 | 0.109 | 0.147 | 0.161 | 0.251 | 0.148 | 0.196 | 0.198 | 0.024 | 0.040 | 0.048 | -0.000 | 0.20/0.61/0.19 | PROMOTE | - |
| tb_h48_u0.03_d0.03_cr1_pf0 | raw8 | 16 | 0.222 | 0.099 | 0.138 | 0.153 | 0.235 | 0.135 | 0.187 | 0.185 | 0.013 | 0.036 | 0.048 | -0.008 | 0.22/0.58/0.20 | PROMOTE | - |
| tb_h8_u0.01_d0.01_cr1_pf0 | raw8 | 16 | 0.220 | 0.128 | 0.148 | 0.165 | 0.249 | 0.154 | 0.209 | 0.204 | 0.029 | 0.026 | 0.060 | 0.004 | 0.23/0.59/0.19 | PROMOTE | - |
