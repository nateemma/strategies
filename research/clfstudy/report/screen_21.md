# Screen: phase 21

Label/input-variant screen (RIDGE + MLP_FAM w64, RIDGE/MLP_FAM only) against baseline phase 19's RIDGE pooled MCC = 0.161 (gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading edge (Constitution III).

Rule: PROMOTE if MLP - RIDGE >= 0.02 on >= 2 of 3 windows, OR ridge_shift >= +0.05 (a rise; a drop is noted, not promoted); else NO. A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE.

| variant | input_set | seq_len | RIDGE A | RIDGE B | RIDGE C | RIDGE pooled | MLP A | MLP B | MLP C | MLP pooled | delta A | delta B | delta C | ridge_shift | balance (S/H/B) | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tb_h24_u0.01_d0.01_cr1_pf0 | ind24 | 16 | 0.139 | 0.070 | 0.150 | 0.119 | 0.182 | 0.079 | 0.170 | 0.144 | 0.044 | 0.009 | 0.020 | -0.042 | 0.40/0.29/0.32 | PROMOTE | - |
| tb_h48_u0.01_d0.01_cr1_pf0 | ind24 | 16 | 0.044 | 0.049 | 0.082 | 0.059 | 0.096 | 0.056 | 0.123 | 0.091 | 0.052 | 0.006 | 0.041 | -0.103 | 0.47/0.16/0.37 | PROMOTE | label harder: large negative ridge_shift (not promoted) |
| tb_h24_u0.02_d0.02_cr1_pf0 | ind24 | 16 | 0.241 | 0.147 | 0.211 | 0.200 | 0.254 | 0.144 | 0.217 | 0.205 | 0.013 | -0.003 | 0.006 | 0.039 | 0.22/0.58/0.20 | NO | - |
| tb_h24_u0.03_d0.03_cr1_pf0 | ind24 | 16 | 0.208 | 0.154 | 0.190 | 0.184 | 0.225 | 0.146 | 0.197 | 0.189 | 0.018 | -0.008 | 0.007 | 0.023 | 0.11/0.77/0.11 | NO | - |
| tb_h48_u0.03_d0.03_cr1_pf0 | ind24 | 16 | 0.215 | 0.133 | 0.191 | 0.180 | 0.233 | 0.130 | 0.203 | 0.189 | 0.018 | -0.003 | 0.012 | 0.019 | 0.22/0.58/0.20 | NO | - |
| tb_h96_u0.01_d0.01_cr1_pf0 | ind24 | 16 | 0.054 | 0.066 | 0.090 | 0.070 | 0.044 | 0.065 | 0.097 | 0.069 | -0.010 | -0.001 | 0.008 | -0.091 | 0.50/0.11/0.39 | NO | label harder: large negative ridge_shift (not promoted) |
| tb_h96_u0.02_d0.02_cr1_pf0 | ind24 | 16 | 0.105 | 0.050 | 0.113 | 0.089 | 0.145 | 0.054 | 0.131 | 0.110 | 0.040 | 0.004 | 0.017 | -0.072 | 0.44/0.19/0.37 | NO | label harder: large negative ridge_shift (not promoted) |
| tb_h96_u0.03_d0.03_cr1_pf0 | ind24 | 16 | 0.190 | 0.094 | 0.161 | 0.148 | 0.206 | 0.094 | 0.164 | 0.155 | 0.016 | -0.000 | 0.003 | -0.013 | 0.34/0.36/0.30 | NO | - |
