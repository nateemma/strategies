# Screen: phase 24

Label/input-variant screen (RIDGE + MLP_FAM w64, RIDGE/MLP_FAM only) against baseline phase 19's RIDGE pooled MCC = 0.161 (gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading edge (Constitution III).

Rule: PROMOTE if MLP - RIDGE >= 0.02 on >= 2 of 3 windows, OR ridge_shift >= +0.05 (a rise; a drop is noted, not promoted); else NO. A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE.

| variant | input_set | seq_len | RIDGE A | RIDGE B | RIDGE C | RIDGE pooled | MLP A | MLP B | MLP C | MLP pooled | delta A | delta B | delta C | ridge_shift | balance (S/H/B) | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fmae_h12_dd0.03_g0.0125_l0.0125 | raw8 | 16 | 0.191 | 0.111 | 0.120 | 0.141 | 0.238 | 0.152 | 0.210 | 0.200 | 0.048 | 0.040 | 0.089 | -0.020 | 0.22/0.57/0.20 | PROMOTE | - |
| fmae_h12_dd0.03_g0.015_l0.015 | raw8 | 16 | 0.195 | 0.123 | 0.124 | 0.147 | 0.235 | 0.157 | 0.209 | 0.200 | 0.041 | 0.034 | 0.085 | -0.014 | 0.18/0.65/0.17 | PROMOTE | - |
| fmae_h12_dd0.05_g0.015_l0.015 | raw8 | 16 | 0.209 | 0.125 | 0.136 | 0.156 | 0.248 | 0.167 | 0.219 | 0.211 | 0.039 | 0.042 | 0.083 | -0.005 | 0.19/0.64/0.17 | PROMOTE | - |
| fmae_h16_dd0.03_g0.015_l0.015 | raw8 | 16 | 0.179 | 0.115 | 0.112 | 0.135 | 0.233 | 0.153 | 0.203 | 0.197 | 0.054 | 0.039 | 0.091 | -0.026 | 0.21/0.58/0.21 | PROMOTE | - |
| qfut_h12_g0.001_l0.001_q0.75 | raw8 | 16 | 0.231 | 0.143 | 0.133 | 0.169 | 0.245 | 0.149 | 0.187 | 0.193 | 0.014 | 0.006 | 0.053 | 0.008 | 0.24/0.57/0.19 | NO | - |
| qfut_h12_g0.001_l0.001_q0.8 | raw8 | 16 | 0.231 | 0.154 | 0.133 | 0.173 | 0.238 | 0.163 | 0.188 | 0.196 | 0.007 | 0.009 | 0.054 | 0.012 | 0.19/0.65/0.16 | NO | - |
