# Screen: phase 20

Label/input-variant screen (RIDGE + MLP_FAM w64, RIDGE/MLP_FAM only) against baseline phase 19's RIDGE pooled MCC = 0.161 (gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading edge (Constitution III).

Rule: PROMOTE if MLP - RIDGE >= 0.02 on >= 2 of 3 windows, OR ridge_shift >= +0.05 (a rise; a drop is noted, not promoted); else NO. A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE.

| variant | input_set | seq_len | RIDGE A | RIDGE B | RIDGE C | RIDGE pooled | MLP A | MLP B | MLP C | MLP pooled | delta A | delta B | delta C | ridge_shift | balance (S/H/B) | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tb_h48_u0.02_d0.02_cr1_pf0 | raw8 | 16 | 0.188 | 0.067 | 0.103 | 0.119 | 0.213 | 0.104 | 0.173 | 0.163 | 0.025 | 0.037 | 0.069 | -0.042 | 0.34/0.36/0.30 | PROMOTE | - |
| tb_h48_u0.005_d0.005_cr1_pf0_atr1 | ind24 | 16 | 0.075 | 0.061 | 0.101 | 0.079 | 0.074 | 0.070 | 0.105 | 0.083 | -0.002 | 0.009 | 0.004 | -0.082 | 0.50/0.14/0.36 | NO | label harder: large negative ridge_shift (not promoted) |
| tb_h48_u0.005_d0.005_cr1_pf0_atr2 | ind24 | 16 | 0.059 | 0.050 | 0.096 | 0.069 | 0.056 | 0.046 | 0.095 | 0.066 | -0.004 | -0.004 | -0.001 | -0.092 | 0.50/0.13/0.38 | NO | label harder: large negative ridge_shift (not promoted) |
| tb_h48_u0.005_d0.005_cr1_pf0_atr3 | ind24 | 16 | 0.084 | 0.057 | 0.103 | 0.081 | 0.084 | 0.070 | 0.111 | 0.088 | 0.001 | 0.012 | 0.008 | -0.080 | 0.47/0.14/0.38 | NO | label harder: large negative ridge_shift (not promoted) |
