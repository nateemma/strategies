# Screen: phase 22

Label/input-variant screen (RIDGE + MLP_FAM w64, RIDGE/MLP_FAM only) against baseline phase 19's RIDGE pooled MCC = 0.161 (gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading edge (Constitution III).

Rule: PROMOTE if MLP - RIDGE >= 0.02 on >= 2 of 3 windows, OR ridge_shift >= +0.05 (a rise; a drop is noted, not promoted); else NO. A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE.

| variant | input_set | seq_len | RIDGE A | RIDGE B | RIDGE C | RIDGE pooled | MLP A | MLP B | MLP C | MLP pooled | delta A | delta B | delta C | ridge_shift | balance (S/H/B) | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tb_h48_u0.02_d0.02_cr1_pf0 | ind24 | 32 | 0.203 | 0.099 | 0.180 | 0.160 | 0.219 | 0.099 | 0.193 | 0.170 | 0.017 | 0.000 | 0.013 | -0.001 | 0.34/0.36/0.30 | NO | - |
| tb_h48_u0.02_d0.02_cr1_pf0 | ind24 | 64 | 0.197 | 0.097 | 0.183 | 0.159 | 0.218 | 0.090 | 0.188 | 0.165 | 0.021 | -0.007 | 0.005 | -0.002 | 0.34/0.36/0.30 | NO | - |
