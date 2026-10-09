# Screen: phase 26

Label/input-variant screen (RIDGE + MLP_FAM w64, RIDGE/MLP_FAM only) against baseline phase 19's RIDGE pooled MCC = 0.161 (gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading edge (Constitution III).

Rule: PROMOTE if MLP - RIDGE >= 0.02 on >= 2 of 3 windows, OR ridge_shift >= +0.05 (a rise; a drop is noted, not promoted); else NO. A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE.

| variant | input_set | seq_len | RIDGE A | RIDGE B | RIDGE C | RIDGE pooled | MLP A | MLP B | MLP C | MLP pooled | delta A | delta B | delta C | ridge_shift | balance (S/H/B) | verdict | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fmae_h12_dd0.1_g0.015_l0.015 | raw8 | 16 | 0.224 | 0.135 | 0.152 | 0.170 | 0.263 | 0.167 | 0.219 | 0.216 | 0.039 | 0.033 | 0.066 | 0.009 | 0.20/0.64/0.16 | PROMOTE | - |
