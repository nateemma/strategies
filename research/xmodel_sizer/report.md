# Cross-model sizers: results (2026-10-17)

Pre-registration `preregistration.json` (2127da0a). Width-book entries; stakes 1.333× / 0.667× on the
model's agree score; thresholds at the 0.5 share of validation-tail run starts. Judge = judge_one PASS
vs the model's width book AND its framework control. Seed 1, 15 folds.

| Arm | Total pp | Width book | Control | Decision | Secondary |
|---|---|---|---|---|---|
| cs_gbr (GBR p_buy) | 58.33 | w_gbr 53.68 (+3.08 / +1.57; paired UNCHANGED p 0.06; Sharpe WORSENED) | fs_gbr_gr 53.46 PASS | **FAIL** (risk vs width) | — |
| cs_mlx (MLX p_buy) | — | — | — | **INCONCLUSIVE** (instrument) | — |
| cs_gbr_mix (GBR + composite) | 60.21 | +3.49 / +3.04; paired IMPROVED p 0.007 | +2.27 / +4.48 PASS | **PASS (PROVISIONAL)** | vs cs_gbr +0.42 / +1.47 PASS |
| cs_mlx_mix (MLX + composite) | 39.28 | w_mlx 34.10 (+3.69 / +1.49; paired IMPROVED p 0.013) | fs_mlx_gr 35.19 PASS | **PASS (PROVISIONAL)** | vs cs_mlx +1.61 / +0.18 PASS |

- cs_gbr's f00–f07 thresholds reproduce 035's exactly (0.7162 … 0.7605).
- cs_mlx instrument: the late half has 3 folds (f10, f11, f12; 1–4 swapped trades each) whose trade set
  differs from w_mlx, above the pre-registered hash-diff limit; no P&L read. Every other check passed.
  cs_mlx_mix's secondary compares against that run anyway, so read it with the caveat.
- Mixing the composite in helps both models in both halves; the same pattern as ARO (cs_mix vs cs_aro
  −0.58 / +1.70 was the exception, early).
