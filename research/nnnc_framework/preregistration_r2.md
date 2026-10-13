# Pre-registration: NNNC framework study, round 2 (2026-10-13)

**Committed before any round-2 arm backtest.** Machine-readable: [preregistration_r2.json](preregistration_r2.json);
`NNNC_FS_ROUND=2 judge.py` reads only that file.

Same method and rule as round 1 ([preregistration.md](preregistration.md)), on top of the adopted 24 h grace.
Controls: `wf_aro_g24` (arm D under the new default, = `fs_aro_gr` 15/15), `fs_gbr_gr`, `fs_gfm_gr`, `fs_mlx_gr`.
Screens and selection rule committed in `e1ad10c` before any result; MLX again not screened for exits
(replay fidelity 0.831: its model-exit-signal trades).

| key | lever | value | early screen sum (pp) |
|---|---|---|---|
| TP | fixed take-profit | 0.022 → 0.04 | +26.79 |
| GR | grace window | 24 h → 48 h | +22.31 |
| TH | prediction threshold | fold threshold − 0.05 | +21.97 |
| TK | ATR take-profit multiple | off → 6.0 | +21.73 |
| SS | post-grace ATR stop (K, floor, cap) | × 2.5 | +5.40 |

Arms `fs2_<m>_<key>` (classes `NNNC_FS2_<M>_<KEY>`), base model's own per-fold weights (all 20 keys verified).
**ADOPT** = PASS on ARO and ≥ 2 of GBR, GFM, MLX; PASS = early and late gain > 0, paired and risk not WORSENED.

**Caveat:** the late half is less fresh than in round 1 — it was used to adopt 24 h grace, and TP, TH and TK
failed it in round 1 at 6.5 h grace. GR 48 h, TK 6.0 and SS × 2.5 are grid edges.
