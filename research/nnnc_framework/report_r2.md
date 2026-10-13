# NNNC framework study, round 2 — report (2026-10-13)

Same method as round 1 ([report.md](report.md)), re-screened on top of the adopted 24 h grace (`ed6049b`).
Pre-registration: `fb4f086` ([preregistration_r2.md](preregistration_r2.md)). Judge: `judge.py` with
`NNNC_FS_ROUND=2` → [verdicts_r2.json](verdicts_r2.json). **PROVISIONAL** (2023–26 halves only; the late
half is less fresh than round 1's — see the pre-registration caveat).

Controls (24 h grace): ARO `wf_aro_g24`, GBR `fs_gbr_gr`, GFM `fs_gfm_gr`, MLX `fs_mlx_gr`.

## Result: nothing adopted — the 24 h operating point holds

Cells = verdict, early / late gain in pp vs the model's 24 h control (sum of per-fold return).

| key | lever | ARO (arm D) | GBR | GFM | MLX | decision |
|---|---|---|---|---|---|---|
| TP | exit:take_profit = 0.04 | FAIL +7.26 / -3.27 | FAIL +3.25 / -3.01 | FAIL +6.94 / -3.92 | FAIL +3.45 / -5.80 | **NOT ADOPTED** |
| GR | exit:grace_hours = 48.0 | PASS +5.96 / +3.98 | FAIL +0.39 / -0.96 | FAIL -1.28 / -1.89 | FAIL -0.48 / +3.71 | **NOT ADOPTED** |
| TH | entry:thr_shift = -0.05 | FAIL +11.05 / -11.26 | FAIL +3.64 / -18.63 | FAIL +7.24 / -16.00 | PASS +0.19 / +0.22 | **NOT ADOPTED** |
| TK | exit:tp_atr_k = 6.0 | PASS +5.47 / +0.61 | FAIL -3.21 / +4.04 | FAIL +4.92 / -2.74 | FAIL +4.45 / -5.39 | **NOT ADOPTED** |
| SS | exit:stop_scale = 2.5 | FAIL +2.96 / -1.07 | FAIL +0.29 / -1.07 | FAIL -0.37 / -2.29 | FAIL -0.62 / -0.58 | **NOT ADOPTED** |

No lever passes ARO and 2 of the 3 replications, so no combination is run.

**Per lever:**
- **TP (fixed take-profit 2.2% → 4%)**: early +3 to +7 pp, late −3 to −6 pp on all four models, the same
  reversal as round 1. ARO exits: take-profit 694 → 318, ROI 33 → 136, unclog_12h 212 → 364. With the
  target at 4%, trades that reached 2.2% mostly end later at ROI or unclog, not at 4%. maxDD rises on
  every model (ARO 3.7 → 7.9%, MLX 2.3 → 5.8%), though the risk verdicts stay UNCHANGED.
- **GR (grace 24 h → 48 h)**: ARO passes both halves. Its paired verdict is **IMPROVED** (p 0.02), and
  the 15-fold total goes 49.97 → 59.91. It does not replicate. GBR and GFM are flat or negative in
  both halves, and MLX is −0.48 early and +3.71 late. Post-grace stops fall again (ARO 199 → 111,
  GBR 396 → 261), but GBR/GFM give the gain back through longer occupancy. Their screens predicted
  +6.2 / +5.4 early; the arms delivered +0.4 / −1.3. So 48 h is an ARO-specific gain, not a framework
  default.
- **TH (threshold −0.05)**: the same out-of-sample reversal as round 1, again on the three guards-open
  models (late −11 to −19 pp; trades ×4 to ×5; ARO and GFM risk WORSENED, maxDD 3.7 → 14.1% and
  6.6 → 24.6%). MLX's guards keep it inert. Two rounds now show that the guards-open models' extra
  sub-threshold signals are a 2023–24 artefact.
- **TK (ATR take-profit k = 6.0)**: passes ARO only. The early and late signs disagree on GBR, GFM and MLX.
- **SS (post-grace stop ×2.5)**: small and negative late on all four.

## Screen vs arm (reported only)

Late-half screen deltas, computed after the pre-registration ([late_screen_r2.json](late_screen_r2.json)),
next to the arms' late gains:

| lever | ARO screen / arm | GBR | GFM | MLX |
|---|---|---|---|---|
| TP 0.04 | -1.28 / -3.27 | +0.02 / -3.01 | -2.94 / -3.92 | (unscreened) |
| GR 48 h | +6.28 / +3.98 | +0.35 / -0.96 | +0.51 / -1.89 | (unscreened) |
| TH -0.05 | -10.93 / -11.26 | -18.94 / -18.63 | -16.66 / -16.00 | +0.22 / +0.22 |
| TK 6.0 | +0.75 / +0.61 | +6.37 / +4.04 | -2.17 / -2.74 | (unscreened) |
| SS 2.5 | -1.25 / -1.07 | -1.24 / -1.07 | -2.11 / -2.29 | (unscreened) |

The screen gets the late-half sign right in 13 of 16 cells. The 3 misses are all near-zero screen
values that the arm turns negative: TP/GBR, and GR on GBR and GFM. As in round 1:
- Entry and stop levers match almost exactly.
- Levers that lengthen holding (grace, wider targets) lose part of their replay gain to entry
  re-sequencing, which the replay cannot see.
- A grace or target screen therefore needs a margin: in two rounds the arm kept roughly
  0–0.7× of the screen.

## What the two rounds say together

1. **24 h grace is the one framework change that replicates across models.** Every other exit lever
   we screened sits at or near its optimum on these models. This covers take-profit (fixed and ATR),
   stop scale, unclog and grace length beyond 24 h.
2. **The wider-take-profit reversal holds in both rounds and on all four models.** The current 2.2%
   target is not leaving robust money on the table.
3. **Moving the threshold is regime-fragile.** The model's p_buy is tightly bunched (arm D median 0.735,
   max 0.81), so −0.05 multiplies trades by 4 to 5, and the added trades lost in 2025–26 in both rounds.
   This is a model output problem (calibration / separation), not a framework parameter to tune. It
   feeds directly into the queued probability-bunching diagnostic.
4. **ARO responds differently from the other models.** It is the only model where 48 h grace and ATR
   k 6.0 pass. Treat that as a per-model operating point if it is ever wanted, not a framework default.

## Tools

`late_screen.py` gains the `NNNC_FS_ROUND` suffix (reads `selection_r2.json`, writes `late_screen_r2.json`).
