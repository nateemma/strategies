# Gate-4 interim review: phase 4 single-family sweeps (lr 3e-4, stride 2)

Classification metrics against fixed gbb labels (ind24, seq_len 16, windows A=2023-H2 / B=2025-H1 / C=2026-H1, 2 seeds). Nothing here is evidence of trading edge (Constitution III). Source: `results.jsonl` (470 rows, last line per key), `report/logs/phase_4c.log`, `protocol_calib*.log`. Rows judged: phase 4, `train.lr == 3e-4`; the 51 lr 1e-4 rows are the aborted first attempt and are ignored. **RNN part is PROVISIONAL**: only LSTM_FAM rows carrying `input_norm: false` count (13 rows), GRU_FAM has no valid rows yet, and the rnn/tcn ablations are still executing.

## 1. Sanity

- TCN_FAM 42 + TRANSFORMER_FAM 54 + MLP_FAM 30 = 126 runs, 0 failed, 0 NaN, epochs_run <= 40 everywhere. LSTM_FAM hidden 256: 2/2 window-A seeds `failed` ("non-finite values in predicted probabilities" at epochs 23/30) under the corrected default too -> diverges under the shared protocol.
- No leakage: everything sits under PERSISTENCE (0.80); A saturates at 0.660-0.664 across 7 unrelated Transformer rungs. No collapse: every config predicts 5-8% SELL / 80-85% HOLD / 10-13% BUY. 2-seed seed-std: A <= 0.004 (rms 0.002), B/C rms 0.015-0.020 per family (max 0.041).
- Stalls (best_epoch <= 6) / runs: TCN A 0/14, B 11/14, C 9/14; TRANSFORMER A 0/18, B 17/18, C 17/18; MLP A 0/10, B 9/10, C 5/10; LSTM_FAM (prov.) A 1/6, B 4/4, C 1/3. On A 17 runs hit the 40-epoch cap (12 of them TCN); on B/C **zero** runs of any family reached the cap.
- Stronger than "stall": no B/C run of the three completed families exceeded 0.551 (C, Transformer L4 seed 0, best_epoch 2). Across every phase, >110 B/C runs of attention / TCN / MLP / CNN / Multihead / TSMamba types max at 0.551, while 0.62-0.64 has been reached only by recurrent cells (LSTM*, GRU, Mamba2; ~25 runs) and Wavenet (1 run), almost all with best_epoch >= 30.
- Validation traces make the difference qualitative, not a patience lottery: production LSTM on C (calibration) starts at val_mcc 0.61 at epoch 1 and climbs to 0.644; Transformer anchor on C seed 0 goes 0.47 0.41 0.52 0.44 0.43 0.45 0.44 0.45 0.49 0.41 0.46 0.47 0.46 (no drift, +/-0.08 swings; val set is 154k rows so this is not sample noise); TCN dil2 on C seed 1 ran 22 epochs at 0.44-0.51 flat. The same configs climb smoothly 0.56 -> 0.605 on A. Read: on B/C these families sit on a plateau from epoch 1; the lr 3e-4 calibration (production LSTM only) did not transfer to them.

## 2. Per family: which axis moves MCC (delta vs anchor, A/B/C; A noise 0.002-0.004, B/C ~0.017 on a difference of 2-seed means)

**TCN_FAM** anchor ch64 k3 d3 dil1 (RF 7 < 16): 0.650 / 0.479 / 0.487, 29.6k, 1.4 min.
- Receptive field is the axis, monotone: depth 2 (RF 5) -0.024/-0.031/-0.048; ch16 -0.034/-0.050/-0.070; kernel 5 (RF 13) -0.008/+0.025/+0.013; depth 6 (RF 13) +0.006/+0.041/+0.016; dil2 depth 4 (RF 31, only rung covering the window) -0.001/+0.035/+0.031. B/C gains are ~2 sigma on 2 seeds.
- Channels 256 (413k, 3.5 min): -0.008/-0.010/+0.022 -> width is inert. Best: **dil2/depth4** (41.9k, 1.33 min, pooled 0.560) = depth 6 (66.6k, 1.39 min, 0.560); carry dil2/depth4.

**TRANSFORMER_FAM** anchor d64 h4 L2 ffn2 learned-pos: 0.664 / 0.486 / 0.513, 69k, 2.7 min.
- Only resolvable axis: pos=none -0.037 on A (std 0.010) and -0.028/-0.036 on B/C. Positional information is load-bearing; rope ties learned on A (-0.004) and is worse-or-noise on B/C (-0.025/-0.023).
- Size is flat: d32 (18k, 2.2 min) 0.662/0.499/0.515; L1 (36k, 1.4 min) 0.662/0.504/0.511; L4 (136k, 4.5 min) 0.664/0.498/0.534; d128 (270k, 3.6 min) 0.661/0.460/0.486; ffn4 0.663/0.469/0.491; heads 8 0.662/0.458/0.498. Bigger is never better; d128/ffn4/heads8 are 0.02-0.03 lower on B/C (~1.5 sigma).
- Best by pooled is L4 (0.565) on one C seed at best_epoch 2; **d64/L1** (0.559) at 1/4 the params and 1/3 the time is the cost-efficient anchor (d32 is equivalent).

**MLP_FAM** anchor w256 d2: 0.646 / 0.469 / 0.505, 165k, 0.6 min.
- Width is inverted: **w64** (29k, 0.53 min) 0.651/0.503/0.526 best on all 3 windows (+0.005 A resolvable, +0.034 B, +0.021 C); w1024 (1.45M) 0.639/0.488/0.500. depth 1 / 3 = anchor within 0.006.
- MLP w64 (pooled 0.560) ties TCN-best and beats phase-3 MLP (0.528): the protocol change alone moved MLP +0.03.

**LSTM_FAM** (provisional, `input_norm: false`): h16 (2.7k, 1.5 min) 0.652/0.470/0.488; h64 (23k) A 0.593 +/- 0.085 (seed 0 stalled at best_epoch 1 -> 0.534, the first window-A stall in the study), B 0.479 +/- 0.031, C one seed 0.505 (the phase-0 diagnostic of the identical config scored 0.633 on C); h256 diverged 2/2. Nothing decidable yet; GRU_FAM has no rows.

## 3. Cross-family vs phase-3 production types (MCC A / B / C = pooled; params; min/run)

| config | A | B | C | pooled | params | min |
|---|---|---|---|---|---|---|
| Transformer prod (ph3, lr1e-4) | 0.664 | 0.503 | 0.526 | 0.564 | 15.6k | 5.0 |
| Wavenet prod (ph3) | 0.659 | 0.497 | 0.525 (3-seed 0.563) | 0.561 | 16.2k | 3.6 |
| LSTM prod (ph3) | 0.664 | 0.493 | 0.517 (3-seed 0.558) | 0.558 | 12.5k | 3.9 |
| LSTM prod, converged (calibration, lr3e-4, C x3) | - | - | 0.640 +/- 0.000 | - | 12.5k | 5.5 |
| TRANSFORMER_FAM L4 | 0.664 | 0.498 | 0.534 | 0.565 | 136k | 4.5 |
| TRANSFORMER_FAM L1 | 0.662 | 0.504 | 0.511 | 0.559 | 36k | 1.4 |
| TCN_FAM dil2/d4 | 0.649 | 0.514 | 0.518 | 0.560 | 42k | 1.3 |
| TCN_FAM d6 | 0.656 | 0.520 | 0.503 | 0.560 | 67k | 1.4 |
| MLP_FAM w64 | 0.651 | 0.503 | 0.526 | 0.560 | 29k | 0.5 |
| LSTM_FAM h16 (prov.) | 0.652 | 0.470 | 0.488 | 0.537 | 2.7k | 1.5 |

Verdict: on **A** the Transformer rungs tie the production ceiling (0.660-0.664); TCN-best 0.656 and MLP 0.651 are 0.008-0.013 under it at std 0.002 (resolvable, small). On **B/C no phase-4 config exceeds the production types**: best B 0.520 vs 0.503 (+0.017, ~1 sigma), best C 0.534 vs 0.526 (tie), and every one of them is 0.10-0.14 below the converged production LSTM (0.640 on C). Phase 4 therefore ties on A and ties at the stalled-plateau level on B/C; the sweep never touched the value the recurrent cell reaches. Pooled leaderboard order (0.565 vs 0.559 vs 0.560) is inside seed noise (per-seed pooled values differ by 0.003-0.028 within a config); the top 8 pooled are one tie.

## 4. FR-028 as written (margin 0.02 over RIDGE on MCC and macro-F1, majority of windows; seed-std <= 0.03 on every window; no collapsed/failed)

RIDGE MCC 0.533 / 0.307 / 0.299, macro-F1 0.656 / 0.517 / 0.508; margin_used 0.02.
- Margin: all 21 completed configs clear both margins on all 3 windows (smallest: TCN ch16 A +0.083 MCC / +0.054 F1). The clause discriminates nothing, as in gate 3.
- Stability: **PASS 19/21**. Stability-only fails: TRANSFORMER d32 (B std 0.041; seeds 0.528/0.470), TRANSFORMER rope (C std 0.032; 0.513/0.467). All others <= 0.028.
- Provisional RNN: LSTM_FAM h16 PASS; h64 stability-only (A 0.085, B 0.031; C has 1 seed); h256 FAIL (status failed x2). GRU_FAM: no rows.
- Family level: TCN_FAM 7/7, MLP_FAM 5/5, TRANSFORMER_FAM 7/9 pass (+2 stability-only). All three completed families are viable as written.
- Miscalibration: on B/C the bar is 0.33 while 0.64 is demonstrably attainable and every neural config sits at 0.42-0.55; the rule passes TCN ch16 (0.417 on C) and cannot tell plateau from converged. Proposed amendment (recorded, not silent): keep FR-028 for viability, but a family is carried as a phase-5 *base* only if at least one B/C seed reaches >= 0.60 OR a patience-40 diagnostic on that family shows a rising validation curve. Re-referencing the B/C margin to the converged LSTM (0.64 minus a tolerance) would fail every family, which is informative but not usable.

## 5. Phase-5 families and anchors

- Evidence: the only components that have ever reached 0.64 on B/C are recurrent cells (and Wavenet once, at lr 1e-4). Compositions with no recurrent part (TCN x Transformer, INCEPTION_MS alone) have no evidence of escaping the plateau. Order phase 5 so each composition tests one hypothesis about B/C convergence:
  1. **CNN_TRANSFORMER**: does a conv front-end give the Transformer what the production LSTM has on B/C? Cheapest test of the hypothesis (Transformer d64/L1 base).
  2. **RNN_ATTNPOOL**: does an attention readout preserve the recurrent B/C convergence? (production LSTM or LSTM_FAM h64 base, whichever the re-run supports).
  3. **CNN_LSTM**: control; the production LSTM already is one and converges.
  4. **TCN_RES** (+ `residual` on/off on TCN_FAM dil2/d4): conditional on the queued `tcn_ablation` (depth-8/dil-2 vs Wavenet control, 3 seeds on C). If Wavenet converges at lr 3e-4 and TCN depth 8 does not, gating/skips are the difference and TCN_RES is the test; if neither converges, drop TCN as a base.
  5. **INCEPTION_MS**: last; kernel 5 vs 3 and channel width were inert in phase 4.
- Anchors to carry: TRANSFORMER_FAM d64/L1 (36k, 1.4 min; pos=learned mandatory, d32 equivalent); TCN_FAM ch64/depth4/dil2 (42k, 1.3 min; RF >= seq_len is the one structural requirement); MLP_FAM w64 (29k, 0.5 min) as reference only (nothing sequential to compose); LSTM_FAM hidden 64 (h16 ties it, h256 diverges) pending the re-run.
- Cost: phase 4 at lr 3e-4 = 7.3 h (TF 2.8, LSTM 1.8, GRU 1.3, TCN 1.0, MLP 0.3) + 2.9 h aborted; study total 29.8 h of 40. Phase 5 at ~3 min/run x 6 x 5 compositions is ~1.5 h; the remaining budget should buy the diagnostics below, not more rungs.
- Wanted from the RNN re-run before finalising: (i) 3 seeds on B/C for LSTM_FAM h64 (C has 1 seed and disagrees with the phase-0 diagnostic by 0.13); (ii) B/C stall counts under the corrected default: if LSTM_FAM converges on C in fewer than 2/3 seeds, the "production LSTM 3/3" calibration does not transfer to the pure family and the phase-5 recurrent base should be the production LSTM; (iii) GRU_FAM anchor rows; (iv) h256 recorded as "diverged under the shared protocol" per methods.md or explained; (v) the tcn_ablation result.

## 6. Open questions

1. Readout: TCN_FAM and TRANSFORMER_FAM read out by global average pool over 16 positions, the label is a last-bar trigger, and review-3 found mean-pool the one structural change that cost the LSTM on A (-0.021). A `pool: last` rung on both anchors (6 runs, ~15 min) would show whether the B/C plateau is partly a readout artefact before phase-5 compositions inherit it (MLP has direct last-bar access and still plateaus, so it is not the whole story).
2. Patience-40 on one Transformer and one TCN config on C (3 seeds, ~25 min): settles "never converges" as protocol vs architecture. Given the flat traces I expect architecture, but it is a 25-minute fact.
3. Record the section-4 ceiling clause in the gate, or keep FR-028 as written and document the phase-5 base selection as an author decision in phase_4.md?
