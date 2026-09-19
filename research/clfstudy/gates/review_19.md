# Gate-19 review (corrected): six representatives on the strict triple-barrier label, TRUE +/-2%, H 48

Classification metrics only (Constitution III); nothing here is evidence of trading edge. ind24 x seq_len 16, v1 windows A/B/C, lr 3e-4, stride 2, cap 40, patience 10. All 57 phase-19 rows carry label {triple_barrier, horizon 48, up 0.02, down 0.02}; wall time 55 min. (Reviewer memo delivered as a message and saved verbatim by the main session; the earlier version of this file scored the floored +3%/-2% run.)

## 0. What changed since the first memo (floored +3%/-2% run)

The first memo scored a label whose buy PT was +3% (sell mirror) because the shim only passed min_gain/min_loss (methods.md). Those rows are purged. On the true +/-2% label: HOLD share fell ~10 pp and BUY/SELL are now symmetric (test SELL/HOLD/BUY: A 28.8/54.6/16.5 -> 27.4/45.0/27.6; B 42.3/35.3/22.5 -> 39.6/26.0/34.4; C 36.6/47.5/15.9 -> 34.8/38.0/27.2). Every model lost ~0.01 pooled MCC (RIDGE 0.172 -> 0.161, MLP 0.176 -> 0.168, LSTM 0.177 -> 0.169, TRANSFORMER 0.167 -> 0.159, TCN 0.161 -> 0.153, Wavenet 0.161 -> 0.151) and PERSISTENCE 0.865 -> 0.850. The Wavenet SELL collapse (2% SELL predictions) is gone. Ordering, margins and the empty viable set are unchanged; the conclusions of the first memo survive the fix.

## 1. Sanity

- 0 failed rows of 57 (SC-001 met): 48 trained ok, 9 MAJORITY "collapsed" by construction. Train/test prior shift is large on A too: train HOLD 29.9% -> test 45.0% (+15 pp); B 35.6% -> 26.0% (-10 pp); C 32.6% -> 38.0%.
- PERSISTENCE 0.859 / 0.831 / 0.860 is label autocorrelation over a 48-bar horizon, not a model; nothing trained is within 0.6 of it. MAJORITY: MCC 0 everywhere; it predicts the TRAIN majority (SELL on A/C, HOLD on B).
- Seed std (2 seeds neural, 3 RIDGE): MCC <= 0.007 on every cell except TCN B (0.012). Between-config gaps are 2-5x the seed std except MLP vs LSTM (0.003: a tie).
- Several types peak in the first epochs (MLP A best_epoch 1-2; Wavenet B/C 0-6). RIDGE C seed 2 hit the cap (best 35/40), immaterial.
- Per class (neural, A/B/C): BUY prec/rec 0.36-0.39 / 0.31-0.43 vs base 27-34%; SELL 0.41-0.49 / 0.24-0.47 vs base 27-40%; HOLD is the easiest class (AUROC 0.72 / 0.63 / 0.68). No argmax collapse. Skews the gate does not see: RIDGE A predicts BUY 15% (true 28%); Wavenet B/C predicts BUY 11% / 17% (true 34% / 27%) and SELL 59% on B. ECE <= 0.09.

## 2. Leaderboard (seed-mean +/- std; pooled = mean of windows)

| type | params | MCC A | MCC B | MCC C | pooled MCC / F1 / AUROC |
|---|---|---|---|---|---|
| LSTM_FAM h32 d2 | 15,779 | 0.217 +/- .001 | 0.101 +/- .002 | 0.189 +/- .005 | 0.169 / 0.434 / 0.630 |
| MLP_FAM w64 | 28,995 | 0.220 +/- .001 | 0.098 +/- .006 | 0.187 +/- .001 | 0.168 / 0.434 / 0.630 |
| RIDGE wd 1e-2 | 1,155 | 0.202 +/- .002 | 0.102 +/- .001 | 0.179 +/- .001 | 0.161 / 0.422 / 0.617 |
| TRANSFORMER_FAM d32 | 18,243 | 0.207 +/- .006 | 0.091 +/- .001 | 0.178 +/- .000 | 0.159 / 0.426 / 0.625 |
| TCN_FAM ch64 | 29,571 | 0.208 +/- .004 | 0.084 +/- .012 | 0.168 +/- .003 | 0.153 / 0.421 / 0.621 |
| Wavenet | 16,227 | 0.208 +/- .007 | 0.088 +/- .005 | 0.158 +/- .002 | 0.151 / 0.402 / 0.624 |
| PERSISTENCE | 0 | 0.859 | 0.831 | 0.860 | 0.850 / 0.900 / 0.925 (tripwire) |
| MAJORITY | 0 | 0.000 | 0.000 | 0.000 | 0.000 / 0.151 / 0.500 |

Per-window rank: A MLP > LSTM > TCN ~ Wavenet ~ TRANSFORMER > RIDGE; B RIDGE > LSTM > MLP > TRANSFORMER > Wavenet > TCN; C LSTM > MLP > RIDGE > TRANSFORMER > TCN > Wavenet. RIDGE moves 6th -> 1st -> 3rd because the A and B fields sit inside 0.018.

## 3. FR-006 / SC-002 answers

- Beats RIDGE by >= 0.02 MCC and macro-F1 on a majority of windows: nobody. MCC margins (A/B/C): MLP +0.018/-0.004/+0.008; LSTM +0.015/-0.001/+0.009; TRANSFORMER +0.005/-0.010/-0.001; TCN +0.006/-0.018/-0.012; Wavenet +0.006/-0.013/-0.022. Every model is at or below RIDGE on B. Gate verdict (viable set empty) is correct as written.
- Pairs separated beyond 2x seed std with consistent sign on a majority of windows: LSTM > TRANSFORMER, LSTM > TCN, LSTM > Wavenet (3/3 each); MLP > TRANSFORMER/TCN/Wavenet (2/3). MLP vs LSTM: a tie. RIDGE vs MLP/LSTM: sign flips on B, not an ordering. Resolvable ordering: MLP ~ LSTM > TRANSFORMER > TCN ~ Wavenet, total spread 0.01-0.03, RIDGE inside the MLP/LSTM band. With 2 seeds the bar certifies reproducibility, not importance.

## 4. Learnability and linearity

- Ceiling under ind24 x 16: MCC 0.22 / 0.10 / 0.19; AUROC 0.67 / 0.59 / 0.64; balanced accuracy 0.46 / 0.40 / 0.45 vs 0.333. RIDGE holds 92% (A), 100% (B), 95% (C) of the best MCC (95% pooled) and 98% of the best AUROC: the argmax signal is linear in the 24 indicators to within 0.018. The non-linear gain is in ranking/calibration: neural AUROC +0.023/+0.004/+0.013 over RIDGE, logloss 0.02-0.03 lower.
- The gbb trigger rule scored on THIS label: 0.159 / 0.078 / 0.120 = 72% / 77% / 64% of the best model. The vol-state axis is still most of what ind24 carries.
- B is hardest on every metric; its test prior is touch-heavy (40/26/34 vs train 35/36/30). A has the larger prior shift yet is easiest, so the direction of the shift (toward barrier touches = a direction call) costs MCC, not its size. ATR-scaled barriers are the direct test.

## 5. Recommendation for the running screen (phases 20-22)

Promotion rule as amended: promote a variant only if MLP - RIDGE >= 0.02 MCC on 2/3 windows, or the RIDGE pooled MCC RISES by >= 0.05 vs 0.161 (a drop is "label harder": recorded, not promoted). The first ATR-1 rows show why: HOLD 14%, PERSISTENCE 0.64-0.69, RIDGE 0.06-0.08, trigger rule ~0 — a ~0.09 drop.
- Stage 1 (ATR scales, floor 0.5%): picture-changing if B rises to >= 0.15 at any scale, or MLP - RIDGE >= 0.02 on 2/3, or RIDGE up >= 0.05. Read the scale whose HOLD share is near 40% as the like-for-like cell. If all three leave RIDGE <= 0.16 and MLP - RIDGE < 0.02, the vol-regime explanation for B is refuted.
- Stage 1b (raw OHLCV): expected null. Worth a third seed only if RIDGE(raw) falls well below RIDGE(ind24) while MLP(raw) stays near 0.17 (the first non-linear recovery in the study).
- Stage 2 (horizon x width map): read as a map; MCC rising with horizon alongside PERSISTENCE is autocorrelation. A cell is interesting only if MCC/PERSISTENCE rises or MLP - RIDGE >= 0.02. Skip cells with any class < 5%.
- Stage 3 (seq_len): informative only if a sequence type pulls ahead of MLP anywhere in stages 1-2.
- Drop from any promoted grid: Wavenet and TCN (below RIDGE on 2/3 windows); keep TRANSFORMER as the single sequence control; keep LSTM in promoted grids, not in screens.

## 6. Open questions

1. Per-class recall floor for the gate (e.g. recall < 0.15 on a class with >= 5% share): Wavenet BUY recall 0.14 on C and RIDGE's 15% BUY share on A pass the current >= 99.5%-one-class rule.
2. gates/phase_19.md header wording ("fixed gbb labels") is cosmetic and should follow the leaderboard's label header.
