# Gate-4 review: phase 4 single-family sweeps (clean rows; lr 3e-4, stride 2)

Classification metrics against fixed gbb labels (ind24, seq_len 16, windows A/B/C, 2 seeds). Nothing here is evidence of trading edge (Constitution III). Sources: `results.jsonl` (last row per key; 35 configs of `configs/phase_4.yaml`), `gates/phase_4.json` (decided 2026-09-19 04:18 UTC, margin 0.020, RIDGE A 0.533 / B 0.535 / C 0.547), `report/methods.md` ("ROOT CAUSE", "The ceiling is the label's trigger rule"). Trigger-rule ceiling (learning-free, BUY/SELL if guard_metric on the right side of 0.3 and bb_width > 0.035): A 0.673 / B 0.644 / C 0.642.

**Retraction of `review_4_interim.md`.** Its B/C claims rested on rows warm-started from the window-A checkpoint (methods.md ROOT CAUSE): the "B/C plateau at 0.42-0.55", "only recurrent cells reach 0.64", the TCN receptive-field axis, the MLP width inversion, the stall counts, and the phase-5 hypotheses about B/C convergence are all withdrawn. What survives: Transformer size is flat and `pos=none` costs MCC; LSTM_FAM hidden 256 diverges on A.

## 1. Sanity

- Rows: 35 configs x 3 windows x 2 seeds = 210 keys; 208 `ok`, 2 `failed`. Every config has n_seeds = 2 on every window (LSTM_FAM h64 window-C seed 0 is the clean C-only ablation row of 01:16 UTC, 0.633; seed 1 re-run 0.635). Stale ids from the input-norm era (19 LSTM/GRU ids, e.g. `f12de4f7b017`) appear in the gate table as history and are ignored here.
- Failures: LSTM_FAM hidden 256 diverged on window A 2/2 under `input_norm: false` (non-finite probabilities at epochs 23/30, best 12/19) and 2/2 under the earlier default -> 4/4. Recorded as a result: diverges under the shared protocol on A; B/C fine (0.631 / 0.630).
- Stalls (best_epoch <= 6): 0 of 208. Min best_epoch A 7, B 18, C 14; runs hitting the 40-epoch cap: A 31, B 48, C 51 of ~70 per window. B/C now converge more often than A.
- Seed std (2 seeds): rms A 0.0024 / B 0.0015 / C 0.0013; max 0.010 (Transformer pos=none, A), everything else <= 0.0044. Between-config gaps inside a family are mostly < 0.01, i.e. most rungs are separated by less than 2x seed std.
- No leakage (everything below PERSISTENCE 0.80, which uses the true previous label) and no collapse: ceiling-bound configs predict ~11% SELL / 72% HOLD / 17% BUY against true shares ~4 / 90 / 6 (BUY recall 1.0, precision ~0.5 = the trigger rule's confusion).

## 2. Per family: which axis moves MCC (delta vs anchor, A/B/C; resolvable = beyond ~0.005 and same sign on all three windows)

- **TCN_FAM** anchor ch64/k3/d3 (RF 7): 0.650 / 0.634 / 0.637, 29.6k params, 1.8 min. ch16 -0.034/-0.032/-0.024 (resolvable); depth 2 (RF 5) -0.024/-0.024/-0.022 (resolvable); ch32 -0.011/-0.008/-0.004 (A only); ch256 -0.008/-0.003/-0.004, kernel 5 -0.009/-0.003/-0.004, depth 6 +0.005/0.000/+0.001, dil2/depth4 -0.002/+0.002/0.000 (all noise). Only starvation (16 channels or RF <= 5) moves it; nothing beats the anchor.
- **TRANSFORMER_FAM** anchor d64/h4/L2/ffn2/learned: 0.664 / 0.641 / 0.638, 69k, 6.3 min. pos=none -0.036/-0.022/-0.014 (resolvable, consistent); rope -0.004/-0.001/0.000; d32 -0.002/0.000/+0.001; d128 -0.003/-0.002/-0.002; L1 -0.002/-0.001/+0.001; L4 0.000/-0.001/+0.001; heads 8 -0.002/-0.001/+0.002; ffn4 -0.001/-0.001/+0.001. Size is flat from 18k to 270k params.
- **MLP_FAM** anchor w256/d2: 0.646 / 0.629 / 0.634, 165k, 1.1 min. w64 +0.005/+0.004/0.000 and w128 +0.003/+0.004/+0.001 (at 2x std, not on C); w1024 -0.007/-0.006/-0.006 (consistent sign, ~2x std); depth 1/3 within 0.002. Nothing decisive.
- **LSTM_FAM** anchor h64: 0.654 / 0.632 / 0.634, 23k, 3.0 min. depth 2 +0.003/+0.005/+0.003 (same sign everywhere, only B clears 2x std); h16 -0.002/-0.002/-0.002; h32 -0.004/-0.002/-0.002; bidir +0.001/-0.001/-0.001; h256 diverged on A.
- **GRU_FAM** anchor h64: 0.651 / 0.628 / 0.630, 17.5k, 3.1 min. depth 2 +0.007/+0.010/+0.007 (resolvable on all three windows; the largest reproducible effect in the phase, still <= 0.01); h32 0.000/+0.003/+0.004; h16 -0.005/0.000/+0.002; h256 -0.004/-0.003/-0.003; bidir +0.001/+0.001/+0.002.

Summary: across 35 configs the only effects above seed noise are removals (pos=none, TCN RF <= 5 or 16 channels) and RNN depth 2 (+0.003 to +0.010). Nothing lifts any family above its anchor by more than 0.01.

## 3. Cross-family vs the trigger-rule ceiling (MCC A / B / C; gap to ceiling 0.673 / 0.644 / 0.642)

| family | best rung | A / B / C | gap A / B / C | params | min/run | cheapest rung within 0.01 of the ceiling on B/C |
|---|---|---|---|---|---|---|
| TRANSFORMER_FAM | d64 anchor (= L4, d32, h8, ffn4) | 0.664 / 0.641 / 0.638 | -0.009 / -0.003 / -0.004 | 69k | 6.3 | d64/L1: 0.662 / 0.640 / 0.639, 36k, 3.0 min (d32: 18k but 5.8 min) |
| GRU_FAM | depth 2 | 0.659 / 0.637 / 0.637 | -0.015 / -0.007 / -0.005 | 42k | 6.1 | h32: 0.651 / 0.631 / 0.634, 5.7k, 3.4 min |
| LSTM_FAM | depth 2 | 0.657 / 0.638 / 0.637 | -0.016 / -0.006 / -0.005 | 56k | 5.8 | h16: 0.653 / 0.630 / 0.633, 2.7k, 2.8 min |
| TCN_FAM | depth 6 | 0.656 / 0.635 / 0.638 | -0.017 / -0.009 / -0.004 | 67k | 2.6 | ch64/k3/d3: 0.650 / 0.634 / 0.637, 29.6k, 1.8 min |
| MLP_FAM | w128 | 0.649 / 0.633 / 0.635 | -0.024 / -0.011 / -0.007 | 66k | 1.05 | w64: 0.651 / 0.632 / 0.633, 29k, 1.0 min |

Every family converges to the trigger-rule ceiling and none exceeds it on any window: best rungs sit 0.003-0.011 under it on B/C and 0.009-0.024 under it on A. The whole cross-family spread (0.013 on A, 0.008 on B/C) is of the same order as the gap to the rule itself. The families rank Transformer > RNN depth 2 > TCN ~ MLP, and that order is stable across windows and above 2-seed std, but it is a ranking of how exactly each family reproduces an AND of two thresholds on the last bar, not of predictive content. The RIDGE reference (0.533 / 0.535 / 0.547) is the linear approximation of the same rule; phase-2 ROCKET (0.516 / 0.526 / 0.45) is below it.

## 4. FR-028 gate as written (`gates/phase_4.json`)

- Rule: margin_used 0.020 (floor; RIDGE pooled seed std 0.002 after quarantining the warm-started seed-2 B/C RIDGE rows), std_bound 0.03 on every window, majority of 3, no failed/collapsed window.
- Result: 34 of the 35 phase-4 configs viable; 0 stability-only failures. The one fail is LSTM_FAM h256 (`b5b58a6c2cc5`, failed on A). Margins are +0.08 to +0.13 MCC and +0.05 to +0.09 macro-F1 on every window; the smallest is TCN ch16 (A +0.083 / +0.054). The gate's viable list has 35 entries because it also passes `f12de4f7b017` (LSTM_FAM h64, superseded input-norm id) — stale history, not a phase-4 config.
- What the rule cannot do: it passes TCN ch16 (0.616 / 0.603 / 0.613) and Transformer d64 (0.664 / 0.641 / 0.638) with the same verdict, because every neural config is 0.07-0.13 over RIDGE and 0.00-0.05 under the trigger rule. As a viability screen it is correct; as a selector among ceiling-bound configs it has no resolution. Proposed amendment (record, do not bend the margin): keep FR-028 for viability and add, as an author decision in `phase_4.md`, (i) a "ceiling-bound" tag for any rung within 0.01 of the trigger rule on B/C, and (ii) the trigger-conditional metrics of section 5 as the metrics phases 5-7 are judged on; whole-window MCC stays reported but is no longer the gate metric once every candidate is ceiling-bound.

## 5. Trigger-conditional metrics (rows written after ~03:00 UTC 2026-09-19)

- Coverage: 23 phase-4 rows carry `trig_*` (MLP_FAM 5 configs x B/C x 2 seeds = 20; Transformer rope 3), all on B/C, none on A; plus 3 phase-0 smoke rows (ignored). `trig_rule_mcc` reproduces the learning-free rule exactly (B 0.645, C 0.642).
- Instrument anomaly: Transformer rope B seed 1 (`3413d77700b0`, 03:14 UTC, first row under sha 52fddd9) records trig_buy_n 714 and trig_rule_mcc 0.376 on the same 173,760-row test set where every later B row has trig_buy_n 35,867 / 0.645 (trig_sell_n 23,746 matches). Exclude that row; the 3 Transformer rows say nothing about the family yet.
- Within the BUY trigger (n 35,867 on B, 31,349 on C; base rate 0.507 / 0.454): argmax precision lift over the base rate is +0.001 to +0.008 on every row (max MLP w1024 on C, 0.462 vs 0.454); 0 of 22 rows reach +0.02. trig_buy_mcc 0.02-0.07. BUY-class AUROC within the trigger: 0.566-0.592 on B and 0.600-0.626 on C (MLP), 21 of 22 clean rows > 0.55, seed-to-seed agreement ~0.01, no monotone size trend (w1024 has the highest trig_buy_mcc and the lowest AUROC). SELL side: AUROC 0.53-0.57, precision lift <= +0.003.
- Reading: the models' BUY probability orders triggered bars a little better than chance (AUROC ~0.6 on C, reproducible across 5 MLP configs and both seeds) but the argmax decision never uses it — every triggered bar is called BUY (recall 1.0, precision = base rate). Whether AUROC 0.6 converts into precision at reduced coverage (e.g. the top 20% of triggered bars by p_buy) is unmeasured; that number, not MCC, is the one that would show information beyond the trigger.

## 6. Options for phases 5-7 given the ceiling

Architecture is closed: five families, 35 configs, 2.7k-1.4M params, all within 0.01 of a two-threshold rule on B/C. Whole-window MCC/macro-F1 cannot produce information from here, so each remaining phase is worth running only if judged on trigger-conditional metrics.

- **Phase 5 (compositions).** 5a: skip; record "considered, not run: every component is ceiling-bound and no composition can exceed a rule its components already reproduce" (0 h; FR-030 is satisfied vacuously, FR-027 still needs the recorded gate). 5b: run the two cheapest compositions (CNN_TRANSFORMER on d64/L1, RNN_ATTNPOOL on GRU h32), B/C only, with `trig_*` as the primary readout; ~1 h; informative only if within-trigger BUY AUROC or precision@coverage moves >= 0.03 vs the component. 5c: full grid as spec'd (~2.5 h at 3-6 min/run); expected null on MCC, the trig readout is the only possible signal.
- **Phase 6 (literature).** Same logic. The FR-031 obligations (citations, applicability, rank, "considered, not tested" reasons) are met by the review without runs. If run: TSMIXER / ITRANSFORMER / XLSTM / TRANSFORMER_MODERN at d32-equivalent sizes, B/C, trig readout, ~2 h. Expected: MCC 0.64, trig AUROC unknown — that is the one thing worth learning.
- **Phase 7 (inputs) — where the information is.** Measure, per row: (i) within-trigger BUY AUROC and precision at a fixed coverage (fix the coverage before running, e.g. top 20% of triggered bars) against the trigger base rate; (ii) whole-window MCC as secondary. Arms: raw8 vs ind24 vs both32 — raw8 cannot see guard_metric/bb_width, so its whole-window MCC will fall to RIDGE level or below and only the trigger-conditional view is a fair comparison; `ind24-minus:guards` and `-minus:volatility` answer R11's "does the model need to see the trigger" directly; seq_len 16/32/64/128 answers whether more history moves within-trigger AUROC (window length was never tested at the ceiling). Cost: MLP w64 (1.0 min) and TCN ch64 (1.8 min) make the ~102-run grid 2-3 h per representative; Transformer d64/L1 (3.0 min) ~5 h; an RNN h16/h32 ~5 h.
- **Families and representatives to carry.** All five pass FR-028. Cheapest ceiling-reaching representatives: MLP_FAM w64 (29k, 1.0 min), TCN_FAM ch64/k3/d3 (29.6k, 1.8 min), TRANSFORMER_FAM d64/L1 (36k, 3.0 min, keep `pos=learned`), GRU_FAM h32 (5.7k, 3.4 min) or LSTM_FAM h16 (2.7k, 2.8 min). Best-metric references if one is wanted: Transformer d64 anchor, GRU h64 depth 2. Drop from further sweeps: any size axis (flat everywhere), bidirectional (inert), TCN kernel/dilation (inert), h256 (diverges or is worse), MLP w1024 (worse, 1.4M params). For phase 7 three representatives (MLP w64, TCN ch64, Transformer L1) cover distinct inductive biases at ~10 h total; adding an RNN is another ~5 h for a family that behaves like the others.

## 7. Open questions

1. Amend FR-028 formally (ceiling-bound tag + trigger-conditional gate metrics) or keep it as written and record the selection rule for phases 5-7 as an author decision in `gates/phase_4.md`?
2. `trig_*` is computed only at training time and rows do not save probabilities (`weights_path` null), so the 187 phase-4 rows without it cannot be filled in without re-running. Add a save-probabilities option before phase 7 so within-trigger precision@coverage can be recomputed offline at any coverage?
3. Fix the trig_buy_n 714 anomaly (rope B/1) before phase 5/6/7 rows inherit it; is it a mask bug in the first row of the sha-52fddd9 build or a real difference in how that row's trigger was evaluated?
4. Which of 5a/5b/5c and whether phase 6 runs at all — the study budget (29.8 h of 40 used at the interim memo, plus the clean re-runs since) will not cover 5c + 6 + a three-representative phase 7.
